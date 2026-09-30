from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.versions.similarity import (
    VersionMatch, VersionSimilarityMatcher, VersionSimilarityWeights,
)
from app.versions.version_manager import VersionRecord
from app.versions.version_search import VersionSearchEngine, VersionSearchResult


@dataclass(slots=True)
class EvolutionTransition:
    from_version: str
    to_version: str
    from_chunk_id: str
    to_chunk_id: str
    from_symbol: str
    to_symbol: str
    from_path: str
    to_path: str
    similarity_score: float
    change: str


@dataclass(slots=True)
class EvolutionItem:
    version_id: str
    version_label: str
    commit_hash: str
    chunk_id: str
    symbol_name: str
    file_path: str
    start_line: int
    end_line: int
    relevance_score: float
    change: str
    code: str


@dataclass(slots=True)
class EvolutionTrack:
    items: list[EvolutionItem]
    transitions: list[EvolutionTransition]


@dataclass(slots=True)
class EvolutionReport:
    query: str
    retrieval_mode: str
    versions_searched: list[str]
    tracks: list[EvolutionTrack]


def _change_label(match: VersionMatch) -> str:
    source, target = match.source, match.target
    if source.code.strip() == target.code.strip():
        return "likely_unchanged"
    if source.symbol_name != target.symbol_name or source.file_path != target.file_path:
        return "likely_renamed_or_moved"
    return "likely_modified"


def _topological_order(records: list[VersionRecord]) -> list[VersionRecord]:
    """Order indexed commits by known parent links, retaining catalog order for ties."""
    position = {record.version_id: index for index, record in enumerate(records)}
    remaining = {record.version_id: record for record in records}
    ordered: list[VersionRecord] = []
    while remaining:
        ready = [record for record in remaining.values()
                 if not any(parent in remaining for parent in record.parents)]
        if not ready:  # Defensive fallback for a malformed catalog.
            ready = list(remaining.values())
        ready.sort(key=lambda record: position[record.version_id])
        for record in ready:
            ordered.append(record)
            remaining.pop(record.version_id, None)
    return ordered


class EvolutionaryRetriever:
    """Query relevant snippets in each version and connect likely continuations."""

    def __init__(self, catalog_path: str | Path, matcher: VersionSimilarityMatcher | None = None):
        self.search_engine = VersionSearchEngine(catalog_path)
        # The baseline works without model downloads; callers can opt into embeddings.
        self.matcher = matcher or VersionSimilarityMatcher(weights=VersionSimilarityWeights(
            semantic=0.0, file_path=0.25, symbol=0.15, structure=0.35, git=0.25,
        ))

    def search(self, query: str, per_version_top_k: int = 10,
               candidate_pool_size: int = 50, min_similarity: float = 0.45,
               retrieval_mode: str = "auto", max_tracks: int = 50) -> EvolutionReport:
        if not query.strip():
            raise ValueError("query must not be empty")
        if per_version_top_k < 1 or candidate_pool_size < 1 or max_tracks < 1:
            raise ValueError("per-version top-k, candidate pool, and max tracks must be positive")
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between zero and one")

        catalog = self.search_engine.catalog
        records = _topological_order(catalog.versions)
        semantic_models = {record.model_name for record in records if record.semantic_index_path}
        all_semantic = all(record.semantic_index_path for record in records)
        can_search_hybrid = all_semantic and len(semantic_models) <= 1
        if retrieval_mode == "hybrid" and not can_search_hybrid:
            raise ValueError("Hybrid search requires a semantic index built with the same model for every version")
        common_mode = ("hybrid" if can_search_hybrid else "lexical") if retrieval_mode == "auto" else retrieval_mode
        hits: dict[str, list[VersionSearchResult]] = {}
        for record in records:
            report = self.search_engine.search(
                query, version=record.label, top_k=per_version_top_k,
                candidate_k=max(candidate_pool_size, per_version_top_k), mode=common_mode,
            )
            hits[record.version_id] = report.results

        record_by_id = {record.version_id: record for record in records}
        incoming: dict[tuple[str, str], list[tuple[tuple[str, str], EvolutionTransition]]] = {}
        outgoing: dict[tuple[str, str], list[tuple[tuple[str, str], EvolutionTransition]]] = {}
        for child in records:
            child_hits = hits[child.version_id]
            if not child_hits:
                continue
            for parent_id in child.parents:
                parent = record_by_id.get(parent_id)
                if parent is None or not hits.get(parent_id):
                    continue
                matches = self.matcher.compare(
                    parent, child,
                    [result.chunk for result in hits[parent_id]],
                    [result.chunk for result in child_hits],
                    matches_per_chunk=3, min_similarity=min_similarity,
                    candidate_pool_size=candidate_pool_size,
                )
                # Greedy one-to-one links avoid turning duplicate functions into a
                # single evolutionary chain. Scores are ranking evidence, not proof.
                used_source: set[str] = set()
                used_target: set[str] = set()
                for match in sorted(matches, key=lambda item: (
                    -item.similarity_score, item.source.chunk_id, item.target.chunk_id,
                )):
                    if match.source.chunk_id in used_source or match.target.chunk_id in used_target:
                        continue
                    used_source.add(match.source.chunk_id)
                    used_target.add(match.target.chunk_id)
                    source_key = (parent.version_id, match.source.chunk_id)
                    target_key = (child.version_id, match.target.chunk_id)
                    transition = EvolutionTransition(
                        from_version=parent.label, to_version=child.label,
                        from_chunk_id=match.source.chunk_id, to_chunk_id=match.target.chunk_id,
                        from_symbol=match.source.symbol_name, to_symbol=match.target.symbol_name,
                        from_path=match.source.file_path, to_path=match.target.file_path,
                        similarity_score=match.similarity_score, change=_change_label(match),
                    )
                    outgoing.setdefault(source_key, []).append((target_key, transition))
                    incoming.setdefault(target_key, []).append((source_key, transition))

        results_by_key = {
            (result.version_id, result.chunk.chunk_id): result
            for version_hits in hits.values() for result in version_hits
        }
        all_keys = list(results_by_key)
        starts = [key for key in all_keys if key not in incoming]
        tracks: list[EvolutionTrack] = []

        def emit_path(key, keys, transitions, seen):
            if key in seen:
                return
            seen = {*seen, key}
            next_links = sorted(outgoing.get(key, []), key=lambda pair: (
                -pair[1].similarity_score, pair[0][0], pair[0][1],
            ))
            if not next_links:
                tracks.append(_make_track(keys, transitions, results_by_key, record_by_id))
                return
            for next_key, transition in next_links:
                if next_key in seen:
                    continue
                emit_path(next_key, [*keys, next_key], [*transitions, transition], seen)
                if len(tracks) >= max_tracks:
                    return

        for start in starts:
            emit_path(start, [start], [], set())
            if len(tracks) >= max_tracks:
                break
        # Cycles should not occur in Git history, but preserve any unvisited results
        # from malformed catalogs as standalone items rather than dropping them.
        covered = { (item.version_id, item.chunk_id) for track in tracks for item in track.items }
        for key in all_keys:
            if key not in covered and len(tracks) < max_tracks:
                tracks.append(_make_track([key], [], results_by_key, record_by_id))

        tracks.sort(key=lambda track: (
            -max((item.relevance_score for item in track.items), default=0.0),
            track.items[0].version_label if track.items else "",
            track.items[0].file_path if track.items else "",
        ))
        return EvolutionReport(
            query=query,
            retrieval_mode=common_mode,
            versions_searched=[record.label for record in records],
            tracks=tracks[:max_tracks],
        )


def _make_track(keys, transitions, results_by_key, records_by_id) -> EvolutionTrack:
    items: list[EvolutionItem] = []
    for index, key in enumerate(keys):
        result = results_by_key[key]
        record = records_by_id[result.version_id]
        if index == 0:
            change = "first_observed"
        else:
            transition = transitions[index - 1]
            change = transition.change
        items.append(EvolutionItem(
            version_id=result.version_id, version_label=result.version_label,
            commit_hash=record.commit_hash, chunk_id=result.chunk.chunk_id,
            symbol_name=result.chunk.symbol_name, file_path=result.chunk.file_path,
            start_line=result.chunk.start_line, end_line=result.chunk.end_line,
            relevance_score=result.score, change=change, code=result.chunk.code,
        ))
    return EvolutionTrack(items=items, transitions=transitions)
