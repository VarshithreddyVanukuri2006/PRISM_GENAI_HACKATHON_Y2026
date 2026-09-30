from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.models.schemas import CodeChunk
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.lexical_retriever import LexicalRetriever
from app.versions.version_manager import VersionCatalog, VersionRecord


@dataclass(slots=True)
class VersionSearchResult:
    rank: int
    version_id: str
    version_label: str
    retrieval_mode: str
    chunk: CodeChunk
    score: float
    lexical_score: float | None
    semantic_score: float | None


@dataclass(slots=True)
class VersionSearchReport:
    retrieval_mode: str
    versions_searched: list[str]
    results: list[VersionSearchResult]


class VersionSearchEngine:
    def __init__(self, catalog_path: str | Path):
        self.catalog_path = Path(catalog_path).resolve()
        self.catalog = VersionCatalog.load(self.catalog_path)
        if not self.catalog.versions:
            raise ValueError(f"Version catalog contains no indexed versions: {self.catalog_path}")

    def _path_for(self, relative_path: str | None) -> Path | None:
        if relative_path is None:
            return None
        root = self.catalog_path.parent.resolve()
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Version artifact path escapes the catalog directory") from exc
        return path

    def select_versions(self, selector: str | None = None) -> list[VersionRecord]:
        if selector is None:
            return list(self.catalog.versions)
        matches = [record for record in self.catalog.versions
                   if record.label == selector or record.version_id == selector
                   or record.commit_hash.startswith(selector)]
        if not matches:
            raise ValueError(f"No indexed version matches '{selector}'")
        if len(matches) > 1:
            raise ValueError(f"Version selector '{selector}' is ambiguous; use a full commit or label")
        return matches

    def search(self, query: str, version: str | None = None, top_k: int = 10,
               candidate_k: int = 50, mode: str = "auto",
               lexical_weight: float = 0.5, semantic_weight: float = 0.5,
               batch_size: int = 32) -> VersionSearchReport:
        if top_k < 0 or candidate_k < 0:
            raise ValueError("top_k and candidate_k must not be negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if mode not in {"auto", "lexical", "hybrid"}:
            raise ValueError("mode must be 'auto', 'lexical', or 'hybrid'")
        selected = self.select_versions(version)
        if not query.strip() or top_k == 0 or candidate_k == 0:
            return VersionSearchReport("lexical" if mode == "auto" else mode,
                                       [record.label for record in selected], [])

        have_all_semantic = all(record.semantic_index_path for record in selected)
        model_names = {record.model_name for record in selected if record.semantic_index_path}
        can_use_hybrid = have_all_semantic and len(model_names) <= 1
        if mode == "hybrid" and not can_use_hybrid:
            raise ValueError("Hybrid search requires a semantic index built with the same model for every selected version")
        use_hybrid = can_use_hybrid if mode == "auto" else mode == "hybrid"
        effective_mode = "hybrid" if use_hybrid else "lexical"

        results: list[VersionSearchResult] = []
        for record in selected:
            bm25_path = self._path_for(record.bm25_index_path)
            if bm25_path is None or not bm25_path.is_file():
                raise ValueError(f"BM25 index is missing for version '{record.label}'")
            if use_hybrid:
                semantic_path = self._path_for(record.semantic_index_path)
                if semantic_path is None or not semantic_path.is_file():
                    raise ValueError(f"Semantic index is missing for version '{record.label}'")
                retriever = HybridRetriever.from_saved_indexes(
                    bm25_path, semantic_path,
                    lexical_weight=lexical_weight, semantic_weight=semantic_weight,
                    model_name=record.model_name, batch_size=batch_size,
                )
                candidates = retriever.search(query, top_k=candidate_k, candidate_k=candidate_k)
                for candidate in candidates:
                    results.append(VersionSearchResult(
                        rank=candidate.lexical_rank or candidate.semantic_rank or 0,
                        version_id=record.version_id, version_label=record.label,
                        retrieval_mode="hybrid", chunk=candidate.chunk,
                        score=candidate.score, lexical_score=candidate.lexical_score,
                        semantic_score=candidate.semantic_score,
                    ))
            else:
                retriever = LexicalRetriever.from_saved_index(bm25_path)
                lexical_results = retriever.search(query, candidate_k)
                raw_scores = [item.score for item in lexical_results]
                low = min(raw_scores, default=0.0)
                high = max(raw_scores, default=0.0)
                for item in lexical_results:
                    normalized = (1.0 if high == low else (item.score - low) / (high - low))
                    results.append(VersionSearchResult(
                        rank=item.rank, version_id=record.version_id,
                        version_label=record.label, retrieval_mode="lexical",
                        chunk=item.chunk, score=normalized,
                        lexical_score=item.score, semantic_score=None,
                    ))

        results.sort(key=lambda item: (
            -item.score, item.version_label, item.chunk.file_path.casefold(),
            item.chunk.start_line, item.chunk.chunk_id,
        ))
        top_results = results[:top_k]
        for rank, result in enumerate(top_results, start=1):
            result.rank = rank
        return VersionSearchReport(effective_mode, [record.label for record in selected], top_results)
