from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.models.schemas import CodeChunk


@dataclass(slots=True)
class FusedCandidate:
    chunk: CodeChunk
    score: float
    semantic_score: float | None
    lexical_score: float | None
    normalized_semantic_score: float
    normalized_lexical_score: float
    semantic_rank: int | None
    lexical_rank: int | None


def _dedupe_by_chunk_id(results: Sequence, source_name: str) -> dict[str, object]:
    unique: dict[str, object] = {}
    for result in results:
        chunk_id = result.chunk.chunk_id
        current = unique.get(chunk_id)
        if current is None or result.score > current.score:
            unique[chunk_id] = result
        elif result.chunk.to_dict() != current.chunk.to_dict():
            raise ValueError(f"Conflicting metadata for chunk_id '{chunk_id}' in {source_name} results")
    return unique


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        # A one-result list (or exact tie) should still contribute its full signal.
        return {key: 1.0 for key in scores}
    span = high - low
    return {key: (value - low) / span for key, value in scores.items()}


def fuse_candidates(lexical_results: Sequence, semantic_results: Sequence,
                    lexical_weight: float = 0.5,
                    semantic_weight: float = 0.5) -> list[FusedCandidate]:
    """Merge and min-max normalize independent lexical and semantic result lists."""
    if lexical_weight < 0 or semantic_weight < 0:
        raise ValueError("Fusion weights must not be negative")
    total_weight = lexical_weight + semantic_weight
    if total_weight == 0:
        raise ValueError("At least one fusion weight must be greater than zero")

    lexical = _dedupe_by_chunk_id(lexical_results, "lexical")
    semantic = _dedupe_by_chunk_id(semantic_results, "semantic")
    lexical_normalized = _normalize({key: item.score for key, item in lexical.items()})
    semantic_normalized = _normalize({key: item.score for key, item in semantic.items()})
    all_ids = set(lexical) | set(semantic)
    candidates: list[FusedCandidate] = []
    for chunk_id in all_ids:
        lexical_item = lexical.get(chunk_id)
        semantic_item = semantic.get(chunk_id)
        chunk = lexical_item.chunk if lexical_item is not None else semantic_item.chunk
        lexical_norm = lexical_normalized.get(chunk_id, 0.0)
        semantic_norm = semantic_normalized.get(chunk_id, 0.0)
        combined = (lexical_weight * lexical_norm + semantic_weight * semantic_norm) / total_weight
        candidates.append(FusedCandidate(
            chunk=chunk,
            score=combined,
            semantic_score=semantic_item.score if semantic_item is not None else None,
            lexical_score=lexical_item.score if lexical_item is not None else None,
            normalized_semantic_score=semantic_norm,
            normalized_lexical_score=lexical_norm,
            semantic_rank=semantic_item.rank if semantic_item is not None else None,
            lexical_rank=lexical_item.rank if lexical_item is not None else None,
        ))

    candidates.sort(key=lambda item: (
        -item.score, item.chunk.file_path.casefold(), item.chunk.start_line,
        item.chunk.chunk_id,
    ))
    return candidates
