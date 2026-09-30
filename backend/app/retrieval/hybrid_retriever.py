from __future__ import annotations

from pathlib import Path

from app.retrieval.candidate_fusion import FusedCandidate, fuse_candidates
from app.retrieval.lexical_retriever import LexicalRetriever
from app.retrieval.semantic_retriever import SemanticRetriever


class HybridRetriever:
    """Two-channel BM25 + semantic retrieval with configurable score fusion."""

    def __init__(self, lexical: LexicalRetriever, semantic: SemanticRetriever,
                 lexical_weight: float = 0.5, semantic_weight: float = 0.5):
        if lexical_weight < 0 or semantic_weight < 0:
            raise ValueError("Fusion weights must not be negative")
        if lexical_weight + semantic_weight == 0:
            raise ValueError("At least one fusion weight must be greater than zero")
        self.lexical = lexical
        self.semantic = semantic
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

    @classmethod
    def from_saved_indexes(cls, bm25_index_path: str | Path,
                           semantic_index_path: str | Path,
                           lexical_weight: float = 0.5,
                           semantic_weight: float = 0.5,
                           model_name: str | None = None,
                           batch_size: int = 32) -> "HybridRetriever":
        lexical = LexicalRetriever.from_saved_index(bm25_index_path)
        semantic = SemanticRetriever.from_saved_index(
            semantic_index_path, model_name=model_name, batch_size=batch_size,
        )
        return cls(lexical, semantic, lexical_weight, semantic_weight)

    def search(self, query: str, top_k: int = 10,
               candidate_k: int = 50) -> list[FusedCandidate]:
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        if candidate_k < 0:
            raise ValueError("candidate_k must not be negative")
        if not query.strip() or top_k == 0 or candidate_k == 0:
            return []
        lexical_results = self.lexical.search(query, candidate_k)
        semantic_results = self.semantic.search(query, candidate_k)
        return fuse_candidates(
            lexical_results, semantic_results,
            lexical_weight=self.lexical_weight,
            semantic_weight=self.semantic_weight,
        )[:top_k]
