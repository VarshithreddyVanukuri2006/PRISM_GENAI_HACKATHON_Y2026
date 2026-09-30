from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from app.graph.code_graph import CodeGraph, GraphExpansion
from app.models.schemas import CodeChunk
from app.retrieval.candidate_fusion import FusedCandidate, fuse_candidates
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.lexical_retriever import LexicalResult
from app.retrieval.reranker import (
    CodeAwareReranker, RerankedResult, RerankerWeights,
)
from app.retrieval.semantic_retriever import SemanticResult


@dataclass(slots=True)
class MultiPassReport:
    pass1_candidate_count: int
    pass2_seed_chunk_ids: list[str]
    pass3_expansions: list[GraphExpansion]
    pass4_candidate_count: int
    pass4_added_chunk_ids: list[str]
    pass5_candidate_count: int
    results: list[RerankedResult] = field(default_factory=list)


def _merge_results(*result_lists: Sequence) -> list:
    merged: dict[str, object] = {}
    for result_list in result_lists:
        for result in result_list:
            chunk_id = result.chunk.chunk_id
            current = merged.get(chunk_id)
            if current is None or result.score > current.score:
                merged[chunk_id] = result
    return list(merged.values())


def _context_for_expansions(query: str, expansions: Sequence[GraphExpansion],
                            chunks: dict[str, CodeChunk], max_chars: int) -> str:
    sections = [query.strip()]
    for expansion in expansions:
        chunk = chunks.get(expansion.chunk_id)
        if chunk is None:
            continue
        details = [chunk.symbol_name, chunk.file_path, chunk.docstring]
        details.extend(chunk.imports)
        details.extend(chunk.calls)
        if chunk.code:
            details.append(chunk.code[:1000])
        sections.append(" ".join(value for value in details if value))
        current = "\n".join(sections)
        if len(current) >= max_chars:
            return current[:max_chars]
    return "\n".join(sections)[:max_chars]


class MultiPassRetriever:
    """Broad hybrid retrieval, graph-context expansion, and final reranking."""

    def __init__(self, hybrid: HybridRetriever, graph: CodeGraph,
                 reranker: CodeAwareReranker | None = None):
        self.hybrid = hybrid
        self.graph = graph
        if reranker is None:
            weights = RerankerWeights(graph=0.10)
            self.reranker = CodeAwareReranker(weights)
        else:
            self.reranker = reranker

    def search(self, query: str, top_k: int = 10, candidate_k: int = 50,
               seed_k: int = 10, context_candidate_k: int = 25,
               max_graph_hops: int = 3, max_graph_candidates: int = 100,
               max_context_chars: int = 4000) -> MultiPassReport:
        if top_k < 0 or candidate_k < 0 or seed_k < 0 or context_candidate_k < 0:
            raise ValueError("top_k, candidate_k, seed_k, and context_candidate_k must not be negative")
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than zero")
        if not query.strip() or top_k == 0 or candidate_k == 0:
            return MultiPassReport(0, [], [], 0, [], 0, [])

        # Pass 1: broad candidates from the existing independent lexical and semantic runs.
        pass1_lexical = self.hybrid.lexical.search(query, candidate_k)
        pass1_semantic = self.hybrid.semantic.search(query, candidate_k)
        pass1_candidates = fuse_candidates(
            pass1_lexical, pass1_semantic,
            lexical_weight=self.hybrid.lexical_weight,
            semantic_weight=self.hybrid.semantic_weight,
        )

        # Pass 2: use current reranking signals to identify graph-expansion seeds.
        seed_results = self.reranker.rerank(query, pass1_candidates, seed_k)
        seed_ids = [result.chunk.chunk_id for result in seed_results]

        # Pass 3: walk resolved code relationships from those seeds.
        expansions = self.graph.expand_from_chunks(
            seed_ids, max_hops=max_graph_hops, max_candidates=max_graph_candidates,
        ) if seed_ids and max_graph_candidates else []

        # Pass 4: search again using names and metadata discovered through the graph.
        context_query = _context_for_expansions(
            query, expansions, self.graph.chunks, max_context_chars,
        ) if expansions else ""
        context_lexical: list[LexicalResult] = []
        context_semantic: list[SemanticResult] = []
        if context_query and context_candidate_k:
            context_lexical = self.hybrid.lexical.search(context_query, context_candidate_k)
            context_semantic = self.hybrid.semantic.search(context_query, context_candidate_k)
        pass4_ids = {result.chunk.chunk_id for result in [*context_lexical, *context_semantic]}
        pass1_ids = {candidate.chunk.chunk_id for candidate in pass1_candidates}

        # Fuse the raw evidence from both query passes against one shared candidate pool.
        merged_lexical = _merge_results(pass1_lexical, context_lexical)
        merged_semantic = _merge_results(pass1_semantic, context_semantic)
        final_candidates = fuse_candidates(
            merged_lexical, merged_semantic,
            lexical_weight=self.hybrid.lexical_weight,
            semantic_weight=self.hybrid.semantic_weight,
        )

        graph_scores: dict[str, float] = {}
        graph_paths: dict[str, list[list[str]]] = {}
        for expansion in expansions:
            graph_scores[expansion.chunk_id] = max(
                graph_scores.get(expansion.chunk_id, 0.0), expansion.graph_score,
            )
            path = graph_paths.setdefault(expansion.chunk_id, [])
            if expansion.relation_path not in path:
                path.append(expansion.relation_path)

        # Graph-discovered chunks remain candidates even if the second lexical/semantic pass missed them.
        present_ids = {candidate.chunk.chunk_id for candidate in final_candidates}
        for chunk_id in graph_scores:
            if chunk_id not in present_ids:
                chunk = self.graph.chunks[chunk_id]
                final_candidates.append(FusedCandidate(
                    chunk=chunk, score=0.0, semantic_score=None, lexical_score=None,
                    normalized_semantic_score=0.0, normalized_lexical_score=0.0,
                    semantic_rank=None, lexical_rank=None,
                ))

        # Pass 5: rerank the complete retrieval and graph candidate set.
        results = self.reranker.rerank(
            query, final_candidates, top_k,
            graph_scores=graph_scores, graph_paths=graph_paths,
        )
        return MultiPassReport(
            pass1_candidate_count=len(pass1_candidates),
            pass2_seed_chunk_ids=seed_ids,
            pass3_expansions=expansions,
            pass4_candidate_count=len(pass4_ids),
            pass4_added_chunk_ids=sorted(pass4_ids - pass1_ids),
            pass5_candidate_count=len(final_candidates),
            results=results,
        )
