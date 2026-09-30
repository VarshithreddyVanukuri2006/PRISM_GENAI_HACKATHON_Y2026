from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.graph.code_graph import CodeGraph
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.multi_pass_retriever import MultiPassRetriever
from app.retrieval.reranker import CodeAwareReranker, RerankerWeights


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-pass retrieval with graph-based context expansion.")
    parser.add_argument("query", help="Natural-language or code-oriented search query")
    parser.add_argument("--bm25-index", required=True)
    parser.add_argument("--semantic-index", required=True)
    parser.add_argument("--graph", required=True, help="Saved Phase F graph JSON")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--seed-k", type=int, default=10)
    parser.add_argument("--context-candidate-k", type=int, default=25)
    parser.add_argument("--max-graph-hops", type=int, default=3)
    parser.add_argument("--max-graph-candidates", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--graph-weight", type=float, default=0.10,
                        help="Graph-proximity feature weight in the final reranking pass")
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--semantic-weight", type=float, default=0.5)
    parser.add_argument("--model", help="Override model in semantic index metadata (must match)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if min(args.candidate_k, args.seed_k, args.context_candidate_k,
           args.max_graph_candidates, args.top_k) < 0:
        parser.error("candidate, seed, graph, and top-k limits must not be negative")
    if args.max_graph_hops < 1:
        parser.error("--max-graph-hops must be at least one")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.graph_weight < 0:
        parser.error("--graph-weight must not be negative")

    try:
        hybrid = HybridRetriever.from_saved_indexes(
            args.bm25_index, args.semantic_index,
            lexical_weight=args.bm25_weight,
            semantic_weight=args.semantic_weight,
            model_name=args.model,
            batch_size=args.batch_size,
        )
        graph = CodeGraph.load(args.graph)
        reranker = CodeAwareReranker(RerankerWeights(graph=args.graph_weight))
        retriever = MultiPassRetriever(hybrid, graph, reranker)
        report = retriever.search(
            args.query, top_k=args.top_k, candidate_k=args.candidate_k,
            seed_k=args.seed_k, context_candidate_k=args.context_candidate_k,
            max_graph_hops=args.max_graph_hops,
            max_graph_candidates=args.max_graph_candidates,
        )
    except (ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))

    print(f"Pass 1 broad candidates: {report.pass1_candidate_count}")
    print(f"Pass 2 seed chunks: {', '.join(report.pass2_seed_chunk_ids) or 'none'}")
    print(f"Pass 3 graph expansions: {len(report.pass3_expansions)}")
    for expansion in report.pass3_expansions:
        target = graph.chunks[expansion.chunk_id]
        print(f"    {expansion.seed_chunk_id} -> {target.symbol_name} ({expansion.hops} hops, graph {expansion.graph_score:.3f})")
        print(f"        {' / '.join(expansion.relation_path)}")
    print(f"Pass 4 contextual candidates: {report.pass4_candidate_count}; new: {len(report.pass4_added_chunk_ids)}")
    print(f"Pass 5 candidates reranked: {report.pass5_candidate_count}")
    if not report.results:
        print("No matching code chunks found.")
        return 0
    for result in report.results:
        chunk = result.chunk
        print(f"#{result.rank}  score {result.final_score:.4f}  {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        print(f"    {chunk.symbol_name} ({chunk.symbol_type}, {chunk.language})")
        for signal in result.signals:
            print(f"    Signal: {signal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
