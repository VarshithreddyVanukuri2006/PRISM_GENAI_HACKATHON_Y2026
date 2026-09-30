from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CodeAwareReranker, RerankerWeights


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval followed by configurable code-aware reranking.")
    parser.add_argument("query", help="Natural-language or code-oriented search query")
    parser.add_argument("--bm25-index", required=True)
    parser.add_argument("--semantic-index", required=True)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-weight", type=float, default=0.5,
                        help="BM25 weight in Phase D candidate fusion")
    parser.add_argument("--semantic-weight", type=float, default=0.5,
                        help="semantic weight in Phase D candidate fusion")
    parser.add_argument("--model", help="Override model in semantic index metadata (must match)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-semantic", type=float, default=0.25)
    parser.add_argument("--weight-lexical", type=float, default=0.25)
    parser.add_argument("--weight-symbol", type=float, default=0.10)
    parser.add_argument("--weight-filename", type=float, default=0.075)
    parser.add_argument("--weight-language", type=float, default=0.025)
    parser.add_argument("--weight-intent", type=float, default=0.10)
    parser.add_argument("--weight-metadata", type=float, default=0.15)
    parser.add_argument("--weight-dependency", type=float, default=0.05)
    args = parser.parse_args()
    if args.candidate_k < 0 or args.top_k < 0:
        parser.error("--candidate-k and --top-k must not be negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")

    weights = RerankerWeights(
        semantic=args.weight_semantic,
        lexical=args.weight_lexical,
        exact_symbol=args.weight_symbol,
        filename=args.weight_filename,
        language=args.weight_language,
        intent=args.weight_intent,
        metadata=args.weight_metadata,
        dependency=args.weight_dependency,
    )
    try:
        reranker = CodeAwareReranker(weights)
        hybrid = HybridRetriever.from_saved_indexes(
            args.bm25_index, args.semantic_index,
            lexical_weight=args.bm25_weight,
            semantic_weight=args.semantic_weight,
            model_name=args.model,
            batch_size=args.batch_size,
        )
        candidates = hybrid.search(args.query, top_k=args.candidate_k,
                                   candidate_k=args.candidate_k)
        results = reranker.rerank(args.query, candidates, args.top_k)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    if not results:
        print("No matching code chunks found.")
        return 0
    for result in results:
        chunk = result.chunk
        print(f"#{result.rank}  reranked {result.final_score:.4f}  {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        print(f"    {chunk.symbol_name} ({chunk.symbol_type}, {chunk.language})")
        print(f"    Hybrid: {result.hybrid_score:.4f}; BM25: {result.lexical_score}; semantic cosine: {result.semantic_score}")
        features = ", ".join(f"{name}={value:.3f}" for name, value in result.features.items())
        print(f"    Features: {features}")
        for signal in result.signals:
            print(f"    Signal: {signal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
