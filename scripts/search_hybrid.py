from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.retrieval.hybrid_retriever import HybridRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Search saved BM25 and semantic indexes with score fusion.")
    parser.add_argument("query", help="Natural-language or code-oriented search query")
    parser.add_argument("--bm25-index", required=True, help="Saved BM25 JSON index")
    parser.add_argument("--semantic-index", required=True, help="Saved FAISS semantic index")
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--semantic-weight", type=float, default=0.5)
    parser.add_argument("--candidate-k", type=int, default=50,
                        help="Candidates fetched from each retriever before fusion")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", help="Override model in semantic index metadata (must match)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.top_k < 0 or args.candidate_k < 0:
        parser.error("--top-k and --candidate-k must not be negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.bm25_weight < 0 or args.semantic_weight < 0:
        parser.error("fusion weights must not be negative")
    if args.bm25_weight + args.semantic_weight == 0:
        parser.error("at least one fusion weight must be greater than zero")

    try:
        retriever = HybridRetriever.from_saved_indexes(
            args.bm25_index, args.semantic_index,
            lexical_weight=args.bm25_weight,
            semantic_weight=args.semantic_weight,
            model_name=args.model,
            batch_size=args.batch_size,
        )
        results = retriever.search(args.query, args.top_k, args.candidate_k)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    if not results:
        print("No matching code chunks found.")
        return 0
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        print(f"#{rank}  hybrid {result.score:.4f}  {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        print(f"    {chunk.symbol_name} ({chunk.symbol_type}, {chunk.language})")
        lexical = "—" if result.lexical_score is None else f"{result.lexical_score:.4f} (norm {result.normalized_lexical_score:.3f})"
        semantic = "—" if result.semantic_score is None else f"{result.semantic_score:.4f} (norm {result.normalized_semantic_score:.3f})"
        print(f"    BM25: {lexical}; semantic cosine: {semantic}")
        if chunk.docstring:
            print(f"    {chunk.docstring}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
