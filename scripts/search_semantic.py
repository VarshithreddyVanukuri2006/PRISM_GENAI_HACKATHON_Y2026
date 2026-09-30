from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.retrieval.semantic_retriever import SemanticRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Search a saved FAISS semantic index.")
    parser.add_argument("query", help="Natural-language or code-oriented search query")
    parser.add_argument("--index", required=True, help="Path used when building the semantic index")
    parser.add_argument("--model", help="Override model from index metadata (must match the indexed model)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.top_k < 0:
        parser.error("--top-k must not be negative")
    try:
        retriever = SemanticRetriever.from_saved_index(args.index, args.model, args.batch_size)
        results = retriever.search(args.query, args.top_k)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    if not results:
        print("No matching code chunks found.")
        return 0
    for result in results:
        chunk = result.chunk
        print(f"#{result.rank}  cosine {result.score:.4f}  {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        print(f"    {chunk.symbol_name} ({chunk.symbol_type}, {chunk.language})")
        if chunk.docstring:
            print(f"    {chunk.docstring}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
