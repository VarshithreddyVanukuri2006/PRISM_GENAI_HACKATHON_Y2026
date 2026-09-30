from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.indexing.bm25_index import BM25Index
from app.retrieval.lexical_retriever import LexicalRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Search indexed code chunks with BM25 lexical retrieval.")
    parser.add_argument("query", help="Natural-language or code-oriented search query")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--chunks", help="Phase A JSONL chunk file")
    source.add_argument("--index", help="Previously saved BM25 index JSON file")
    parser.add_argument("--save-index", help="Save the built BM25 index for later searches")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 0:
        parser.error("--top-k must not be negative")

    try:
        if args.index:
            retriever = LexicalRetriever.from_saved_index(args.index)
        else:
            retriever = LexicalRetriever.from_jsonl(args.chunks)
            if args.save_index:
                retriever.index.save(args.save_index)
        results = retriever.search(args.query, args.top_k)
    except ValueError as exc:
        parser.error(str(exc))

    if not results:
        print("No matching code chunks found.")
        return 0
    for result in results:
        chunk = result.chunk
        print(f"#{result.rank}  BM25 {result.score:.4f}  {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        print(f"    {chunk.symbol_name} ({chunk.symbol_type}, {chunk.language})")
        if chunk.docstring:
            print(f"    {chunk.docstring}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
