from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.versions.version_search import VersionSearchEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Search a specific indexed Git version or all versions in a catalog.")
    parser.add_argument("query")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--version", help="Exact version label or commit ID/prefix; omit to search all")
    parser.add_argument("--mode", choices=("auto", "lexical", "hybrid"), default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--semantic-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.top_k < 0 or args.candidate_k < 0:
        parser.error("--top-k and --candidate-k must not be negative")
    try:
        report = VersionSearchEngine(args.catalog).search(
            args.query, version=args.version, top_k=args.top_k,
            candidate_k=args.candidate_k, mode=args.mode,
            lexical_weight=args.bm25_weight, semantic_weight=args.semantic_weight,
            batch_size=args.batch_size,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Search mode: {report.retrieval_mode}; versions: {', '.join(report.versions_searched)}")
    if not report.results:
        print("No matching code chunks found.")
        return 0
    for result in report.results:
        chunk = result.chunk
        print(f"#{result.rank} [{result.version_label}] {result.score:.4f} {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        print(f"    {chunk.symbol_name} ({chunk.symbol_type}, {chunk.language}, commit {result.version_id[:12]})")
        print(f"    BM25: {result.lexical_score}; semantic: {result.semantic_score}")
        if chunk.docstring:
            print(f"    {chunk.docstring}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
