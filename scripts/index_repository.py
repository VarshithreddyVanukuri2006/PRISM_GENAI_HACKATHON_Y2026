from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.indexing.index_manager import index_repository, write_jsonl
from app.ingestion.repository_loader import RepositoryError


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a local repository and write Phase A code chunks.")
    parser.add_argument("repository", help="Path to a local repository")
    parser.add_argument("--output", help="JSONL output path (default: data/indexes/<repository-id>.jsonl)")
    args = parser.parse_args()
    try:
        summary = index_repository(args.repository)
        output = Path(args.output) if args.output else ROOT / "data" / "indexes" / f"{summary.repository_id}.jsonl"
        write_jsonl(summary, output)
    except RepositoryError as exc:
        parser.error(str(exc))
    print(f"Repository: {summary.repository_path}")
    print(f"Repository ID: {summary.repository_id}")
    print(f"Source files read: {summary.number_of_files}")
    print(f"Code chunks: {summary.number_of_chunks}")
    print(f"Unreadable supported files: {summary.skipped_files}")
    for warning in summary.warnings:
        print(f"Warning: {warning}")
    print(f"Chunk metadata: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
