from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.embeddings.sentence_transformer import DEFAULT_EMBEDDING_MODEL
from app.versions.version_manager import VersionManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Index a Git commit/tag as an isolated CodeLens version.")
    parser.add_argument("repository", help="Local Git repository path")
    parser.add_argument("revision", help="Git branch, tag, or commit ID")
    parser.add_argument("--output-dir", default="data/versions", help="Version index root")
    parser.add_argument("--label", help="Human-readable version label; defaults to the revision")
    parser.add_argument("--semantic", action="store_true", help="Also build a CPU semantic index for this version")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    try:
        catalog_path, record = VersionManager(args.output_dir).index_git_version(
            args.repository, args.revision, label=args.label,
            include_semantic=args.semantic, model_name=args.model,
            batch_size=args.batch_size,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Version: {record.label}")
    print(f"Commit: {record.commit_hash}")
    print(f"Files: {record.number_of_files}; chunks: {record.number_of_chunks}")
    print(f"BM25 index: {record.bm25_index_path}")
    if record.semantic_index_path:
        print(f"Semantic index: {record.semantic_index_path}")
    print(f"Catalog: {catalog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
