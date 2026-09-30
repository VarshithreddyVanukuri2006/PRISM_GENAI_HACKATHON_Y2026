from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.embeddings.sentence_transformer import DEFAULT_EMBEDDING_MODEL
from app.retrieval.lexical_retriever import load_chunks_jsonl
from app.versions.similarity import VersionSimilarityMatcher, VersionSimilarityWeights
from app.versions.version_manager import VersionCatalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Find similar code units between two indexed Git versions.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--from-version", required=True, help="Source version label or commit ID/prefix")
    parser.add_argument("--to-version", required=True, help="Target version label or commit ID/prefix")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--candidate-pool-size", type=int, default=50)
    parser.add_argument("--matches-per-chunk", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument("--weight-semantic", type=float, default=0.55)
    parser.add_argument("--weight-path", type=float, default=0.15)
    parser.add_argument("--weight-symbol", type=float, default=0.10)
    parser.add_argument("--weight-structure", type=float, default=0.12)
    parser.add_argument("--weight-git", type=float, default=0.08)
    args = parser.parse_args()
    if args.candidate_pool_size < 1 or args.matches_per_chunk < 0:
        parser.error("candidate pool must be positive and matches per chunk must not be negative")
    catalog_path = Path(args.catalog).resolve()
    try:
        catalog = VersionCatalog.load(catalog_path)
        engine = _CatalogSelector(catalog.versions)
        source_record = engine.select(args.from_version)
        target_record = engine.select(args.to_version)
        if source_record.version_id == target_record.version_id:
            raise ValueError("Source and target versions must be different")
        source_path = catalog_path.parent / source_record.chunks_path
        target_path = catalog_path.parent / target_record.chunks_path
        source_chunks = load_chunks_jsonl(source_path)
        target_chunks = load_chunks_jsonl(target_path)
        matcher = VersionSimilarityMatcher(
            args.model,
            VersionSimilarityWeights(
                semantic=args.weight_semantic,
                file_path=args.weight_path,
                symbol=args.weight_symbol,
                structure=args.weight_structure,
                git=args.weight_git,
            ),
        )
        matches = matcher.compare(
            source_record, target_record, source_chunks, target_chunks,
            matches_per_chunk=args.matches_per_chunk,
            min_similarity=args.min_similarity,
            candidate_pool_size=args.candidate_pool_size,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Comparing {source_record.label} -> {target_record.label}; matched pairs: {len(matches)}")
    for match in matches:
        print(f"{match.similarity_score:.4f}  {match.source.symbol_name} [{match.source.file_path}] -> "
              f"{match.target.symbol_name} [{match.target.file_path}]")
        print("    " + ", ".join(f"{name}={score:.3f}" for name, score in match.components.items()))
    return 0


class _CatalogSelector:
    def __init__(self, records):
        self.records = records

    def select(self, selector):
        matches = [record for record in self.records
                   if record.label == selector or record.version_id == selector
                   or record.commit_hash.startswith(selector)]
        if len(matches) != 1:
            raise ValueError(f"Version selector '{selector}' matched {len(matches)} versions")
        return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
