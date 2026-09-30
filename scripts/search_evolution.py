from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.embeddings.sentence_transformer import DEFAULT_EMBEDDING_MODEL
from app.versions.evolution import EvolutionaryRetriever
from app.versions.similarity import VersionSimilarityMatcher, VersionSimilarityWeights


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve query-relevant code and trace likely continuations across indexed Git versions."
    )
    parser.add_argument("query")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--per-version-top-k", type=int, default=10)
    parser.add_argument("--candidate-pool-size", type=int, default=50)
    parser.add_argument("--max-tracks", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.45)
    parser.add_argument("--mode", choices=("auto", "lexical", "hybrid"), default="auto")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--semantic-weight", type=float, default=0.0,
                        help="Set above zero to use embedding similarity; may require model dependencies.")
    parser.add_argument("--weight-path", type=float, default=0.25)
    parser.add_argument("--weight-symbol", type=float, default=0.15)
    parser.add_argument("--weight-structure", type=float, default=0.35)
    parser.add_argument("--weight-git", type=float, default=0.25)
    args = parser.parse_args()
    try:
        matcher = VersionSimilarityMatcher(
            model_name=args.model,
            weights=VersionSimilarityWeights(
                semantic=args.semantic_weight, file_path=args.weight_path,
                symbol=args.weight_symbol, structure=args.weight_structure,
                git=args.weight_git,
            ),
        )
        report = EvolutionaryRetriever(args.catalog, matcher=matcher).search(
            args.query, per_version_top_k=args.per_version_top_k,
            candidate_pool_size=args.candidate_pool_size,
            min_similarity=args.min_similarity, retrieval_mode=args.mode,
            max_tracks=args.max_tracks,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(f"Evolutionary retrieval ({report.retrieval_mode}) for: {report.query}")
    print("Versions: " + (" -> ".join(report.versions_searched) or "none"))
    print(f"Likely implementation tracks: {len(report.tracks)}")
    for track_index, track in enumerate(report.tracks, start=1):
        print(f"\nTrack {track_index}")
        for item_index, item in enumerate(track.items):
            if item_index:
                transition = track.transitions[item_index - 1]
                print(f"  -- {transition.change} (similarity={transition.similarity_score:.3f}) -->")
            print(f"  {item.version_label}: {item.symbol_name} [{item.file_path}:{item.start_line}-"
                  f"{item.end_line}] relevance={item.relevance_score:.3f} ({item.change})")
    if not report.tracks:
        print("No query-relevant code was retrieved from the indexed versions.")
    print("Similarity links are ranking evidence and should be reviewed; they do not prove equivalence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
