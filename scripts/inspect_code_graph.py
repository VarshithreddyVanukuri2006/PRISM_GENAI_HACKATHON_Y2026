from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.graph.code_graph import CodeGraph


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect direct relationships for a code chunk in a saved graph.")
    parser.add_argument("--graph", required=True, help="Saved graph JSON file")
    parser.add_argument("--chunk-id", required=True, help="Chunk ID to inspect")
    args = parser.parse_args()
    try:
        code_graph = CodeGraph.load(args.graph)
        chunk = code_graph.chunks.get(args.chunk_id)
        if chunk is None:
            parser.error(f"Unknown chunk_id: {args.chunk_id}")
        print(f"{chunk.symbol_name} ({chunk.file_path}:{chunk.start_line}-{chunk.end_line})")
        relationships = code_graph.related_to_chunk(args.chunk_id)
    except (ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    if not relationships:
        print("No resolved direct relationships.")
        return 0
    for relationship in relationships:
        node = relationship["node"]
        name = node.get("symbol_name") or node.get("file_path")
        print(f"{relationship['direction']:8} {relationship['relation']:10} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
