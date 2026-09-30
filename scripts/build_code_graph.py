from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.graph.code_graph import CodeGraph
from app.retrieval.lexical_retriever import load_chunks_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a repository code relationship graph from Phase A chunks.")
    parser.add_argument("chunks", help="Phase A JSONL chunk file")
    parser.add_argument("--output", required=True, help="Graph JSON output path")
    args = parser.parse_args()
    try:
        chunks = load_chunks_jsonl(args.chunks)
        code_graph = CodeGraph.from_chunks(chunks)
        output = code_graph.save(args.output)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    summary = code_graph.summary()
    print(f"Code chunks: {len(chunks)}")
    print(f"Graph nodes: {summary['node_count']}")
    print(f"Graph edges: {summary['edge_count']}")
    print(f"Relationship counts: {summary['relations']}")
    print(f"Graph JSON: {output}")
    for warning in summary["warnings"]:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
