from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.embeddings.sentence_transformer import DEFAULT_EMBEDDING_MODEL
from app.indexing.embedding_index import SemanticIndex
from app.retrieval.lexical_retriever import load_chunks_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a CPU FAISS semantic index from Phase A JSONL chunks.")
    parser.add_argument("chunks", help="Input Phase A JSONL chunks")
    parser.add_argument("--output", required=True, help="FAISS index file path")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Sentence Transformers model name or local path")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    try:
        chunks = load_chunks_jsonl(args.chunks)
        index = SemanticIndex.build(chunks, args.model, args.batch_size)
        index_path, metadata_path = index.save(args.output)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Indexed chunks: {len(chunks)}")
    print(f"Embedding model: {args.model}")
    print(f"Vector dimensions: {index.dimension}")
    print(f"FAISS index: {index_path}")
    print(f"Chunk metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
