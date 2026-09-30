from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.indexing.bm25_index import BM25Index
from app.models.schemas import CodeChunk


@dataclass(slots=True)
class LexicalResult:
    rank: int
    chunk: CodeChunk
    score: float


def load_chunks_jsonl(path: str | Path) -> list[CodeChunk]:
    source = Path(path)
    chunks: list[CodeChunk] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    chunks.append(CodeChunk(**json.loads(line)))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"Invalid chunk record at {source}:{line_number}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read chunk file {source}: {exc}") from exc
    return chunks


class LexicalRetriever:
    def __init__(self, index: BM25Index):
        self.index = index

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "LexicalRetriever":
        return cls(BM25Index.build(load_chunks_jsonl(path)))

    @classmethod
    def from_saved_index(cls, path: str | Path) -> "LexicalRetriever":
        return cls(BM25Index.load(path))

    def search(self, query: str, top_k: int = 10) -> list[LexicalResult]:
        return [
            LexicalResult(rank=rank, chunk=chunk, score=score)
            for rank, (chunk, score) in enumerate(self.index.search(query, top_k), start=1)
        ]
