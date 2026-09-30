from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.embeddings.sentence_transformer import DEFAULT_EMBEDDING_MODEL
from app.indexing.embedding_index import Embedder, SemanticIndex
from app.models.schemas import CodeChunk


@dataclass(slots=True)
class SemanticResult:
    rank: int
    chunk: CodeChunk
    score: float


class SemanticRetriever:
    def __init__(self, index: SemanticIndex):
        self.index = index

    @classmethod
    def build(cls, chunks: list[CodeChunk], model_name: str = DEFAULT_EMBEDDING_MODEL,
              batch_size: int = 32, embedder: Embedder | None = None) -> "SemanticRetriever":
        return cls(SemanticIndex.build(chunks, model_name, batch_size, embedder))

    @classmethod
    def from_saved_index(cls, path: str | Path, model_name: str | None = None,
                         batch_size: int = 32,
                         embedder: Embedder | None = None) -> "SemanticRetriever":
        return cls(SemanticIndex.load(path, model_name, batch_size, embedder))

    def search(self, query: str, top_k: int = 10) -> list[SemanticResult]:
        return [
            SemanticResult(rank=rank, chunk=chunk, score=score)
            for rank, (chunk, score) in enumerate(self.index.search(query, top_k), start=1)
        ]
