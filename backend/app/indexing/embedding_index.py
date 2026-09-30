from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol, Sequence

from app.embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder,
)
from app.metadata.extractor import searchable_text
from app.models.schemas import CodeChunk


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str]): ...


def _load_vector_dependencies():
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Semantic retrieval requires FAISS and NumPy. Install the optional dependencies "
            "with: pip install -r backend/requirements-semantic.txt"
        ) from exc
    return faiss, np


def _normalize(vectors, np):
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Embedding model returned an empty or invalid vector matrix")
    if not np.isfinite(array).all():
        raise ValueError("Embedding model returned non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Embedding model returned a zero-length vector")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


class SemanticIndex:
    """Exact cosine-similarity index backed by FAISS IndexFlatIP."""

    def __init__(self, chunks: list[CodeChunk], vector_index, embedder: Embedder,
                 dimension: int):
        self.chunks = chunks
        self.vector_index = vector_index
        self.embedder = embedder
        self.dimension = dimension

    @classmethod
    def build(cls, chunks: Sequence[CodeChunk], model_name: str = DEFAULT_EMBEDDING_MODEL,
              batch_size: int = 32, embedder: Embedder | None = None) -> "SemanticIndex":
        if not chunks:
            raise ValueError("Cannot build a semantic index without code chunks")
        faiss, np = _load_vector_dependencies()
        model = embedder or SentenceTransformerEmbedder(model_name, batch_size)
        texts = [searchable_text(chunk) for chunk in chunks]
        vectors = _normalize(model.encode(texts), np)
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding model returned {vectors.shape[0]} vectors for {len(chunks)} code chunks"
            )
        vector_index = faiss.IndexFlatIP(int(vectors.shape[1]))
        vector_index.add(vectors)
        return cls(list(chunks), vector_index, model, int(vectors.shape[1]))

    def search(self, query: str, top_k: int = 10) -> list[tuple[CodeChunk, float]]:
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        if not query.strip() or top_k == 0 or not self.chunks:
            return []
        _, np = _load_vector_dependencies()
        query_vector = _normalize(self.embedder.encode([query]), np)
        if query_vector.shape[0] != 1 or query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension does not match index dimension {self.dimension}"
            )
        scores, ids = self.vector_index.search(query_vector, min(top_k, len(self.chunks)))
        results: list[tuple[CodeChunk, float]] = []
        for score, chunk_id in zip(scores[0], ids[0]):
            if int(chunk_id) < 0:
                continue
            results.append((self.chunks[int(chunk_id)], float(score)))
        return results

    def save(self, index_path: str | Path) -> tuple[Path, Path]:
        faiss, _ = _load_vector_dependencies()
        destination = Path(index_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = Path(str(destination) + ".json")
        payload = {
            "format_version": 1,
            "model_name": self.embedder.model_name,
            "dimension": self.dimension,
            "count": len(self.chunks),
            "metric": "cosine",
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        faiss.write_index(self.vector_index, str(destination))
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return destination, metadata_path

    @classmethod
    def load(cls, index_path: str | Path, model_name: str | None = None,
             batch_size: int = 32, embedder: Embedder | None = None) -> "SemanticIndex":
        faiss, _ = _load_vector_dependencies()
        source = Path(index_path)
        metadata_path = Path(str(source) + ".json")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if payload.get("format_version") != 1:
                raise ValueError("unsupported semantic index format version")
            stored_model_name = payload["model_name"]
            selected_model_name = model_name or stored_model_name
            if selected_model_name != stored_model_name:
                raise ValueError(
                    f"Index uses model '{stored_model_name}', but '{selected_model_name}' was requested; "
                    "rebuild the index to use another model"
                )
            chunks = [CodeChunk(**item) for item in payload["chunks"]]
            vector_index = faiss.read_index(str(source))
            dimension = int(payload["dimension"])
            count = int(payload["count"])
            if len(chunks) != count or int(vector_index.ntotal) != count:
                raise ValueError("semantic index and metadata counts do not match")
            if int(vector_index.d) != dimension:
                raise ValueError("semantic index dimension does not match metadata")
            model = embedder or SentenceTransformerEmbedder(selected_model_name, batch_size)
            if getattr(model, "model_name", selected_model_name) != selected_model_name:
                raise ValueError("provided embedder model does not match the semantic index")
            return cls(chunks, vector_index, model, dimension)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Unable to load semantic index {source}: {exc}") from exc
