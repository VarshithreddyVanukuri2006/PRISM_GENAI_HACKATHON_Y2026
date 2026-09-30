from __future__ import annotations

import math
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.indexing.embedding_index import SemanticIndex
from app.models.schemas import CodeChunk
from app.retrieval.semantic_retriever import SemanticRetriever


class Array(list):
    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)

    @property
    def ndim(self):
        return 2

    def __truediv__(self, other):
        return Array([[value / norm[0] for value in row] for row, norm in zip(self, other)])


class BoolArray:
    def __init__(self, values):
        self.values = values

    def all(self):
        return all(self.values)

    def any(self):
        return any(self.values)


class NormArray(list):
    def __eq__(self, other):
        return BoolArray([row[0] == other for row in self])


class FakeNumpy:
    float32 = "float32"

    @staticmethod
    def asarray(values, dtype=None):
        return Array([list(map(float, row)) for row in values])

    @staticmethod
    def isfinite(values):
        return BoolArray(math.isfinite(value) for row in values for value in row)

    class linalg:
        @staticmethod
        def norm(values, axis=1, keepdims=True):
            return NormArray([[math.sqrt(sum(value * value for value in row))] for row in values])

    @staticmethod
    def ascontiguousarray(values, dtype=None):
        return values


class FakeVectorIndex:
    def __init__(self, dimension):
        self.d = dimension
        self.vectors = []

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        self.vectors.extend([list(vector) for vector in vectors])

    def search(self, queries, k):
        query = queries[0]
        scored = [(sum(left * right for left, right in zip(query, vector)), i)
                  for i, vector in enumerate(self.vectors)]
        scored.sort(key=lambda value: (-value[0], value[1]))
        scored = scored[:k]
        return [[score for score, _ in scored]], [[i for _, i in scored]]


class FakeFaiss:
    IndexFlatIP = FakeVectorIndex

    @staticmethod
    def write_index(index, path):
        with open(path, "wb") as handle:
            pickle.dump(index, handle)

    @staticmethod
    def read_index(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)


class FakeEmbedder:
    model_name = "test/tiny-embedding"

    def encode(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "password" in lower or "credential" in lower:
                vectors.append([1.0, 0.0, 0.0])
            elif "jwt" in lower or "token" in lower:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class SemanticIndexTests(unittest.TestCase):
    def setUp(self):
        self.dependencies = patch(
            "app.indexing.embedding_index._load_vector_dependencies",
            return_value=(FakeFaiss, FakeNumpy),
        )
        self.dependencies.start()
        self.addCleanup(self.dependencies.stop)
        self.chunks = [
            CodeChunk("pw", "repo", "users/passwords.py", "Python", "hash_password", "function", 1, 2,
                      "def hash_password(password): return digest(password)", "Hash a user's password before storing credentials."),
            CodeChunk("jwt", "repo", "auth/tokens.py", "Python", "decode_jwt", "function", 1, 2,
                      "def decode_jwt(token): return jwt.decode(token)", "Decode and validate a signed JWT."),
        ]

    def test_semantic_query_ranks_natural_language_match(self):
        retriever = SemanticRetriever.build(self.chunks, embedder=FakeEmbedder())
        results = retriever.search("store user password credentials", top_k=2)
        self.assertEqual(results[0].chunk.symbol_name, "hash_password")
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertEqual([result.rank for result in results], [1, 2])

    def test_saved_index_round_trip_keeps_chunks_and_scores(self):
        retriever = SemanticRetriever.build(self.chunks, embedder=FakeEmbedder())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "semantic.faiss"
            retriever.index.save(path)
            restored = SemanticRetriever.from_saved_index(path, embedder=FakeEmbedder())
            results = restored.search("where is the JWT token decoded", top_k=1)
        self.assertEqual(results[0].chunk.symbol_name, "decode_jwt")
        self.assertAlmostEqual(results[0].score, 1.0)

    def test_empty_query_and_zero_top_k_return_no_results(self):
        index = SemanticIndex.build(self.chunks, embedder=FakeEmbedder())
        self.assertEqual(index.search(" "), [])
        self.assertEqual(index.search("password", top_k=0), [])

    def test_rejects_model_index_dimension_mismatch(self):
        index = SemanticIndex.build(self.chunks, embedder=FakeEmbedder())

        class WrongDimension(FakeEmbedder):
            def encode(self, texts):
                return [[1.0, 0.0] for _ in texts]

        index.embedder = WrongDimension()
        with self.assertRaisesRegex(ValueError, "dimension"):
            index.search("password")

    def test_rejects_empty_corpus(self):
        with self.assertRaisesRegex(ValueError, "without code chunks"):
            SemanticIndex.build([], embedder=FakeEmbedder())


if __name__ == "__main__":
    unittest.main()
