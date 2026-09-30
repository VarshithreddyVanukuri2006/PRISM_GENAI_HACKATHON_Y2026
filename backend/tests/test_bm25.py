from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.indexing.bm25_index import BM25Index, tokenize
from app.retrieval.lexical_retriever import LexicalRetriever, load_chunks_jsonl
from app.indexing.index_manager import index_repository, write_jsonl


class BM25Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = index_repository(ROOT / "data" / "demo")
        cls.index = BM25Index.build(cls.summary.chunks)

    def test_password_hash_query_retrieves_expected_function_first(self) -> None:
        results = self.index.search("Where is the user's password hashed before storage?", top_k=5)
        self.assertTrue(results)
        self.assertEqual(results[0][0].symbol_name, "hash_password")
        self.assertGreater(results[0][1], 0)

    def test_tokenizer_splits_python_and_camel_case_identifiers(self) -> None:
        tokens = tokenize("verifyUser_token")
        self.assertIn("verifyuser_token", tokens)
        self.assertIn("verify", tokens)
        self.assertIn("user", tokens)
        self.assertIn("token", tokens)

    def test_persisted_index_round_trip_preserves_search_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bm25.json"
            self.index.save(path)
            loaded = BM25Index.load(path)
        original = self.index.search("decode jwt token", top_k=3)
        restored = loaded.search("decode jwt token", top_k=3)
        self.assertEqual([item[0].chunk_id for item in original], [item[0].chunk_id for item in restored])
        self.assertEqual([item[1] for item in original], [item[1] for item in restored])

    def test_empty_or_unmatched_query_returns_no_results(self) -> None:
        self.assertEqual(self.index.search("   "), [])
        self.assertEqual(self.index.search("zzqv987654"), [])

    def test_retriever_reads_phase_a_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_jsonl(self.summary, Path(temporary) / "chunks.jsonl")
            chunks = load_chunks_jsonl(path)
            retriever = LexicalRetriever(BM25Index.build(chunks))
            results = retriever.search("resize uploaded image", top_k=1)
        self.assertEqual(results[0].chunk.symbol_name, "resize_image")
        self.assertEqual(results[0].rank, 1)


if __name__ == "__main__":
    unittest.main()
