from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.models.schemas import CodeChunk
from app.retrieval.candidate_fusion import fuse_candidates
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.lexical_retriever import LexicalResult
from app.retrieval.semantic_retriever import SemanticResult


def chunk(chunk_id: str, path: str) -> CodeChunk:
    return CodeChunk(chunk_id, "repo", path, "Python", chunk_id, "function", 1, 2,
                     f"def {chunk_id}(): pass")


class FixedRetriever:
    def __init__(self, results):
        self.results = results
        self.last_call = None

    def search(self, query, top_k=10):
        self.last_call = (query, top_k)
        return self.results[:top_k]


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.lex_best = chunk("lex_only", "a.py")
        self.shared = chunk("shared", "b.py")
        self.sem_only = chunk("sem_only", "c.py")

    def test_minmax_fusion_merges_deduplicates_and_preserves_scores(self):
        lexical = [LexicalResult(1, self.lex_best, 10.0), LexicalResult(2, self.shared, 2.0)]
        semantic = [SemanticResult(1, self.shared, 0.95), SemanticResult(2, self.sem_only, 0.10)]
        results = fuse_candidates(lexical, semantic, lexical_weight=0.25, semantic_weight=0.75)

        self.assertEqual([item.chunk.chunk_id for item in results], ["shared", "lex_only", "sem_only"])
        self.assertEqual(len({item.chunk.chunk_id for item in results}), 3)
        self.assertAlmostEqual(results[0].score, 0.75)
        self.assertEqual(results[0].lexical_score, 2.0)
        self.assertEqual(results[0].semantic_score, 0.95)
        self.assertAlmostEqual(results[0].normalized_lexical_score, 0.0)
        self.assertAlmostEqual(results[0].normalized_semantic_score, 1.0)
        self.assertIsNone(results[1].semantic_score)

    def test_weights_control_the_fused_order(self):
        lexical = [LexicalResult(1, self.lex_best, 9), LexicalResult(2, self.shared, 1)]
        semantic = [SemanticResult(1, self.shared, 0.9), SemanticResult(2, self.lex_best, 0.1)]
        lex_first = fuse_candidates(lexical, semantic, lexical_weight=1, semantic_weight=0)
        sem_first = fuse_candidates(lexical, semantic, lexical_weight=0, semantic_weight=1)
        self.assertEqual(lex_first[0].chunk.chunk_id, "lex_only")
        self.assertEqual(sem_first[0].chunk.chunk_id, "shared")

    def test_tied_single_item_normalizes_to_full_signal(self):
        results = fuse_candidates([LexicalResult(1, self.lex_best, 3.5)], [], 1, 1)
        self.assertEqual(results[0].normalized_lexical_score, 1.0)
        self.assertEqual(results[0].score, 0.5)

    def test_hybrid_retriever_limits_candidates_and_final_top_k(self):
        lexical = FixedRetriever([
            LexicalResult(1, self.lex_best, 10), LexicalResult(2, self.shared, 5),
        ])
        semantic = FixedRetriever([
            SemanticResult(1, self.shared, 0.9), SemanticResult(2, self.sem_only, 0.4),
        ])
        retriever = HybridRetriever(lexical, semantic)
        results = retriever.search("query", top_k=2, candidate_k=1)
        self.assertEqual(lexical.last_call, ("query", 1))
        self.assertEqual(semantic.last_call, ("query", 1))
        self.assertEqual(len(results), 2)

    def test_invalid_weights_and_limits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            fuse_candidates([], [], 0, 0)
        with self.assertRaisesRegex(ValueError, "negative"):
            HybridRetriever(FixedRetriever([]), FixedRetriever([]), -0.1, 1.1)
        retriever = HybridRetriever(FixedRetriever([]), FixedRetriever([]))
        with self.assertRaisesRegex(ValueError, "candidate_k"):
            retriever.search("query", candidate_k=-1)


if __name__ == "__main__":
    unittest.main()
