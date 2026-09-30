from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.models.schemas import CodeChunk
from app.retrieval.candidate_fusion import FusedCandidate
from app.retrieval.reranker import CodeAwareReranker, RerankerWeights


def candidate(chunk_id: str, path: str, symbol: str, language: str = "Python",
              docstring: str = "", imports=None, calls=None) -> FusedCandidate:
    chunk = CodeChunk(
        chunk_id=chunk_id, repository_id="repo", file_path=path, language=language,
        symbol_name=symbol, symbol_type="function", start_line=1, end_line=3,
        code=f"def {symbol}(): pass", docstring=docstring,
        imports=imports or [], calls=calls or [],
    )
    return FusedCandidate(chunk, 0.5, 0.8, 5.0, 0.7, 0.6, 1, 1)


class RerankerTests(unittest.TestCase):
    def test_extracts_exact_symbol_metadata_and_dependency_signals(self):
        item = candidate(
            "hash_password", "users/hash_password.py", "hash_password",
            docstring="Hash a password before storing credentials.",
            imports=["hashlib"], calls=["hashlib.sha256"],
        )
        reranker = CodeAwareReranker()
        features, signals = reranker.extract_features("Call hash_password to hash a password", item)
        self.assertEqual(features["exact_symbol"], 1.0)
        self.assertGreater(features["filename"], 0)
        self.assertGreater(features["metadata"], 0)
        self.assertIn("Exact symbol name appears in query: hash_password", signals)

    def test_language_weight_ranks_explicit_language_match(self):
        py = candidate("py", "auth/verify.py", "verify_token", "Python")
        java = candidate("java", "auth/Verifier.java", "verifyToken", "Java")
        reranker = CodeAwareReranker(RerankerWeights(
            semantic=0, lexical=0, exact_symbol=0, filename=0, language=1,
            intent=0, metadata=0, dependency=0,
        ))
        results = reranker.rerank("In Java, where is the token verified?", [py, java])
        self.assertEqual(results[0].chunk.language, "Java")
        self.assertEqual(results[0].features["language"], 1.0)
        self.assertEqual(results[1].features["language"], 0.0)

    def test_role_and_dependency_features_use_available_metadata(self):
        matching = candidate("verify", "auth/middleware.py", "verify_token",
                             docstring="Validate JWT token claims.",
                             imports=["jwt"], calls=["decode_jwt"])
        other = candidate("view", "ui/page.py", "render_page", docstring="Render a page.")
        reranker = CodeAwareReranker()
        matched_features, matched_signals = reranker.extract_features("verify JWT token", matching)
        other_features, _ = reranker.extract_features("verify JWT token", other)
        self.assertGreater(matched_features["intent"], other_features["intent"])
        self.assertGreater(matched_features["dependency"], other_features["dependency"])
        self.assertTrue(any(signal.startswith("Import/call terms matched:") for signal in matched_signals))

    def test_weight_ablation_changes_order_and_preserves_hybrid_scores(self):
        explicit = candidate("explicit", "misc.py", "hash_password")
        other = candidate("other", "misc.py", "store_user")
        semantic_only = CodeAwareReranker(RerankerWeights(
            semantic=1, lexical=0, exact_symbol=0, filename=0, language=0,
            intent=0, metadata=0, dependency=0,
        ))
        symbol_only = CodeAwareReranker(RerankerWeights(
            semantic=0, lexical=0, exact_symbol=1, filename=0, language=0,
            intent=0, metadata=0, dependency=0,
        ))
        first = semantic_only.rerank("hash_password", [explicit, other])
        second = symbol_only.rerank("hash_password", [other, explicit])
        self.assertEqual(first[0].chunk.chunk_id, "explicit")
        self.assertEqual(second[0].chunk.chunk_id, "explicit")
        self.assertEqual(second[0].semantic_score, 0.8)
        self.assertEqual(second[0].lexical_score, 5.0)
        self.assertEqual(second[0].hybrid_score, 0.5)

    def test_empty_query_and_invalid_weights(self):
        self.assertEqual(CodeAwareReranker().rerank(" ", [candidate("a", "a.py", "a")]), [])
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            CodeAwareReranker(RerankerWeights(0, 0, 0, 0, 0, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "negative"):
            CodeAwareReranker(RerankerWeights(semantic=-1))


if __name__ == "__main__":
    unittest.main()
