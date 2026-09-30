from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.api.service import CodeLensService
from app.main import create_app


@unittest.skipUnless(__import__("shutil").which("git"), "Git executable is required")
class APITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.repository = root / "repo"
        self.repository.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.name", "CodeLens API Test")
        self.git("config", "user.email", "codelens-api@example.invalid")
        (self.repository / "auth.py").write_text(
            'def authorize_request(bearer_token):\n'
            '    """Validate bearer token before a protected endpoint is served."""\n'
            '    return decode_session_token(bearer_token)\n\n'
            'def decode_session_token(token):\n'
            '    """Decode a signed session token."""\n'
            '    return {"subject": token}\n\n'
            'def parse_authorization(headers):\n'
            '    """Extract a bearer credential from request headers."""\n'
            '    return headers.get("Authorization", "").removeprefix("Bearer ")\n',
            encoding="utf-8",
        )
        (self.repository / "accounts.py").write_text(
            'def hash_password(password, salt):\n'
            '    """Hash an account password with a salt before storage."""\n'
            '    return hashlib.sha256((salt + password).encode()).hexdigest()\n',
            encoding="utf-8",
        )
        (self.repository / "media.py").write_text(
            'def resize_profile_image(image, width, height):\n'
            '    """Resize an uploaded profile image to the requested dimensions."""\n'
            '    return image.resize((width, height))\n',
            encoding="utf-8",
        )
        (self.repository / "payments.py").write_text(
            'def capture_payment(card, amount):\n'
            '    """Check the amount, then charge the payment card."""\n'
            '    if amount <= 0:\n'
            '        raise ValueError("invalid amount")\n'
            '    return charge_card(card, amount)\n',
            encoding="utf-8",
        )
        self.git("add", ".")
        self.git("commit", "-m", "initial retrieval fixture")
        self.v1_commit = self.git("rev-parse", "HEAD").strip()

        service = CodeLensService(root / "indexes", root / "versions")
        app = create_app(service=service, cors_origins=["http://localhost:5173"])
        self.client = TestClient(app)
        response = self.client.post("/api/repositories/index", json={
            "repository_path": str(self.repository), "name": "api-fixture", "build_graph": False,
        })
        self.assertEqual(response.status_code, 201, response.text)
        self.repository_data = response.json()

    def git(self, *args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(self.repository), *args],
            check=True, capture_output=True, text=True,
        )
        return completed.stdout

    def test_health_repositories_and_cors(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(health.json()["indexed_repositories"], 1)

        repositories = self.client.get("/api/repositories")
        self.assertEqual(repositories.status_code, 200)
        self.assertEqual(repositories.json()["repositories"][0]["number_of_chunks"], 6)

        preflight = self.client.options("/api/search", headers={
            "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(preflight.headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_five_natural_language_queries_return_source_from_index(self):
        examples = [
            ("Where is a bearer token validated before protected endpoint access?", "authorize_request"),
            ("How are account passwords hashed with salt before storage?", "hash_password"),
            ("Where is an uploaded profile image resized?", "resize_profile_image"),
            ("Which function checks the amount and charges a payment card?", "capture_payment"),
            ("How is a bearer credential extracted from request headers?", "parse_authorization"),
        ]
        for query, expected_symbol in examples:
            with self.subTest(query=query):
                response = self.client.post("/api/search", json={
                    "query": query, "repository": self.repository_data["repository_id"],
                    "top_k": 5, "retrieval_method": "bm25",
                })
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                self.assertGreaterEqual(payload["retrieval_latency_ms"], 0)
                self.assertTrue(payload["results"])
                match = next((item for item in payload["results"] if item["symbol_name"] == expected_symbol), None)
                self.assertIsNotNone(match, f"{expected_symbol} absent for query: {query}")
                self.assertIn(match["code"], (self.repository / match["file_path"]).read_text(encoding="utf-8"))
                self.assertEqual(match["retrieval_method"], "bm25")
                self.assertGreater(match["end_line"], 0)

    def test_search_validates_and_reports_missing_indexes(self):
        invalid = self.client.post("/api/search", json={"query": "   "})
        self.assertEqual(invalid.status_code, 422)
        missing = self.client.post("/api/search", json={
            "query": "token", "repository": "missing-repository", "retrieval_method": "bm25",
        })
        self.assertEqual(missing.status_code, 404)
        unsupported = self.client.post("/api/search", json={
            "query": "token", "repository": self.repository_data["repository_id"],
            "retrieval_method": "semantic",
        })
        self.assertEqual(unsupported.status_code, 404)

    def test_graph_endpoint_returns_only_indexed_code_relationships(self):
        result = self.client.post("/api/search", json={
            "query": "Where is a bearer token validated before protected endpoint access?",
            "repository": self.repository_data["repository_id"], "top_k": 5, "retrieval_method": "bm25",
        }).json()["results"][0]
        graph = self.client.get("/api/graph", params={
            "repository": self.repository_data["repository_id"], "chunk_id": result["chunk_id"],
        })
        self.assertEqual(graph.status_code, 200, graph.text)
        payload = graph.json()
        self.assertEqual(payload["selected_chunk_id"], result["chunk_id"])
        self.assertTrue(any(node.get("selected") for node in payload["nodes"]))
        self.assertTrue(payload["edges"])
        valid_relations = {"CALLS", "CONTAINS", "REFERENCES", "INHERITS", "IMPORTS"}
        self.assertTrue(all(set(edge["relation"].split(",")).issubset(valid_relations) for edge in payload["edges"]))

    def test_hosted_read_only_mode_blocks_filesystem_indexing(self):
        with patch.dict("os.environ", {"CODELENS_READ_ONLY": "true"}):
            repository = self.client.post("/api/repositories/index", json={
                "repository_path": str(self.repository), "build_graph": True,
            })
            version = self.client.post("/api/versions/index", json={
                "repository_path": str(self.repository), "revision": self.v1_commit,
            })
        self.assertEqual(repository.status_code, 403)
        self.assertEqual(version.status_code, 403)

    def test_version_index_listing_search_and_evolution(self):
        v1 = self.client.post("/api/versions/index", json={
            "repository_path": str(self.repository), "revision": self.v1_commit, "label": "v1",
        })
        self.assertEqual(v1.status_code, 201, v1.text)
        self.assertEqual(v1.json()["version"]["label"], "v1")

        auth_path = self.repository / "auth.py"
        auth_path.write_text(auth_path.read_text(encoding="utf-8").replace(
            "authorize_request", "authenticate_request"), encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-m", "rename authentication entry point")
        v2_commit = self.git("rev-parse", "HEAD").strip()
        v2 = self.client.post("/api/versions/index", json={
            "repository_path": str(self.repository), "revision": v2_commit, "label": "v2",
        })
        self.assertEqual(v2.status_code, 201, v2.text)

        versions = self.client.get("/api/versions", params={"repository": self.repository_data["repository_id"]})
        self.assertEqual(versions.status_code, 200)
        self.assertEqual({item["label"] for item in versions.json()["versions"]}, {"v1", "v2"})

        search = self.client.post("/api/search", json={
            "query": "validate bearer token", "repository": self.repository_data["repository_id"],
            "version": "all", "top_k": 10,
        })
        self.assertEqual(search.status_code, 200, search.text)
        self.assertTrue(any(item["version_label"] == "v1" for item in search.json()["results"]))

        evolution = self.client.post("/api/evolution", json={
            "query": "validate bearer token before protected endpoint",
            "repository": self.repository_data["repository_id"],
            "retrieval_mode": "lexical", "min_similarity": 0.4,
        })
        self.assertEqual(evolution.status_code, 200, evolution.text)
        self.assertEqual(set(evolution.json()["versions_searched"]), {"v1", "v2"})
        self.assertTrue(evolution.json()["tracks"])

    def test_evaluation_uses_explicit_judgments_and_real_results(self):
        search = self.client.post("/api/search", json={
            "query": "password hash salt", "repository": self.repository_data["repository_id"],
            "top_k": 10, "retrieval_method": "bm25",
        }).json()
        relevant = next(item for item in search["results"] if item["symbol_name"] == "hash_password")
        evaluation = self.client.post("/api/evaluate", json={
            "repository": self.repository_data["repository_id"], "retrieval_method": "bm25",
            "cases": [{"query": "password hash salt", "relevance": {relevant["chunk_id"]: 2}}],
        })
        self.assertEqual(evaluation.status_code, 200, evaluation.text)
        result = evaluation.json()
        self.assertGreaterEqual(result["ndcg_at_10"], 0)
        self.assertLessEqual(result["ndcg_at_10"], 1)
        self.assertGreater(result["mrr"], 0)
        self.assertGreaterEqual(result["mean_latency_ms"], 0)


if __name__ == "__main__":
    unittest.main()
