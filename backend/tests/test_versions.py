from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.versions.similarity import VersionSimilarityMatcher
from app.versions.evolution import EvolutionaryRetriever
from app.versions.version_manager import VersionManager
from app.versions.version_search import VersionSearchEngine


class FakeEmbedder:
    model_name = "fake-version-model"

    def encode(self, texts):
        return [[1.0, 0.0] if "validate jwt token before protected route" in text.lower()
                else [0.0, 1.0] for text in texts]


@unittest.skipUnless(shutil.which("git"), "Git executable is required")
class VersionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.name", "CodeLens Test")
        self.git("config", "user.email", "codelens@example.invalid")
        (self.root / "src").mkdir()
        (self.root / "src" / "auth.py").write_text(
            'def authenticate(token):\n'
            '    """Validate JWT token before protected route."""\n'
            '    return validate_token(token)\n\n'
            'def validate_token(token):\n'
            '    return bool(token)\n', encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-m", "first authentication flow")
        self.v1_commit = self.git("rev-parse", "HEAD").strip()
        self.git("tag", "v1")

        (self.root / "src" / "auth.py").write_text(
            'def authenticate_request(token):\n'
            '    """Validate JWT token before protected route."""\n'
            '    return validate_token(token)\n\n'
            'def validate_token(token):\n'
            '    return bool(token)\n', encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-m", "rename authentication entry point")
        self.v2_commit = self.git("rev-parse", "HEAD").strip()
        self.git("tag", "v2")

        self.output_dir = Path(self.temporary.name) / "indexes"
        manager = VersionManager(self.output_dir)
        self.catalog_path, self.v1 = manager.index_git_version(self.root, "v1", label="v1")
        second_path, self.v2 = manager.index_git_version(self.root, "v2", label="v2")
        self.assertEqual(self.catalog_path, second_path)

    def git(self, *args: str) -> str:
        completed = subprocess.run(["git", "-C", str(self.root), *args], check=True,
                                   capture_output=True, text=True)
        return completed.stdout

    def test_indexes_isolated_commits_with_stable_repo_and_version_ids(self):
        self.assertEqual(self.v1.commit_hash, self.v1_commit)
        self.assertEqual(self.v2.commit_hash, self.v2_commit)
        self.assertEqual(self.v1.version_id, self.v1.commit_hash)
        self.assertEqual(self.v2.parents, [self.v1_commit])
        self.assertTrue((self.catalog_path.parent / self.v1.chunks_path).is_file())
        self.assertTrue((self.catalog_path.parent / self.v2.bm25_index_path).is_file())

        from app.retrieval.lexical_retriever import load_chunks_jsonl
        first = load_chunks_jsonl(self.catalog_path.parent / self.v1.chunks_path)
        second = load_chunks_jsonl(self.catalog_path.parent / self.v2.chunks_path)
        self.assertTrue(all(chunk.repository_id == first[0].repository_id for chunk in second))
        self.assertTrue(all(chunk.version_id == self.v1_commit for chunk in first))
        self.assertNotEqual(first[0].chunk_id, second[0].chunk_id)

    def test_searches_one_version_or_all_versions(self):
        engine = VersionSearchEngine(self.catalog_path)
        only = engine.search("validate JWT token", version="v1", mode="lexical")
        self.assertEqual(only.versions_searched, ["v1"])
        self.assertTrue(only.results)
        self.assertTrue(all(result.version_label == "v1" for result in only.results))

        cross = engine.search("validate JWT token", mode="auto", top_k=10)
        self.assertEqual(cross.retrieval_mode, "lexical")
        self.assertEqual(set(cross.versions_searched), {"v1", "v2"})
        self.assertEqual({result.version_label for result in cross.results}, {"v1", "v2"})

    def test_hybrid_mode_requires_semantic_index_for_every_selected_version(self):
        engine = VersionSearchEngine(self.catalog_path)
        with self.assertRaisesRegex(ValueError, "requires a semantic index"):
            engine.search("token", mode="hybrid")

    def test_similarity_matches_renamed_function_using_multiple_signals(self):
        from app.retrieval.lexical_retriever import load_chunks_jsonl
        first = load_chunks_jsonl(self.catalog_path.parent / self.v1.chunks_path)
        second = load_chunks_jsonl(self.catalog_path.parent / self.v2.chunks_path)
        matcher = VersionSimilarityMatcher(embedder=FakeEmbedder())
        matches = matcher.compare(self.v1, self.v2, first, second, matches_per_chunk=1)
        renamed = next(match for match in matches if match.source.symbol_name == "authenticate")
        self.assertEqual(renamed.target.symbol_name, "authenticate_request")
        self.assertGreater(renamed.components["semantic"], 0.9)
        self.assertEqual(renamed.components["git"], 0.75)
        self.assertGreater(renamed.similarity_score, 0.5)

    def test_evolutionary_search_tracks_query_relevant_rename_across_versions(self):
        retriever = EvolutionaryRetriever(self.catalog_path)
        report = retriever.search(
            "Validate JWT token before protected route", per_version_top_k=10,
            retrieval_mode="lexical", min_similarity=0.4,
        )
        self.assertEqual(report.versions_searched, ["v1", "v2"])
        self.assertEqual(report.retrieval_mode, "lexical")
        renamed_tracks = [track for track in report.tracks
                          if any(item.symbol_name == "authenticate" for item in track.items)
                          and any(item.symbol_name == "authenticate_request" for item in track.items)]
        self.assertTrue(renamed_tracks)
        transition = renamed_tracks[0].transitions[0]
        self.assertEqual(transition.change, "likely_renamed_or_moved")
        self.assertGreater(transition.similarity_score, 0.4)

    def test_evolutionary_search_validates_query_and_limits(self):
        retriever = EvolutionaryRetriever(self.catalog_path)
        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            retriever.search("   ")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            retriever.search("token", max_tracks=0)


if __name__ == "__main__":
    unittest.main()
