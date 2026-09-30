from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.indexing.index_manager import index_repository, write_jsonl
from app.ingestion.repository_loader import RepositoryError, discover_source_files
from app.metadata.extractor import searchable_text
from app.parsing.python_parser import parse_python


class ParserTests(unittest.TestCase):
    def test_python_ast_chunks_functions_classes_methods_and_metadata(self) -> None:
        source = '''import hashlib\n\nclass PasswordStore:\n    """Store credentials safely."""\n    def hash_password(self, password):\n        return hashlib.sha256(password.encode()).hexdigest()\n\ndef hash_value(value):\n    return str(value)\n'''
        chunks = parse_python(source, "users/passwords.py", "repo")
        self.assertEqual([chunk.symbol_type for chunk in chunks], ["class", "method", "function"])
        method = chunks[1]
        self.assertEqual(method.symbol_name, "PasswordStore.hash_password")
        self.assertEqual(method.class_name, "PasswordStore")
        self.assertEqual(method.imports, ["hashlib"])
        self.assertEqual(method.calls, ["hashlib.sha256", "hexdigest", "password.encode"])
        self.assertIn("SYMBOL: PasswordStore.hash_password", searchable_text(method))

    def test_malformed_python_returns_no_chunks(self) -> None:
        warnings: list[str] = []
        self.assertEqual(parse_python("def broken(:\n", "broken.py", "repo", warnings), [])
        self.assertIn("broken.py: Python syntax error at line 1", warnings[0])


class IngestionTests(unittest.TestCase):
    def test_filters_generated_directories_and_indexes_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "node_modules" / "pkg").mkdir(parents=True)
            (root / "src" / "example.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
            (root / "node_modules" / "pkg" / "ignored.py").write_text("def ignored(): pass\n", encoding="utf-8")
            (root / "notes.txt").write_text("not source", encoding="utf-8")

            discovered_root, files = discover_source_files(root)
            self.assertEqual(discovered_root, root.resolve())
            self.assertEqual([path.name for path in files], ["example.py"])
            summary = index_repository(root)
            self.assertEqual(summary.number_of_files, 1)
            self.assertEqual(summary.number_of_chunks, 1)
            self.assertEqual(summary.chunks[0].symbol_name, "hello")

            output = write_jsonl(summary, root / "out" / "chunks.jsonl")
            stored = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(stored["chunk_id"], summary.chunks[0].chunk_id)

    def test_invalid_repository_path_has_clear_error(self) -> None:
        with self.assertRaisesRegex(RepositoryError, "does not exist"):
            discover_source_files(Path("missing-repository-for-codelens"))


if __name__ == "__main__":
    unittest.main()
