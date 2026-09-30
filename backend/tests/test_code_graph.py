from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.graph.code_graph import CodeGraph, file_node_id, symbol_node_id
from app.models.schemas import CodeChunk


class NodeView:
    def __init__(self, graph):
        self.graph = graph

    def __call__(self, data=False):
        if data:
            return list(self.graph._nodes.items())
        return list(self.graph._nodes)

    def __getitem__(self, key):
        return self.graph._nodes[key]


class EdgeView:
    def __init__(self, graph):
        self.graph = graph

    def __call__(self, data=False):
        if data:
            return [(source, target, attributes)
                    for (source, target), attributes in self.graph._edges.items()]
        return list(self.graph._edges)


class FakeDiGraph:
    def __init__(self):
        self._nodes = {}
        self._edges = {}
        self.nodes = NodeView(self)
        self.edges = EdgeView(self)

    def add_node(self, node_id, **attributes):
        self._nodes.setdefault(node_id, {}).update(attributes)

    def add_edge(self, source, target, **attributes):
        self.add_node(source)
        self.add_node(target)
        self._edges.setdefault((source, target), {}).update(attributes)

    def has_edge(self, source, target):
        return (source, target) in self._edges

    def __getitem__(self, source):
        return {target: data for (origin, target), data in self._edges.items() if origin == source}

    def in_edges(self, node, data=False):
        items = [(source, target) for source, target in self._edges if target == node]
        return [(source, target, self._edges[(source, target)]) for source, target in items] if data else items

    def out_edges(self, node, data=False):
        items = [(source, target) for source, target in self._edges if source == node]
        return [(source, target, self._edges[(source, target)]) for source, target in items] if data else items

    def number_of_nodes(self):
        return len(self._nodes)

    def number_of_edges(self):
        return len(self._edges)


class CodeGraphTests(unittest.TestCase):
    def setUp(self):
        self.nx_patch = patch("app.graph.code_graph._load_networkx",
                              return_value=SimpleNamespace(DiGraph=FakeDiGraph))
        self.nx_patch.start()
        self.addCleanup(self.nx_patch.stop)
        self.chunks = [
            CodeChunk("route", "repo", "auth/routes.py", "Python", "protected_route", "function",
                      1, 3, "def protected_route(): return AuthMiddleware.verify_token()",
                      imports=["auth.tokens"], calls=["AuthMiddleware.verify_token"]),
            CodeChunk("auth_base", "repo", "auth/middleware.py", "Python", "AuthBase", "class",
                      1, 1, "class AuthBase: pass"),
            CodeChunk("auth_class", "repo", "auth/middleware.py", "Python", "AuthMiddleware", "class",
                      3, 7, "class AuthMiddleware(AuthBase):\n    def verify_token(self): return decode_jwt()"),
            CodeChunk("verify", "repo", "auth/middleware.py", "Python", "AuthMiddleware.verify_token", "method",
                      4, 4, "def verify_token(self): return decode_jwt()", class_name="AuthMiddleware",
                      parent_symbol="AuthMiddleware", calls=["decode_jwt"], imports=["auth.tokens"]),
            CodeChunk("decode", "repo", "auth/tokens.py", "Python", "decode_jwt", "function",
                      1, 2, "def decode_jwt(): return jwt.decode()", calls=["jwt.decode"]),
        ]

    def test_builds_file_containment_call_and_import_relationships(self):
        code_graph = CodeGraph.from_chunks(self.chunks)
        summary = code_graph.summary()
        self.assertEqual(summary["node_types"]["file"], 3)
        self.assertEqual(summary["relations"]["CONTAINS"], 5)
        self.assertEqual(summary["relations"]["CALLS"], 2)
        self.assertGreaterEqual(summary["relations"]["IMPORTS"], 1)
        self.assertEqual(summary["relations"]["INHERITS"], 1)
        self.assertGreaterEqual(summary["relations"]["REFERENCES"], 1)
        self.assertTrue(code_graph.graph.has_edge(symbol_node_id("route"), symbol_node_id("verify")))
        self.assertTrue(code_graph.graph.has_edge(file_node_id("auth/routes.py"), file_node_id("auth/tokens.py")))
        self.assertIn("INHERITS", code_graph.graph[symbol_node_id("auth_class")][symbol_node_id("auth_base")]["relations"])

    def test_related_to_chunk_reports_only_resolved_direct_edges(self):
        code_graph = CodeGraph.from_chunks(self.chunks)
        relationships = code_graph.related_to_chunk("verify")
        normalized = {(item["direction"], relation, item["node"].get("symbol_name"))
                      for item in relationships for relation in item["relation"].split(",")}
        self.assertIn(("incoming", "CALLS", "protected_route"), normalized)
        self.assertIn(("incoming", "CONTAINS", "AuthMiddleware"), normalized)
        self.assertIn(("outgoing", "CALLS", "decode_jwt"), normalized)

    def test_multi_hop_expansion_follows_calls_with_relationship_paths(self):
        code_graph = CodeGraph.from_chunks(self.chunks)
        expansions = code_graph.expand_from_chunks(["route"], max_hops=3)
        by_id = {item.chunk_id: item for item in expansions}
        self.assertIn("verify", by_id)
        self.assertEqual(by_id["verify"].hops, 1)
        self.assertIn("decode", by_id)
        self.assertEqual(by_id["decode"].hops, 2)
        self.assertTrue(any("CALLS" in step for step in by_id["decode"].relation_path))

    def test_graph_serialization_round_trip(self):
        code_graph = CodeGraph.from_chunks(self.chunks)
        with tempfile.TemporaryDirectory() as temporary:
            path = code_graph.save(Path(temporary) / "graph.json")
            restored = CodeGraph.load(path)
        self.assertEqual(restored.summary(), code_graph.summary())
        self.assertEqual(restored.chunks["verify"].symbol_name, "AuthMiddleware.verify_token")
        self.assertTrue(restored.graph.has_edge(symbol_node_id("verify"), symbol_node_id("decode")))

    def test_ambiguous_calls_are_left_unresolved(self):
        duplicate_a = CodeChunk("a", "repo", "one.py", "Python", "handle", "function", 1, 1, "pass")
        duplicate_b = CodeChunk("b", "repo", "two.py", "Python", "handle", "function", 1, 1, "pass")
        caller = CodeChunk("caller", "repo", "main.py", "Python", "caller", "function", 1, 2,
                           "def caller(): handle()", calls=["handle"])
        graph = CodeGraph.from_chunks([duplicate_a, duplicate_b, caller])
        self.assertFalse(graph.graph.has_edge(symbol_node_id("caller"), symbol_node_id("a")))
        self.assertFalse(graph.graph.has_edge(symbol_node_id("caller"), symbol_node_id("b")))


if __name__ == "__main__":
    unittest.main()
