from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.graph.code_graph import CodeGraph, symbol_node_id
from app.models.schemas import CodeChunk
from app.retrieval.candidate_fusion import FusedCandidate
from app.retrieval.lexical_retriever import LexicalResult
from app.retrieval.multi_pass_retriever import MultiPassRetriever
from app.retrieval.reranker import CodeAwareReranker, RerankerWeights
from app.retrieval.semantic_retriever import SemanticResult
from test_code_graph import FakeDiGraph


class ContextRetriever:
    def __init__(self, initial, contextual):
        self.initial = initial
        self.contextual = contextual
        self.queries = []

    def search(self, query, top_k):
        self.queries.append(query)
        result = self.contextual if "create_pool" in query else self.initial
        return result[:top_k]


class MultiPassTests(unittest.TestCase):
    def setUp(self):
        self.execute = CodeChunk("execute", "repo", "db/query.py", "Python", "execute_query", "function",
                                 1, 3, "def execute_query(): return get_connection()",
                                 docstring="Execute a database query", calls=["get_connection"])
        self.connection = CodeChunk("connection", "repo", "db/connection.py", "Python", "get_connection", "function",
                                    1, 3, "def get_connection(): return create_pool()",
                                    docstring="Get a connection from a pool", calls=["create_pool"])
        self.pool = CodeChunk("pool", "repo", "db/pool.py", "Python", "create_pool", "function",
                              1, 3, "def create_pool(): pass", docstring="Create a database connection pool")
        graph = FakeDiGraph()
        for chunk in [self.execute, self.connection, self.pool]:
            graph.add_node(symbol_node_id(chunk.chunk_id), kind="function", chunk_id=chunk.chunk_id,
                           file_path=chunk.file_path, symbol_name=chunk.symbol_name)
        graph.add_edge(symbol_node_id("execute"), symbol_node_id("connection"), relations=["CALLS"])
        graph.add_edge(symbol_node_id("connection"), symbol_node_id("pool"), relations=["CALLS"])
        self.graph = CodeGraph(graph, [self.execute, self.connection, self.pool])

        lexical_initial = [LexicalResult(1, self.execute, 4.0)]
        lexical_context = [LexicalResult(1, self.pool, 3.0)]
        semantic_initial = [SemanticResult(1, self.execute, 0.9)]
        semantic_context = [SemanticResult(1, self.pool, 0.85)]

        class HybridStub:
            lexical_weight = 0.5
            semantic_weight = 0.5

            def __init__(self, lexical, semantic):
                self.lexical = lexical
                self.semantic = semantic

        self.hybrid = HybridStub(
            ContextRetriever(lexical_initial, lexical_context),
            ContextRetriever(semantic_initial, semantic_context),
        )
        weights = RerankerWeights(semantic=0, lexical=0, exact_symbol=0, filename=0,
                                  language=0, intent=0, metadata=0, dependency=0, graph=1)
        self.retriever = MultiPassRetriever(self.hybrid, self.graph, CodeAwareReranker(weights))

    def test_runs_all_five_passes_and_recovers_graph_neighbor(self):
        report = self.retriever.search(
            "Where is the database connection used before query execution?",
            top_k=3, candidate_k=5, seed_k=1, context_candidate_k=5,
            max_graph_hops=3,
        )
        self.assertEqual(report.pass1_candidate_count, 1)
        self.assertEqual(report.pass2_seed_chunk_ids, ["execute"])
        self.assertEqual({item.chunk_id for item in report.pass3_expansions}, {"connection", "pool"})
        self.assertEqual(report.pass4_candidate_count, 1)
        self.assertEqual(report.pass4_added_chunk_ids, ["pool"])
        self.assertEqual(report.pass5_candidate_count, 3)
        self.assertEqual(report.results[0].chunk.chunk_id, "connection")
        self.assertEqual(report.results[0].features["graph"], 1.0)
        self.assertTrue(any(signal.startswith("Graph path:") for signal in report.results[0].signals))
        self.assertIn("create_pool", self.hybrid.lexical.queries[1])

    def test_empty_query_skips_all_passes(self):
        report = self.retriever.search(" ")
        self.assertEqual(report.pass1_candidate_count, 0)
        self.assertEqual(report.results, [])


if __name__ == "__main__":
    unittest.main()
