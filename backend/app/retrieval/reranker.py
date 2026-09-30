from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Sequence

from app.indexing.bm25_index import tokenize
from app.models.schemas import CodeChunk
from app.retrieval.candidate_fusion import FusedCandidate


ROLE_GROUPS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"auth", "authentication", "authorize", "authorization", "token", "jwt",
                "protected", "validate", "validation", "verify", "verified", "permission"}),
     frozenset({"auth", "authentication", "authenticate", "authorization", "authorize",
                "middleware", "token", "jwt", "validate", "validation", "verify",
                "permission", "protected", "claims", "credential"})),
    (frozenset({"payment", "charge", "billing", "card", "transaction", "refund"}),
     frozenset({"payment", "payments", "charge", "card", "billing", "transaction",
                "refund", "invoice"})),
    (frozenset({"database", "db", "query", "sql", "connection", "pool"}),
     frozenset({"database", "db", "query", "sql", "connection", "pool", "execute",
                "cursor", "commit"})),
    (frozenset({"image", "resize", "upload", "storage", "file", "thumbnail"}),
     frozenset({"image", "resize", "upload", "storage", "file", "thumbnail", "bucket",
                "compress"})),
    (frozenset({"password", "hash", "hashed", "credential", "credentials"}),
     frozenset({"password", "hash", "hashed", "credential", "credentials", "salt",
                "digest", "encrypt"})),
    (frozenset({"normalize", "normalization", "sanitize", "parse", "transform", "convert"}),
     frozenset({"normalize", "normalization", "sanitize", "parse", "transform", "convert",
                "clean", "format"})),
)

LANGUAGE_NAMES = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "java": "Java",
    "c++": "C++",
}


@dataclass(slots=True)
class RerankerWeights:
    semantic: float = 0.25
    lexical: float = 0.25
    exact_symbol: float = 0.10
    filename: float = 0.075
    language: float = 0.025
    intent: float = 0.10
    metadata: float = 0.15
    dependency: float = 0.05
    graph: float = 0.0

    def validate(self) -> None:
        values = asdict(self)
        if any(value < 0 for value in values.values()):
            raise ValueError("Reranker weights must not be negative")
        if sum(values.values()) <= 0:
            raise ValueError("At least one reranker weight must be greater than zero")


@dataclass(slots=True)
class RerankedResult:
    rank: int
    chunk: CodeChunk
    final_score: float
    features: dict[str, float]
    signals: list[str]
    semantic_score: float | None
    lexical_score: float | None
    hybrid_score: float


def _query_language(query: str) -> str | None:
    lowered = query.casefold()
    for language in sorted(LANGUAGE_NAMES, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(language)}(?![a-z0-9])", lowered):
            return LANGUAGE_NAMES[language]
    return None


def _identifier_phrase_present(query: str, symbol_name: str) -> bool:
    lowered = query.casefold()
    # A qualified method name should still match when a developer names only the method.
    symbols = {symbol_name.casefold(), symbol_name.rsplit(".", 1)[-1].casefold()}
    # Require identifier-like boundaries so a short symbol does not match arbitrary prose.
    return any(re.search(rf"(?<![a-z0-9_]){re.escape(symbol)}(?![a-z0-9_])", lowered)
               for symbol in symbols)


def _metadata_text(chunk: CodeChunk) -> str:
    values = [chunk.file_path, chunk.language, chunk.symbol_name, chunk.symbol_type,
              chunk.docstring, chunk.class_name or "", chunk.parent_symbol or ""]
    values.extend(chunk.imports)
    values.extend(chunk.calls)
    return " ".join(values)


def _ratio(matched: set[str], denominator: set[str]) -> float:
    return len(matched) / len(denominator) if denominator else 0.0


class CodeAwareReranker:
    """Configurable weighted scorer over hybrid results and code metadata."""

    def __init__(self, weights: RerankerWeights | None = None):
        self.weights = weights or RerankerWeights()
        self.weights.validate()

    def extract_features(self, query: str, candidate: FusedCandidate,
                         graph_score: float = 0.0) -> tuple[dict[str, float], list[str]]:
        chunk = candidate.chunk
        query_tokens = set(tokenize(query))
        metadata_tokens = set(tokenize(_metadata_text(chunk)))
        filename_tokens = set(tokenize(chunk.file_path))
        dependency_tokens = set(tokenize(" ".join([*chunk.imports, *chunk.calls])))

        symbol_match = _identifier_phrase_present(query, chunk.symbol_name)
        query_language = _query_language(query)
        language_match = query_language is None or chunk.language.casefold() == query_language.casefold()
        role_matches: set[str] = set()
        role_expected: set[str] = set()
        for triggers, role_terms in ROLE_GROUPS:
            if query_tokens.intersection(triggers):
                role_expected.update(role_terms)
        candidate_tokens = metadata_tokens
        if role_expected:
            role_matches = query_tokens.intersection(candidate_tokens).intersection(role_expected)
            # Query terms themselves can be concrete actions; include their direct matches.
            role_matches.update(candidate_tokens.intersection(role_expected))
        intent_score = _ratio(role_matches, role_expected)

        filename_matches = query_tokens.intersection(filename_tokens)
        metadata_matches = query_tokens.intersection(metadata_tokens)
        dependency_matches = query_tokens.intersection(dependency_tokens)
        features = {
            "semantic": candidate.normalized_semantic_score,
            "lexical": candidate.normalized_lexical_score,
            "exact_symbol": 1.0 if symbol_match else 0.0,
            "filename": _ratio(filename_matches, query_tokens),
            "language": 1.0 if language_match else 0.0,
            "intent": intent_score,
            "metadata": _ratio(metadata_matches, query_tokens),
            "dependency": _ratio(dependency_matches, query_tokens),
            "graph": max(0.0, min(1.0, graph_score)),
        }

        signals: list[str] = []
        if symbol_match:
            signals.append(f"Exact symbol name appears in query: {chunk.symbol_name}")
        if filename_matches:
            signals.append("Filename terms matched: " + ", ".join(sorted(filename_matches)))
        if query_language:
            if language_match:
                signals.append(f"Requested language matches: {chunk.language}")
            else:
                signals.append(f"Requested language does not match: {chunk.language} (query asks for {query_language})")
        else:
            signals.append("Query does not specify a programming language")
        if role_matches:
            signals.append("Function-role terms matched: " + ", ".join(sorted(role_matches)))
        if metadata_matches:
            signals.append("Metadata terms matched: " + ", ".join(sorted(metadata_matches)))
        if dependency_matches:
            signals.append("Import/call terms matched: " + ", ".join(sorted(dependency_matches)))
        return features, signals

    def rerank(self, query: str, candidates: Sequence[FusedCandidate],
               top_k: int = 10, graph_scores: dict[str, float] | None = None,
               graph_paths: dict[str, list[list[str]]] | None = None) -> list[RerankedResult]:
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        if not query.strip() or not candidates or top_k == 0:
            return []

        weights = asdict(self.weights)
        total_weight = sum(weights.values())
        graph_scores = graph_scores or {}
        graph_paths = graph_paths or {}
        scored: list[tuple[FusedCandidate, dict[str, float], list[str], float]] = []
        for candidate in candidates:
            graph_score = graph_scores.get(candidate.chunk.chunk_id, 0.0)
            features, signals = self.extract_features(query, candidate, graph_score)
            for path in graph_paths.get(candidate.chunk.chunk_id, []):
                signals.append("Graph path: " + " / ".join(path))
            final_score = sum(weights[name] * features[name] for name in weights) / total_weight
            scored.append((candidate, features, signals, final_score))
        scored.sort(key=lambda item: (
            -item[3], item[0].chunk.file_path.casefold(), item[0].chunk.start_line,
            item[0].chunk.chunk_id,
        ))
        return [
            RerankedResult(
                rank=rank, chunk=candidate.chunk, final_score=score, features=features,
                signals=signals, semantic_score=candidate.semantic_score,
                lexical_score=candidate.lexical_score, hybrid_score=candidate.score,
            )
            for rank, (candidate, features, signals, score) in enumerate(scored[:top_k], start=1)
        ]
