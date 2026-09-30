from __future__ import annotations

import ast
import math
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from app.embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder,
)
from app.indexing.bm25_index import tokenize
from app.metadata.extractor import searchable_text
from app.models.schemas import CodeChunk
from app.versions.version_manager import VersionRecord


@dataclass(slots=True)
class VersionSimilarityWeights:
    semantic: float = 0.55
    file_path: float = 0.15
    symbol: float = 0.10
    structure: float = 0.12
    git: float = 0.08

    def validate(self) -> None:
        values = asdict(self)
        if any(weight < 0 for weight in values.values()):
            raise ValueError("Version similarity weights must not be negative")
        if sum(values.values()) <= 0:
            raise ValueError("At least one version similarity weight must be greater than zero")


@dataclass(slots=True)
class VersionMatch:
    source: CodeChunk
    target: CodeChunk
    similarity_score: float
    components: dict[str, float]


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _path_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.casefold().replace("\\", "/"),
                           right.casefold().replace("\\", "/")).ratio()


def _symbol_similarity(left: str, right: str) -> float:
    return _jaccard(set(tokenize(left)), set(tokenize(right)))


def _structural_signature(chunk: CodeChunk) -> tuple[str, Counter[str], int | None, int]:
    try:
        tree = ast.parse(chunk.code)
    except SyntaxError:
        return chunk.symbol_type, Counter(), None, len(chunk.calls)
    root = next((node for node in tree.body if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), None)
    if root is None:
        return chunk.symbol_type, Counter(), None, len(chunk.calls)
    shape = Counter(type(node).__name__ for node in ast.walk(root))
    arity: int | None = None
    if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = root.args
        arity = len(arguments.posonlyargs) + len(arguments.args) + len(arguments.kwonlyargs)
        if arguments.vararg:
            arity += 1
        if arguments.kwarg:
            arity += 1
    return chunk.symbol_type, shape, arity, len(chunk.calls)


def _counter_jaccard(left: Counter[str], right: Counter[str]) -> float:
    terms = set(left) | set(right)
    if not terms:
        return 0.0
    intersection = sum(min(left[term], right[term]) for term in terms)
    union = sum(max(left[term], right[term]) for term in terms)
    return intersection / union if union else 0.0


def _structure_similarity(left, right) -> float:
    left_type, left_shape, left_arity, left_calls = left
    right_type, right_shape, right_arity, right_calls = right
    if left_type == right_type:
        type_score = 1.0
    elif {left_type, right_type} <= {"function", "method"}:
        type_score = 0.5
    else:
        type_score = 0.0
    tree_score = _counter_jaccard(left_shape, right_shape)
    if left_arity is None or right_arity is None:
        arity_score = 0.5
    else:
        arity_score = 1.0 - abs(left_arity - right_arity) / max(left_arity, right_arity, 1)
    call_score = 1.0 - abs(left_calls - right_calls) / max(left_calls, right_calls, 1)
    return 0.45 * type_score + 0.30 * tree_score + 0.15 * arity_score + 0.10 * call_score


def _git_similarity(source: CodeChunk, target: CodeChunk,
                    source_version: VersionRecord, target_version: VersionRecord) -> float:
    if source.repository_id != target.repository_id:
        return 0.0
    adjacent = (source_version.commit_hash in target_version.parents
                or target_version.commit_hash in source_version.parents)
    renamed = False
    if source_version.commit_hash in target_version.parents:
        renamed = target_version.renames.get(source.file_path) == target.file_path
    elif target_version.commit_hash in source_version.parents:
        renamed = source_version.renames.get(target.file_path) == source.file_path
    if renamed:
        return 1.0
    if adjacent and source.file_path == target.file_path:
        return 0.75
    if adjacent:
        return 0.25
    return 0.4 if source.file_path == target.file_path else 0.0


def _as_rows(vectors) -> list[list[float]]:
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    rows = [[float(value) for value in row] for row in vectors]
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Embedding model returned vectors with inconsistent dimensions")
    return rows


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding model returned inconsistent dimensions across versions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class VersionSimilarityMatcher:
    """Match likely equivalent code units across two Git versions."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL,
                 weights: VersionSimilarityWeights | None = None,
                 embedder=None):
        self.model_name = model_name
        self.weights = weights or VersionSimilarityWeights()
        self.weights.validate()
        self.embedder = embedder
        if self.weights.semantic > 0 and self.embedder is None:
            self.embedder = SentenceTransformerEmbedder(model_name)

    def compare(self, source_version: VersionRecord, target_version: VersionRecord,
                source_chunks: list[CodeChunk], target_chunks: list[CodeChunk],
                matches_per_chunk: int = 3, min_similarity: float = 0.0,
                candidate_pool_size: int = 50) -> list[VersionMatch]:
        if matches_per_chunk < 0 or candidate_pool_size < 1:
            raise ValueError("matches_per_chunk must not be negative and candidate_pool_size must be positive")
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between zero and one")
        if not source_chunks or not target_chunks or matches_per_chunk == 0:
            return []

        source_structures = [_structural_signature(chunk) for chunk in source_chunks]
        target_structures = [_structural_signature(chunk) for chunk in target_chunks]
        source_tokens = [set(tokenize(searchable_text(chunk))) for chunk in source_chunks]
        target_tokens = [set(tokenize(searchable_text(chunk))) for chunk in target_chunks]
        candidates_by_source: list[list[int]] = []
        for source_index, source in enumerate(source_chunks):
            cheap: list[tuple[float, int]] = []
            for target_index, target in enumerate(target_chunks):
                path_score = _path_similarity(source.file_path, target.file_path)
                symbol_score = _symbol_similarity(source.symbol_name, target.symbol_name)
                structure_score = _structure_similarity(
                    source_structures[source_index], target_structures[target_index],
                )
                git_score = _git_similarity(source, target, source_version, target_version)
                text_overlap = _jaccard(source_tokens[source_index], target_tokens[target_index])
                shortlist_score = (0.20 * path_score + 0.22 * symbol_score
                                   + 0.18 * structure_score + 0.25 * text_overlap
                                   + 0.15 * git_score)
                cheap.append((shortlist_score, target_index))
            cheap.sort(key=lambda item: (-item[0], target_chunks[item[1]].chunk_id))
            candidates_by_source.append([index for _, index in cheap[:candidate_pool_size]])

        source_vectors: list[list[float]] = []
        target_vectors: list[list[float]] = []
        if self.weights.semantic > 0:
            model_texts = [searchable_text(chunk) for chunk in [*source_chunks, *target_chunks]]
            vectors = _as_rows(self.embedder.encode(model_texts))
            if len(vectors) != len(model_texts):
                raise ValueError("Embedding model returned an unexpected number of vectors")
            source_vectors = vectors[:len(source_chunks)]
            target_vectors = vectors[len(source_chunks):]

        weights = asdict(self.weights)
        total_weight = sum(weights.values())
        matches: list[VersionMatch] = []
        for source_index, source in enumerate(source_chunks):
            ranked: list[VersionMatch] = []
            for target_index in candidates_by_source[source_index]:
                target = target_chunks[target_index]
                if self.weights.semantic > 0:
                    cosine = _cosine(source_vectors[source_index], target_vectors[target_index])
                    semantic_score = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
                else:
                    semantic_score = 0.0
                components = {
                    "semantic": semantic_score,
                    "file_path": _path_similarity(source.file_path, target.file_path),
                    "symbol": _symbol_similarity(source.symbol_name, target.symbol_name),
                    "structure": _structure_similarity(
                        source_structures[source_index], target_structures[target_index],
                    ),
                    "git": _git_similarity(source, target, source_version, target_version),
                }
                score = sum(weights[name] * components[name] for name in weights) / total_weight
                if score >= min_similarity:
                    ranked.append(VersionMatch(source, target, score, components))
            ranked.sort(key=lambda item: (-item.similarity_score, item.target.chunk_id))
            matches.extend(ranked[:matches_per_chunk])
        return matches
