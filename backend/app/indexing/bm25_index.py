from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.metadata.extractor import searchable_text
from app.models.schemas import CodeChunk


def tokenize(text: str) -> list[str]:
    """Tokenize prose and code identifiers, including snake_case and camelCase."""
    expanded: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalnum() or char == "_":
            current.append(char)
        elif current:
            expanded.append("".join(current))
            current.clear()
    if current:
        expanded.append("".join(current))

    tokens: list[str] = []
    for raw_part in expanded:
        raw_part = raw_part.strip("_")
        part = raw_part.lower()
        if not part:
            continue
        tokens.append(part)
        # Preserve the full identifier and add useful subword terms.
        for subpart in raw_part.split("_"):
            if subpart and subpart not in tokens[-1:]:
                tokens.append(subpart.lower())
        for identifier_part in raw_part.split("_"):
            camel_parts = []
            start = 0
            for index in range(1, len(identifier_part)):
                if identifier_part[index].isupper():
                    camel_parts.append(identifier_part[start:index].lower())
                    start = index
            if camel_parts:
                camel_parts.append(identifier_part[start:].lower())
                tokens.extend(camel_parts)
    return tokens


@dataclass(slots=True)
class BM25Index:
    chunks: list[CodeChunk]
    document_tokens: list[list[str]]
    document_frequencies: dict[str, int]
    average_document_length: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, chunks: Iterable[CodeChunk], k1: float = 1.5,
              b: float = 0.75) -> "BM25Index":
        if k1 <= 0:
            raise ValueError("BM25 k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between zero and one")
        chunk_list = list(chunks)
        documents = [tokenize(searchable_text(chunk)) for chunk in chunk_list]
        frequencies: Counter[str] = Counter()
        for document in documents:
            frequencies.update(set(document))
        total_tokens = sum(map(len, documents))
        average = total_tokens / len(documents) if documents else 0.0
        return cls(chunk_list, documents, dict(frequencies), average, k1, b)

    def search(self, query: str, top_k: int = 10) -> list[tuple[CodeChunk, float]]:
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        terms = tokenize(query)
        if not terms or not self.chunks or top_k == 0:
            return []

        count = len(self.chunks)
        scored: list[tuple[int, float]] = []
        for doc_id, document in enumerate(self.document_tokens):
            term_counts = Counter(document)
            length = len(document)
            score = 0.0
            for term in terms:
                tf = term_counts.get(term, 0)
                if not tf:
                    continue
                df = self.document_frequencies.get(term, 0)
                # Positive Robertson/Sparck Jones IDF avoids negative relevance
                # values for terms occurring in more than half of the corpus.
                idf = math.log1p((count - df + 0.5) / (df + 0.5))
                norm = tf + self.k1 * (
                    1 - self.b + self.b * length / (self.average_document_length or 1.0)
                )
                score += idf * (tf * (self.k1 + 1) / norm)
            if score > 0:
                scored.append((doc_id, score))

        scored.sort(key=lambda item: (
            -item[1], self.chunks[item[0]].file_path.casefold(),
            self.chunks[item[0]].start_line, self.chunks[item[0]].chunk_id,
        ))
        return [(self.chunks[doc_id], score) for doc_id, score in scored[:top_k]]

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "k1": self.k1,
            "b": self.b,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "document_tokens": self.document_tokens,
            "document_frequencies": self.document_frequencies,
            "average_document_length": self.average_document_length,
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if payload.get("format_version") != 1:
                raise ValueError("unsupported BM25 index format version")
            chunks = [CodeChunk(**item) for item in payload["chunks"]]
            documents = payload["document_tokens"]
            frequencies = payload["document_frequencies"]
            average = payload["average_document_length"]
            k1 = payload["k1"]
            b = payload["b"]
            if len(chunks) != len(documents):
                raise ValueError("chunk and token document counts do not match")
            if k1 <= 0 or not 0 <= b <= 1:
                raise ValueError("invalid BM25 parameters")
            return cls(chunks, documents, frequencies, average, k1, b)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Unable to load BM25 index {source}: {exc}") from exc
