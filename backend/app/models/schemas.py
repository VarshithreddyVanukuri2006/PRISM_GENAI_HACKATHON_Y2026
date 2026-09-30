from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CodeChunk:
    chunk_id: str
    repository_id: str
    file_path: str
    language: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    code: str
    docstring: str = ""
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    class_name: str | None = None
    parent_symbol: str | None = None
    version_id: str | None = None
    version_label: str | None = None
    commit_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IndexSummary:
    repository_id: str
    repository_path: str
    number_of_files: int
    number_of_chunks: int
    skipped_files: int
    warnings: list[str] = field(default_factory=list)
    chunks: list[CodeChunk] = field(default_factory=list, repr=False)
    version_id: str | None = None
    version_label: str | None = None
    commit_hash: str | None = None
