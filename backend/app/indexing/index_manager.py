from __future__ import annotations

import json
import hashlib
from pathlib import Path

from app.ingestion.repository_loader import (
    discover_source_files, read_source_files, repository_id,
)
from app.models.schemas import CodeChunk, IndexSummary
from app.parsing.code_parser import parse_file


def index_repository(repository_path: str | Path, *, repository_id_override: str | None = None,
                     version_id: str | None = None, version_label: str | None = None,
                     commit_hash: str | None = None) -> IndexSummary:
    root, files = discover_source_files(repository_path)
    repo_id = repository_id_override or repository_id(root)
    chunks: list[CodeChunk] = []
    warnings: list[str] = []
    read_count = 0
    for path, source in read_source_files(files):
        read_count += 1
        relative = path.relative_to(root).as_posix()
        file_chunks = parse_file(source, path, relative, repo_id, warnings)
        for chunk in file_chunks:
            if version_id:
                chunk.chunk_id = hashlib.sha1(
                    f"{chunk.chunk_id}:{version_id}".encode("utf-8")
                ).hexdigest()[:20]
            chunk.version_id = version_id
            chunk.version_label = version_label
            chunk.commit_hash = commit_hash
        chunks.extend(file_chunks)
    return IndexSummary(
        repository_id=repo_id, repository_path=str(root),
        number_of_files=read_count, number_of_chunks=len(chunks),
        skipped_files=len(files) - read_count, warnings=warnings, chunks=chunks,
        version_id=version_id, version_label=version_label, commit_hash=commit_hash,
    )


def write_jsonl(summary: IndexSummary, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in summary.chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return destination
