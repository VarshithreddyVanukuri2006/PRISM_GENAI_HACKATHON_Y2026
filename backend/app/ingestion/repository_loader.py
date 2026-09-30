from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator

from app.config import IGNORED_DIRS, SUPPORTED_EXTENSIONS
from app.ingestion.file_filter import should_skip


class RepositoryError(ValueError):
    """Raised when a repository path cannot be indexed."""


def repository_id(root: Path) -> str:
    canonical = str(root.resolve()).casefold().encode("utf-8")
    return hashlib.sha1(canonical).hexdigest()[:16]


def discover_source_files(repository_path: str | Path) -> tuple[Path, list[Path]]:
    root = Path(repository_path).expanduser()
    if not root.exists():
        raise RepositoryError(f"Repository path does not exist: {root}")
    if not root.is_dir():
        raise RepositoryError(f"Repository path is not a directory: {root}")
    root = root.resolve()

    files: list[Path] = []
    ignored = {entry.casefold() for entry in IGNORED_DIRS}
    for current, directories, filenames in os.walk(root):
        directories[:] = sorted(name for name in directories if name.casefold() not in ignored)
        parent = Path(current)
        for filename in filenames:
            path = parent / filename
            if not should_skip(path, root):
                files.append(path)
    files.sort(key=lambda p: p.relative_to(root).as_posix().casefold())
    return root, files


def read_source_files(files: list[Path]) -> Iterator[tuple[Path, str]]:
    for path in files:
        try:
            yield path, path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue


def language_for(path: Path) -> str:
    return SUPPORTED_EXTENSIONS[path.suffix.lower()]
