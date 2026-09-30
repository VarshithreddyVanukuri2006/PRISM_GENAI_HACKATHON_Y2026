from pathlib import Path

from app.config import IGNORED_DIRS, MAX_SOURCE_FILE_BYTES, SUPPORTED_EXTENSIONS


def is_supported_source(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def should_skip(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
        if any(part.lower() in IGNORED_DIRS for part in relative.parts[:-1]):
            return True
        if not is_supported_source(path):
            return True
        return path.stat().st_size > MAX_SOURCE_FILE_BYTES
    except (OSError, ValueError):
        return True
