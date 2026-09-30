from pathlib import Path

from app.ingestion.repository_loader import language_for
from app.models.schemas import CodeChunk
from app.parsing.python_parser import parse_python


def parse_file(source: str, path: Path, relative_path: str,
               repository_id: str, warnings: list[str] | None = None) -> list[CodeChunk]:
    """Parse supported files. Python uses its AST; other languages await adapters."""
    if language_for(path) == "Python":
        return parse_python(source, relative_path, repository_id, warnings)
    if warnings is not None:
        warnings.append(f"{relative_path}: {language_for(path)} is recognized for ingestion, but has no symbol parser in Phase A")
    return []
