"""Metadata extraction is performed alongside AST parsing in Phase A."""

from app.models.schemas import CodeChunk


def searchable_text(chunk: CodeChunk) -> str:
    """Build a deterministic representation for future retrieval phases."""
    lines = [
        f"FILE: {chunk.file_path}", f"LANGUAGE: {chunk.language}",
        f"TYPE: {chunk.symbol_type}", f"SYMBOL: {chunk.symbol_name}",
    ]
    if chunk.docstring:
        lines.append(f"DOCSTRING: {chunk.docstring}")
    if chunk.imports:
        lines.append(f"IMPORTS: {', '.join(chunk.imports)}")
    if chunk.calls:
        lines.append(f"CALLS: {', '.join(chunk.calls)}")
    lines.extend(("CODE:", chunk.code))
    return "\n".join(lines)
