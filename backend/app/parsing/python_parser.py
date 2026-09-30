from __future__ import annotations

import ast
import hashlib

from app.models.schemas import CodeChunk


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted_name(node.func)
        if name:
            self.calls.add(name)
        self.generic_visit(node)


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _imports(tree: ast.Module) -> list[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            found.add(base)
    return sorted(found)


def _iter_symbols(body: list[ast.stmt], class_name: str | None = None,
                  parent: str | None = None):
    """Yield classes, functions and methods with lexical parent metadata."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "method" if class_name else "function"
            qualified = f"{class_name}.{node.name}" if class_name else node.name
            yield node, qualified, kind, class_name, parent
            yield from _iter_symbols(node.body, class_name, qualified)
        elif isinstance(node, ast.ClassDef):
            qualified = f"{parent}.{node.name}" if parent else node.name
            yield node, qualified, "class", class_name, parent
            yield from _iter_symbols(node.body, node.name, qualified)


def parse_python(source: str, relative_path: str, repository_id: str,
                 warnings: list[str] | None = None) -> list[CodeChunk]:
    """Parse Python into class/function/method chunks with deterministic IDs."""
    try:
        tree = ast.parse(source, filename=relative_path)
    except SyntaxError as exc:
        if warnings is not None:
            location = f"line {exc.lineno}" if exc.lineno else "unknown line"
            warnings.append(f"{relative_path}: Python syntax error at {location}: {exc.msg}")
        return []
    except (ValueError, TypeError) as exc:
        if warnings is not None:
            warnings.append(f"{relative_path}: unable to parse Python source: {exc}")
        return []

    lines = source.splitlines()
    imports = _imports(tree)
    chunks: list[CodeChunk] = []
    for node, symbol_name, symbol_type, class_name, parent in _iter_symbols(tree.body):
        start = getattr(node, "lineno", 1)
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            decorator_lines = [d.lineno for d in decorators if hasattr(d, "lineno")]
            if decorator_lines:
                start = min(start, *decorator_lines)
        end = getattr(node, "end_lineno", start)
        code = "\n".join(lines[start - 1:end])
        docstring = ast.get_docstring(node) or ""
        calls = _CallCollector()
        calls.visit(node)
        stable_key = f"{repository_id}:{relative_path}:{symbol_name}:{symbol_type}:{start}:{end}"
        chunk_id = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:20]
        chunks.append(CodeChunk(
            chunk_id=chunk_id, repository_id=repository_id, file_path=relative_path,
            language="Python", symbol_name=symbol_name, symbol_type=symbol_type,
            start_line=start, end_line=end, code=code, docstring=docstring,
            imports=imports, calls=sorted(calls.calls), class_name=class_name,
            parent_symbol=parent,
        ))
    return chunks
