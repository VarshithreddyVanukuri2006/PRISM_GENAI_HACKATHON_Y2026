from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.models.schemas import CodeChunk


def _load_networkx():
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "Code graph support requires NetworkX. Install it with: "
            "pip install -r backend/requirements-graph.txt"
        ) from exc
    return nx


def file_node_id(file_path: str) -> str:
    return f"file::{file_path}"


def symbol_node_id(chunk_id: str) -> str:
    return f"symbol::{chunk_id}"


@dataclass(slots=True)
class GraphExpansion:
    seed_chunk_id: str
    chunk_id: str
    hops: int
    relation_path: list[str]
    graph_score: float


class CodeGraph:
    """Directed graph connecting source files and parsed code symbols."""

    FORMAT_VERSION = 1

    def __init__(self, graph, chunks: Iterable[CodeChunk], warnings: list[str] | None = None):
        self.graph = graph
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.warnings = warnings or []

    @classmethod
    def from_chunks(cls, chunks: Iterable[CodeChunk]) -> "CodeGraph":
        nx = _load_networkx()
        chunk_list = list(chunks)
        by_id: dict[str, CodeChunk] = {}
        warnings: list[str] = []
        for chunk in chunk_list:
            if chunk.chunk_id in by_id:
                raise ValueError(f"Duplicate chunk_id in graph input: {chunk.chunk_id}")
            by_id[chunk.chunk_id] = chunk

        graph = nx.DiGraph()
        files: dict[str, list[CodeChunk]] = defaultdict(list)
        symbols_by_name: dict[str, list[CodeChunk]] = defaultdict(list)
        classes_by_name: dict[str, list[CodeChunk]] = defaultdict(list)

        for chunk in chunk_list:
            files[chunk.file_path].append(chunk)
            symbols_by_name[chunk.symbol_name].append(chunk)
            symbols_by_name[chunk.symbol_name.rsplit(".", 1)[-1]].append(chunk)
            if chunk.symbol_type == "class":
                classes_by_name[chunk.symbol_name].append(chunk)
                classes_by_name[chunk.symbol_name.rsplit(".", 1)[-1]].append(chunk)
            graph.add_node(file_node_id(chunk.file_path), kind="file", file_path=chunk.file_path)
            graph.add_node(
                symbol_node_id(chunk.chunk_id), kind=chunk.symbol_type,
                chunk_id=chunk.chunk_id, symbol_name=chunk.symbol_name,
                file_path=chunk.file_path, language=chunk.language,
                start_line=chunk.start_line, end_line=chunk.end_line,
            )

        def add_relation(source: str, target: str, relation: str) -> None:
            if graph.has_edge(source, target):
                relations = graph[source][target].setdefault("relations", [])
                if relation not in relations:
                    relations.append(relation)
            else:
                graph.add_edge(source, target, relations=[relation])

        def resolve_symbol(name: str, file_path: str, class_only: bool = False) -> CodeChunk | None:
            mapping = classes_by_name if class_only else symbols_by_name
            final = name.rsplit(".", 1)[-1]
            local = []
            for candidate in files.get(file_path, []):
                if class_only and candidate.symbol_type != "class":
                    continue
                if candidate.symbol_name == name or candidate.symbol_name.rsplit(".", 1)[-1] == final:
                    local.append(candidate)
            if len(local) == 1:
                return local[0]
            exact = mapping.get(name, [])
            exact = list({candidate.chunk_id: candidate for candidate in exact}.values())
            if len(exact) == 1:
                return exact[0]
            matches = mapping.get(final, [])
            matches = list({candidate.chunk_id: candidate for candidate in matches}.values())
            return matches[0] if len(matches) == 1 else None

        # Attach top-level symbols to their file and nested symbols to their parent symbol.
        for chunk in chunk_list:
            current = symbol_node_id(chunk.chunk_id)
            parent = None
            if chunk.parent_symbol:
                parent_chunk = resolve_symbol(chunk.parent_symbol, chunk.file_path)
                if parent_chunk is not None:
                    parent = symbol_node_id(parent_chunk.chunk_id)
                else:
                    warnings.append(
                        f"{chunk.file_path}:{chunk.start_line}: parent symbol '{chunk.parent_symbol}' was not resolved"
                    )
            add_relation(parent or file_node_id(chunk.file_path), current, "CONTAINS")

        # Link resolvable calls. Ambiguous global names are deliberately left disconnected.
        for chunk in chunk_list:
            source = symbol_node_id(chunk.chunk_id)
            for call in chunk.calls:
                target_chunk = resolve_symbol(call, chunk.file_path)
                if target_chunk is not None:
                    add_relation(source, symbol_node_id(target_chunk.chunk_id), "CALLS")

        # Record other AST name/attribute references that resolve to repository symbols.
        for chunk in chunk_list:
            try:
                tree = ast.parse(chunk.code)
            except SyntaxError:
                continue
            collector = _ReferenceCollector()
            collector.visit(tree)
            source = symbol_node_id(chunk.chunk_id)
            for reference in collector.names:
                target_chunk = resolve_symbol(reference, chunk.file_path)
                if target_chunk is not None and target_chunk.chunk_id != chunk.chunk_id:
                    add_relation(source, symbol_node_id(target_chunk.chunk_id), "REFERENCES")

        # Imports connect files to repository files when a matching module exists.
        file_imports: dict[str, set[str]] = defaultdict(set)
        for chunk in chunk_list:
            file_imports[chunk.file_path].update(chunk.imports)
        known_files = set(files)
        for source_path, imports in file_imports.items():
            for imported in imports:
                target_path = _resolve_module_path(source_path, imported, known_files)
                if target_path:
                    add_relation(file_node_id(source_path), file_node_id(target_path), "IMPORTS")

        # Python AST chunks retain their class source, so inheritance can be resolved locally.
        for chunk in chunk_list:
            if chunk.symbol_type != "class":
                continue
            try:
                tree = ast.parse(chunk.code)
            except SyntaxError:
                continue
            class_node = next((node for node in tree.body if isinstance(node, ast.ClassDef)), None)
            if class_node is None:
                continue
            for base in class_node.bases:
                base_name = _ast_name(base)
                if not base_name:
                    continue
                base_chunk = resolve_symbol(base_name, chunk.file_path, class_only=True)
                if base_chunk is not None:
                    add_relation(symbol_node_id(chunk.chunk_id), symbol_node_id(base_chunk.chunk_id), "INHERITS")

        return cls(graph, chunk_list, warnings)

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def summary(self) -> dict[str, object]:
        node_types = Counter(data.get("kind", "unknown") for _, data in self.graph.nodes(data=True))
        relation_counts: Counter[str] = Counter()
        for _, _, data in self.graph.edges(data=True):
            relation_counts.update(data.get("relations", []))
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "node_types": dict(sorted(node_types.items())),
            "relations": dict(sorted(relation_counts.items())),
            "warnings": list(self.warnings),
        }

    def related_to_chunk(self, chunk_id: str) -> list[dict[str, object]]:
        """Return direct, factual incoming/outgoing relationships for one code chunk."""
        if chunk_id not in self.chunks:
            raise KeyError(f"Unknown chunk_id: {chunk_id}")
        node_id = symbol_node_id(chunk_id)
        results: list[dict[str, object]] = []
        for source, target, data in self.graph.in_edges(node_id, data=True):
            results.append(self._relationship(source, target, data, direction="incoming"))
        for source, target, data in self.graph.out_edges(node_id, data=True):
            results.append(self._relationship(source, target, data, direction="outgoing"))
        results.sort(key=lambda item: (
            str(item["direction"]), str(item["relation"]), str(item["node"]["file_path"]),
            str(item["node"].get("symbol_name", "")),
        ))
        return results

    def expand_from_chunks(self, seed_chunk_ids: Iterable[str], max_hops: int = 3,
                           max_candidates: int = 100) -> list[GraphExpansion]:
        """Traverse resolved graph edges and return reachable code chunks with paths."""
        if max_hops < 1:
            raise ValueError("max_hops must be at least one")
        if max_candidates < 0:
            raise ValueError("max_candidates must not be negative")
        seeds = list(dict.fromkeys(seed_chunk_ids))
        unknown = [chunk_id for chunk_id in seeds if chunk_id not in self.chunks]
        if unknown:
            raise KeyError(f"Unknown seed chunk_id: {unknown[0]}")
        expansions: dict[str, GraphExpansion] = {}
        for seed_id in seeds:
            start = symbol_node_id(seed_id)
            queue = deque([(start, 0, [], {start})])
            while queue and len(expansions) < max_candidates:
                current, hops, path, visited = queue.popleft()
                if hops >= max_hops:
                    continue
                transitions: list[tuple[str, str, dict, str]] = []
                for source, target, data in self.graph.out_edges(current, data=True):
                    for relation in data.get("relations", []):
                        transitions.append((source, target, data, relation))
                for source, target, data in self.graph.in_edges(current, data=True):
                    for relation in data.get("relations", []):
                        transitions.append((source, target, data, relation))
                transitions.sort(key=lambda item: (item[3], item[0], item[1]))
                for source, target, _, relation in transitions:
                    neighbor = target if source == current else source
                    if neighbor in visited:
                        continue
                    next_hops = hops + 1
                    attributes = self.graph.nodes[neighbor]
                    label = attributes.get("symbol_name") or attributes.get("file_path") or neighbor
                    if source == current:
                        path_step = f"{relation} -> {label}"
                    else:
                        inverse = {"CALLS": "CALLED_BY", "CONTAINS": "CONTAINED_BY",
                                   "IMPORTS": "IMPORTED_BY", "INHERITS": "SUBCLASS_OF",
                                   "REFERENCES": "REFERENCED_BY"}.get(relation, relation)
                        path_step = f"{inverse} <- {label}"
                    next_path = [*path, path_step]
                    target_chunk_id = attributes.get("chunk_id")
                    if target_chunk_id and target_chunk_id != seed_id:
                        expansion = GraphExpansion(
                            seed_chunk_id=seed_id, chunk_id=target_chunk_id,
                            hops=next_hops, relation_path=next_path,
                            graph_score=1.0 / next_hops,
                        )
                        current_expansion = expansions.get(target_chunk_id)
                        if current_expansion is None or expansion.hops < current_expansion.hops:
                            expansions[target_chunk_id] = expansion
                    queue.append((neighbor, next_hops, next_path, {*visited, neighbor}))
                    if len(expansions) >= max_candidates:
                        break
        return sorted(expansions.values(), key=lambda item: (
            item.hops, item.chunk_id, item.seed_chunk_id,
        ))

    def _relationship(self, source: str, target: str, data: dict,
                      direction: str) -> dict[str, object]:
        other_id = source if direction == "incoming" else target
        attributes = dict(self.graph.nodes[other_id])
        attributes["node_id"] = other_id
        relations = list(data.get("relations", []))
        return {
            "direction": direction,
            "relation": ",".join(relations),
            "node": attributes,
        }

    def to_dict(self) -> dict[str, object]:
        nodes = [{"node_id": node_id, **dict(data)}
                 for node_id, data in self.graph.nodes(data=True)]
        edges = [{"source": source, "target": target, **dict(data)}
                 for source, target, data in self.graph.edges(data=True)]
        return {
            "format_version": self.FORMAT_VERSION,
            "summary": self.summary(),
            "nodes": nodes,
            "edges": edges,
            "chunks": [chunk.to_dict() for chunk in self.chunks.values()],
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "CodeGraph":
        nx = _load_networkx()
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if payload.get("format_version") != cls.FORMAT_VERSION:
                raise ValueError("unsupported code graph format version")
            chunks = [CodeChunk(**item) for item in payload["chunks"]]
            graph = nx.DiGraph()
            for node in payload["nodes"]:
                attributes = dict(node)
                node_id = attributes.pop("node_id")
                graph.add_node(node_id, **attributes)
            for edge in payload["edges"]:
                attributes = dict(edge)
                source_id = attributes.pop("source")
                target_id = attributes.pop("target")
                graph.add_edge(source_id, target_id, **attributes)
            warnings = payload.get("summary", {}).get("warnings", [])
            return cls(graph, chunks, warnings)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Unable to load code graph {source}: {exc}") from exc


def _ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _ast_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


class _ReferenceCollector(ast.NodeVisitor):
    def __init__(self):
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = _ast_name(node)
        if name:
            self.names.add(name)
        self.generic_visit(node)


def _resolve_module_path(source_path: str, imported: str, known_files: set[str]) -> str | None:
    leading_dots = len(imported) - len(imported.lstrip("."))
    module_name = imported[leading_dots:]
    if leading_dots:
        package_parts = Path(source_path).parent.parts
        trim = max(0, leading_dots - 1)
        base_parts = package_parts[:len(package_parts) - trim] if trim else package_parts
        module_parts = [*base_parts, *module_name.split(".")] if module_name else list(base_parts)
    else:
        module_parts = module_name.split(".") if module_name else []
    relative_module = "/".join(part for part in module_parts if part)
    candidates = [f"{relative_module}.py", f"{relative_module}/__init__.py"] if relative_module else []
    for candidate in candidates:
        if candidate in known_files:
            return candidate
    return None
