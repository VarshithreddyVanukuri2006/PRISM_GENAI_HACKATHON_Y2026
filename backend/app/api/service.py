from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.graph.code_graph import CodeGraph
from app.indexing.bm25_index import BM25Index
from app.indexing.embedding_index import SemanticIndex
from app.indexing.index_manager import index_repository, write_jsonl
from app.models.api_schemas import (
    EvolutionRequest, EvolutionResponse, EvolutionTrackResponse,
    EvolutionItemResponse, EvolutionTransitionResponse, EvaluationCaseResult,
    EvaluationRequest, EvaluationResponse, RepositoryIndexRequest, RepositoryIndexResponse,
    RepositoryInfo, SearchRequest, SearchResponse, SearchResult,
    VersionInfo, VersionIndexRequest, VersionIndexResponse, VersionListResponse,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.lexical_retriever import LexicalRetriever
from app.retrieval.multi_pass_retriever import MultiPassRetriever
from app.retrieval.reranker import CodeAwareReranker
from app.retrieval.semantic_retriever import SemanticRetriever
from app.versions.evolution import EvolutionaryRetriever
from app.versions.version_manager import VersionCatalog, VersionManager, VersionRecord
from app.versions.version_search import VersionSearchEngine

logger = logging.getLogger("codelens.api")


@dataclass(slots=True)
class RepositoryRecord:
    repository_id: str
    name: str
    repository_path: str | None
    artifact_key: str
    chunks_path: Path | None
    bm25_path: Path | None
    semantic_path: Path | None
    graph_path: Path | None
    number_of_files: int
    number_of_chunks: int
    has_versions: bool = False


def _version_info(repository_id: str, repository_path: str, record: VersionRecord) -> VersionInfo:
    return VersionInfo(
        repository_id=repository_id, repository_path=repository_path,
        version_id=record.version_id, label=record.label, commit_hash=record.commit_hash,
        parents=record.parents, changed_paths=record.changed_paths,
        number_of_files=record.number_of_files, number_of_chunks=record.number_of_chunks,
        semantic_indexed=bool(record.semantic_index_path), model_name=record.model_name,
    )


class CodeLensService:
    """Filesystem-backed integration layer around the project's existing retrieval modules."""

    def __init__(self, index_dir: str | Path, version_dir: str | Path):
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.version_dir = Path(version_dir).expanduser().resolve()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._lexical_cache: dict[str, tuple[int, LexicalRetriever]] = {}
        self._semantic_cache: dict[str, tuple[tuple[int, int], SemanticRetriever]] = {}
        self._hybrid_cache: dict[str, tuple[tuple[int, int, int], HybridRetriever]] = {}
        self._graph_cache: dict[str, tuple[int, CodeGraph]] = {}
        self._snapshot_signature = None
        self._cached_records: list[RepositoryRecord] = []
        self._cached_infos: list[RepositoryInfo] = []

    def _signature(self) -> tuple[tuple[str, int, int], ...]:
        paths = [*self.index_dir.glob("*.bm25.json"), *self.index_dir.glob("*.jsonl"),
                 *self.index_dir.glob("*.semantic.faiss"), *self.index_dir.glob("*.graph.json")]
        if self.manifest_path.exists():
            paths.append(self.manifest_path)
        paths.extend(path for path in self.version_dir.rglob("catalog.json") if self.version_dir.exists())
        signature = []
        for path in paths:
            try:
                stat = path.stat()
                signature.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
            except OSError:
                continue
        return tuple(sorted(signature))

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "repositories.json"

    def _manifest(self) -> dict[str, dict[str, str]]:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            repositories = payload.get("repositories", {})
            if not isinstance(repositories, dict):
                raise ValueError("repositories manifest must contain an object")
            return repositories
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError, AttributeError) as exc:
            raise ValueError(f"Unable to read repository manifest {self.manifest_path}: {exc}") from exc

    def _save_manifest(self, repository_id: str, name: str, path: str) -> None:
        data = self._manifest()
        data[repository_id] = {"name": name, "path": path}
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"repositories": data}, indent=2), encoding="utf-8")
        temporary.replace(self.manifest_path)

    def _catalogs(self) -> list[tuple[Path, VersionCatalog]]:
        catalogs: list[tuple[Path, VersionCatalog]] = []
        if not self.version_dir.exists():
            return catalogs
        for path in sorted(self.version_dir.rglob("catalog.json")):
            try:
                catalogs.append((path.resolve(), VersionCatalog.load(path)))
            except ValueError as exc:
                logger.warning("Skipping unreadable version catalog %s: %s", path, exc)
        return catalogs

    def list_repositories(self) -> list[RepositoryInfo]:
        signature = self._signature()
        if signature == self._snapshot_signature:
            return list(self._cached_infos)
        manifest = self._manifest()
        records: dict[str, RepositoryRecord] = {}
        bm25_paths = sorted(self.index_dir.glob("*.bm25.json"))
        for bm25_path in bm25_paths:
            artifact_key = bm25_path.name[:-len(".bm25.json")]
            chunks_path = self.index_dir / f"{artifact_key}.jsonl"
            metadata = manifest.get(artifact_key, {})
            try:
                index = BM25Index.load(bm25_path)
                chunks = index.chunks
            except (ValueError, OSError) as exc:
                logger.exception("Could not inspect BM25 index %s", bm25_path)
                raise ValueError(f"BM25 index is unreadable ({bm25_path.name}): {exc}") from exc
            repository_id = chunks[0].repository_id if chunks else artifact_key
            if repository_id != artifact_key and repository_id in manifest:
                metadata = manifest[repository_id]
            semantic_path = self.index_dir / f"{artifact_key}.semantic.faiss"
            graph_path = self.index_dir / f"{artifact_key}.graph.json"
            records[repository_id] = RepositoryRecord(
                repository_id=repository_id,
                name=metadata.get("name") or (Path(metadata["path"]).name if metadata.get("path") else artifact_key),
                repository_path=metadata.get("path"), artifact_key=artifact_key,
                chunks_path=chunks_path if chunks_path.is_file() else None,
                bm25_path=bm25_path,
                semantic_path=semantic_path if semantic_path.is_file() and Path(str(semantic_path) + ".json").is_file() else None,
                graph_path=graph_path if graph_path.is_file() else None,
                number_of_files=len({chunk.file_path for chunk in chunks}),
                number_of_chunks=len(chunks),
            )

        # Accept Phase A JSONL indexes even when they have not yet been saved as BM25.
        for chunks_path in sorted(self.index_dir.glob("*.jsonl")):
            if chunks_path.stem in {record.artifact_key for record in records.values()}:
                continue
            try:
                chunks = LexicalRetriever.from_jsonl(chunks_path).index.chunks
            except ValueError as exc:
                logger.warning("Skipping unreadable chunk file %s: %s", chunks_path, exc)
                continue
            artifact_key = chunks_path.stem
            repository_id = chunks[0].repository_id if chunks else artifact_key
            metadata = manifest.get(repository_id, {})
            records[repository_id] = RepositoryRecord(
                repository_id, metadata.get("name", artifact_key), metadata.get("path"),
                artifact_key, chunks_path, None, None, None,
                len({chunk.file_path for chunk in chunks}), len(chunks),
            )

        for catalog_path, catalog in self._catalogs():
            record = records.get(catalog.repository_id)
            if record is None:
                root_name = Path(catalog.repository_path).name or catalog.repository_id
                record = RepositoryRecord(
                    catalog.repository_id, root_name, catalog.repository_path,
                    catalog.repository_id, None, None, None, None, 0, 0,
                )
                records[catalog.repository_id] = record
            record.has_versions = bool(catalog.versions)

        infos: list[RepositoryInfo] = []
        for record in sorted(records.values(), key=lambda item: item.name.casefold()):
            methods = []
            if record.bm25_path or record.chunks_path:
                methods.append("bm25")
            if record.semantic_path:
                methods.extend(["semantic", "hybrid", "reranked"])
                if record.graph_path:
                    methods.append("multi_pass")
            if record.has_versions:
                methods.append("version_search")
            infos.append(RepositoryInfo(
                repository_id=record.repository_id, name=record.name,
                repository_path=record.repository_path, number_of_files=record.number_of_files,
                number_of_chunks=record.number_of_chunks, available_methods=methods,
                has_versions=record.has_versions,
            ))
        self._snapshot_signature = signature
        self._cached_records = list(records.values())
        self._cached_infos = infos
        return infos

    def _records(self) -> list[RepositoryRecord]:
        self.list_repositories()
        return list(self._cached_records)

    def _resolve_repository(self, selector: str | None) -> RepositoryRecord:
        records = self._records()
        if selector is None:
            if len(records) == 1:
                return records[0]
            if not records:
                raise FileNotFoundError("No indexed repositories were found. Index a repository first.")
            raise ValueError("repository is required when more than one repository is indexed")
        normalized = selector.casefold()
        matches = [record for record in records if selector == record.repository_id
                   or normalized == record.name.casefold()
                   or (record.repository_path and str(Path(record.repository_path).resolve()).casefold() == str(Path(selector).expanduser().resolve()).casefold())]
        if not matches:
            # A version-only catalog also acts as a repository descriptor.
            for _, catalog in self._catalogs():
                if selector in {catalog.repository_id, Path(catalog.repository_path).name}:
                    return RepositoryRecord(catalog.repository_id, Path(catalog.repository_path).name,
                                            catalog.repository_path, catalog.repository_id,
                                            None, None, None, None, 0, 0, bool(catalog.versions))
            raise FileNotFoundError(f"No indexed repository matches '{selector}'")
        if len(matches) > 1:
            raise ValueError(f"Repository selector '{selector}' is ambiguous; use its repository ID")
        return matches[0]

    def _info(self, record: RepositoryRecord) -> RepositoryInfo:
        info = next((item for item in self.list_repositories() if item.repository_id == record.repository_id), None)
        if info:
            return info
        return RepositoryInfo(
            repository_id=record.repository_id, name=record.name,
            repository_path=record.repository_path, number_of_files=record.number_of_files,
            number_of_chunks=record.number_of_chunks, available_methods=["version_search"],
            has_versions=record.has_versions,
        )

    def index_repository(self, request: RepositoryIndexRequest) -> RepositoryIndexResponse:
        path = Path(request.repository_path).expanduser()
        summary = index_repository(path)
        artifact_key = summary.repository_id
        chunks_path = self.index_dir / f"{artifact_key}.jsonl"
        bm25_path = self.index_dir / f"{artifact_key}.bm25.json"
        write_jsonl(summary, chunks_path)
        BM25Index.build(summary.chunks).save(bm25_path)
        graph_path: Path | None = None
        semantic_path: Path | None = None
        warnings = list(summary.warnings)
        old_graph = self.index_dir / f"{artifact_key}.graph.json"
        old_semantic = self.index_dir / f"{artifact_key}.semantic.faiss"
        if old_graph.exists():
            old_graph.unlink()
        if old_semantic.exists():
            old_semantic.unlink()
        Path(str(old_semantic) + ".json").unlink(missing_ok=True)
        if request.build_graph:
            try:
                graph = CodeGraph.from_chunks(summary.chunks)
                graph_path = self.index_dir / f"{artifact_key}.graph.json"
                graph.save(graph_path)
            except RuntimeError as exc:
                warnings.append(str(exc))
        if request.include_semantic:
            try:
                semantic_path = self.index_dir / f"{artifact_key}.semantic.faiss"
                SemanticIndex.build(summary.chunks, model_name=request.model_name or None).save(semantic_path)
            except (RuntimeError, ValueError) as exc:
                semantic_path = None
                warnings.append(str(exc))
        repo_name = request.name or Path(summary.repository_path).name
        self._save_manifest(summary.repository_id, repo_name, summary.repository_path)
        self._invalidate(artifact_key)
        return RepositoryIndexResponse(
            repository_id=summary.repository_id, name=repo_name,
            repository_path=summary.repository_path, chunks_path=str(chunks_path),
            bm25_index_path=str(bm25_path),
            semantic_index_path=str(semantic_path) if semantic_path else None,
            graph_path=str(graph_path) if graph_path else None,
            number_of_files=summary.number_of_files, number_of_chunks=summary.number_of_chunks,
            skipped_files=summary.skipped_files, warnings=warnings,
        )

    def _invalidate(self, artifact_key: str) -> None:
        for cache in (self._lexical_cache, self._semantic_cache, self._hybrid_cache, self._graph_cache):
            cache.pop(str((self.index_dir / f"{artifact_key}.bm25.json").resolve()), None)
        self._lexical_cache.clear()
        self._semantic_cache.clear()
        self._hybrid_cache.clear()
        self._graph_cache.clear()

    def _lexical(self, record: RepositoryRecord) -> LexicalRetriever:
        path = record.bm25_path or record.chunks_path
        if path is None:
            raise FileNotFoundError(f"No lexical index is available for '{record.name}'")
        stat = path.stat()
        key = str(path.resolve())
        cached = self._lexical_cache.get(key)
        if cached and cached[0] == stat.st_mtime_ns:
            return cached[1]
        retriever = (LexicalRetriever.from_saved_index(path) if path.name.endswith(".bm25.json")
                     else LexicalRetriever.from_jsonl(path))
        self._lexical_cache[key] = (stat.st_mtime_ns, retriever)
        return retriever

    def _semantic(self, record: RepositoryRecord) -> SemanticRetriever:
        path = record.semantic_path
        if path is None:
            raise FileNotFoundError(f"No semantic index is available for '{record.name}'")
        metadata = Path(str(path) + ".json")
        stat = (path.stat().st_mtime_ns, metadata.stat().st_mtime_ns)
        key = str(path.resolve())
        cached = self._semantic_cache.get(key)
        if cached and cached[0] == stat:
            return cached[1]
        retriever = SemanticRetriever.from_saved_index(path)
        self._semantic_cache[key] = (stat, retriever)
        return retriever

    def _hybrid(self, record: RepositoryRecord) -> HybridRetriever:
        if record.bm25_path is None or record.semantic_path is None:
            raise FileNotFoundError(f"Hybrid retrieval requires BM25 and semantic indexes for '{record.name}'")
        metadata = Path(str(record.semantic_path) + ".json")
        stats = (record.bm25_path.stat().st_mtime_ns, record.semantic_path.stat().st_mtime_ns,
                 metadata.stat().st_mtime_ns)
        key = str(record.bm25_path.resolve())
        cached = self._hybrid_cache.get(key)
        if cached and cached[0] == stats:
            return cached[1]
        retriever = HybridRetriever(self._lexical(record), self._semantic(record))
        self._hybrid_cache[key] = (stats, retriever)
        return retriever

    def _graph(self, record: RepositoryRecord) -> CodeGraph:
        path = record.graph_path
        if path is None:
            raise FileNotFoundError(f"No code graph is available for '{record.name}'")
        stat = path.stat().st_mtime_ns
        key = str(path.resolve())
        cached = self._graph_cache.get(key)
        if cached and cached[0] == stat:
            return cached[1]
        graph = CodeGraph.load(path)
        self._graph_cache[key] = (stat, graph)
        return graph

    def _catalog_for(self, selector: str | None) -> tuple[Path, VersionCatalog]:
        catalogs = self._catalogs()
        if selector is not None:
            record = self._resolve_repository(selector)
            matches = [(path, catalog) for path, catalog in catalogs if catalog.repository_id == record.repository_id]
        else:
            matches = catalogs
        if not matches:
            raise FileNotFoundError("No Git version catalog is indexed for this repository")
        if len(matches) > 1:
            raise ValueError("repository is required when multiple version catalogs exist")
        return matches[0]

    def list_versions(self, repository: str | None = None) -> VersionListResponse:
        catalogs = self._catalogs()
        if repository:
            record = self._resolve_repository(repository)
            catalogs = [(path, catalog) for path, catalog in catalogs if catalog.repository_id == record.repository_id]
        elif len(catalogs) > 1:
            catalogs = sorted(catalogs, key=lambda item: item[1].repository_id)
        versions = [
            _version_info(catalog.repository_id, catalog.repository_path, version)
            for _, catalog in catalogs for version in catalog.versions
        ]
        repo_id = catalogs[0][1].repository_id if len(catalogs) == 1 else None
        return VersionListResponse(repository_id=repo_id, versions=versions)

    def index_version(self, request: VersionIndexRequest) -> VersionIndexResponse:
        catalog_path, version = VersionManager(self.version_dir).index_git_version(
            request.repository_path, request.revision, label=request.label,
            include_semantic=request.include_semantic, model_name=request.model_name,
        )
        from app.ingestion.repository_loader import repository_id
        root = Path(request.repository_path).expanduser().resolve()
        repo_id = repository_id(root)
        self._save_manifest(repo_id, root.name, str(root))
        return VersionIndexResponse(
            catalog_path=str(catalog_path), version=_version_info(repo_id, str(root), version),
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        if request.version and request.version.casefold() != "working-tree" and request.repository is None:
            _, only_catalog = self._catalog_for(None)
            record = self._resolve_repository(only_catalog.repository_id)
        else:
            record = self._resolve_repository(request.repository)
        if request.version and request.version.casefold() != "working-tree":
            if request.retrieval_method not in {"auto", "bm25", "hybrid"}:
                raise ValueError("Version-aware search supports retrieval_method 'auto', 'bm25', or 'hybrid'")
            catalog_path, catalog = self._catalog_for(record.repository_id)
            engine = VersionSearchEngine(catalog_path)
            selected_version = None if request.version.casefold() == "all" else request.version
            mode = "hybrid" if request.retrieval_method == "hybrid" else "lexical"
            if request.retrieval_method == "auto":
                mode = "auto"
            report = engine.search(request.query, version=selected_version, top_k=request.top_k,
                                   candidate_k=request.candidate_k, mode=mode)
            results = [self._search_result(
                result.chunk, result.score, result.rank, result.retrieval_mode,
                lexical_score=result.lexical_score, semantic_score=result.semantic_score,
                signals=[f"Retrieved from Git version {result.version_label}"],
            ) for result in report.results]
            method = f"version_{report.retrieval_mode}"
            version_label = request.version
            response_repository = self._info(record)
            return SearchResponse(
                query=request.query, repository=response_repository, version=version_label,
                retrieval_method=method, retrieval_latency_ms=(time.perf_counter() - started) * 1000,
                total_candidates=len(results), retrieval_details={}, results=results,
            )

        requested = request.retrieval_method
        if requested == "auto":
            if record.semantic_path and record.graph_path:
                method = "multi_pass"
            elif record.semantic_path:
                method = "reranked"
            else:
                method = "bm25"
        else:
            method = requested
        lexical = None
        semantic = None
        hybrid = None
        reranked_results = None
        multipass = None
        fused = None
        if method == "bm25":
            lexical = self._lexical(record).search(request.query, request.top_k)
        elif method == "semantic":
            semantic = self._semantic(record).search(request.query, request.top_k)
        elif method in {"hybrid", "reranked", "multi_pass"}:
            hybrid = self._hybrid(record)
            if method == "hybrid":
                fused = hybrid.search(request.query, request.top_k, request.candidate_k)
            elif method == "reranked":
                candidates = hybrid.search(request.query, request.candidate_k, request.candidate_k)
                reranked_results = CodeAwareReranker().rerank(request.query, candidates, request.top_k)
            else:
                multipass = MultiPassRetriever(hybrid, self._graph(record)).search(
                    request.query, top_k=request.top_k, candidate_k=request.candidate_k,
                    seed_k=request.seed_k, context_candidate_k=request.context_candidate_k,
                    max_graph_hops=request.max_graph_hops,
                )
                reranked_results = multipass.results
        else:
            raise ValueError(f"Unsupported retrieval method '{method}'")

        results: list[SearchResult] = []
        if lexical is not None:
            results = [self._search_result(
                item.chunk, item.score, item.rank, "bm25", lexical_score=item.score,
                signals=["BM25 lexical score from indexed code and metadata"],
            ) for item in lexical]
        elif semantic is not None:
            results = [self._search_result(
                item.chunk, item.score, item.rank, "semantic", semantic_score=item.score,
                signals=["Cosine similarity from the persisted semantic index"],
            ) for item in semantic]
        elif method == "hybrid" and hybrid is not None:
            results = [self._search_result(
                item.chunk, item.score, rank, "hybrid", lexical_score=item.lexical_score,
                semantic_score=item.semantic_score,
                features={"normalized_lexical": item.normalized_lexical_score,
                          "normalized_semantic": item.normalized_semantic_score},
                signals=[signal for signal, present in (
                    ("BM25 candidate", item.lexical_score is not None),
                    ("Semantic candidate", item.semantic_score is not None),
                ) if present],
            ) for rank, item in enumerate(fused, start=1)]
        elif reranked_results is not None:
            results = [self._search_result(
                item.chunk, item.final_score, item.rank, method,
                lexical_score=item.lexical_score, semantic_score=item.semantic_score,
                features=item.features, signals=item.signals,
            ) for item in reranked_results]

        elapsed = (time.perf_counter() - started) * 1000
        retrieval_details: dict[str, object] = {}
        total_candidates = len(results)
        if multipass is not None:
            total_candidates = multipass.pass5_candidate_count
            retrieval_details = {
                "pass1_candidates": multipass.pass1_candidate_count,
                "graph_expansions": len(multipass.pass3_expansions),
                "pass4_candidates": multipass.pass4_candidate_count,
                "pass4_added_chunk_ids": multipass.pass4_added_chunk_ids,
            }
        return SearchResponse(
            query=request.query, repository=self._info(record), version=None,
            retrieval_method=method, retrieval_latency_ms=elapsed,
            total_candidates=total_candidates, retrieval_details=retrieval_details, results=results,
        )

    def graph_for_chunk(self, repository: str, chunk_id: str) -> dict[str, object]:
        """Expose AST-resolved graph nodes and edges for a selected indexed chunk."""
        record = self._resolve_repository(repository)
        try:
            graph = self._graph(record)
        except FileNotFoundError:
            # Older lexical indexes can still provide real code relationships without a
            # re-index: construct the AST graph from the snippets already persisted there.
            chunks_path = record.bm25_path or record.chunks_path
            if chunks_path is None:
                raise
            key = f"{chunks_path.resolve()}#derived-graph"
            modified = chunks_path.stat().st_mtime_ns
            cached_graph = self._graph_cache.get(key)
            if cached_graph and cached_graph[0] == modified:
                graph = cached_graph[1]
            else:
                graph = CodeGraph.from_chunks(self._lexical(record).index.chunks)
                self._graph_cache[key] = (modified, graph)
        relationships = graph.related_to_chunk(chunk_id)
        selected = graph.chunks[chunk_id]

        def chunk_payload(chunk) -> dict[str, object]:
            return {
                "chunk_id": chunk.chunk_id, "repository_id": chunk.repository_id,
                "file_path": chunk.file_path, "language": chunk.language,
                "symbol_name": chunk.symbol_name, "symbol_type": chunk.symbol_type,
                "start_line": chunk.start_line, "end_line": chunk.end_line,
                "code": chunk.code,
            }

        selected_node_id = f"symbol::{chunk_id}"
        nodes: dict[str, dict[str, object]] = {
            selected_node_id: {"node_id": selected_node_id, "kind": selected.symbol_type,
                              **chunk_payload(selected), "selected": True}
        }
        edges: list[dict[str, object]] = []
        for relationship in relationships:
            other = relationship["node"]
            other_id = str(other["node_id"])
            other_chunk_id = other.get("chunk_id")
            other_chunk = graph.chunks.get(str(other_chunk_id)) if other_chunk_id else None
            nodes.setdefault(other_id, {
                **dict(other), **(chunk_payload(other_chunk) if other_chunk else {}), "selected": False,
            })
            if relationship["direction"] == "incoming":
                source, target = other_id, selected_node_id
            else:
                source, target = selected_node_id, other_id
            edges.append({"source": source, "target": target,
                          "relation": relationship["relation"],
                          "direction": relationship["direction"]})
        return {"repository_id": record.repository_id, "selected_chunk_id": chunk_id,
                "nodes": list(nodes.values()), "edges": edges,
                "summary": graph.summary()}

    @staticmethod
    def _search_result(chunk, score: float, rank: int, method: str,
                       lexical_score: float | None = None,
                       semantic_score: float | None = None,
                       features: dict[str, float] | None = None,
                       signals: list[str] | None = None) -> SearchResult:
        return SearchResult(
            rank=rank, chunk_id=chunk.chunk_id, repository_id=chunk.repository_id,
            version_id=chunk.version_id, version_label=chunk.version_label,
            file_path=chunk.file_path, language=chunk.language,
            symbol_name=chunk.symbol_name, symbol_type=chunk.symbol_type,
            start_line=chunk.start_line, end_line=chunk.end_line, code=chunk.code,
            score=score, retrieval_method=method,
            metadata={
                "docstring": chunk.docstring, "imports": chunk.imports,
                "calls": chunk.calls, "class_name": chunk.class_name,
                "parent_symbol": chunk.parent_symbol, "commit_hash": chunk.commit_hash,
            },
            features=features or {}, signals=signals or [],
            lexical_score=lexical_score, semantic_score=semantic_score,
        )

    def evolve(self, request: EvolutionRequest) -> EvolutionResponse:
        started = time.perf_counter()
        catalog_path, catalog = self._catalog_for(request.repository)
        report = EvolutionaryRetriever(catalog_path).search(
            request.query, per_version_top_k=request.per_version_top_k,
            candidate_pool_size=request.candidate_pool_size,
            min_similarity=request.min_similarity, retrieval_mode=request.retrieval_mode,
            max_tracks=request.max_tracks,
        )
        tracks = [EvolutionTrackResponse(
            items=[EvolutionItemResponse(
                version_id=item.version_id, version_label=item.version_label, commit_hash=item.commit_hash,
                chunk_id=item.chunk_id, symbol_name=item.symbol_name, file_path=item.file_path,
                start_line=item.start_line, end_line=item.end_line, relevance_score=item.relevance_score,
                change=item.change, code=item.code,
            ) for item in track.items],
            transitions=[EvolutionTransitionResponse(
                from_version=transition.from_version, to_version=transition.to_version,
                from_chunk_id=transition.from_chunk_id, to_chunk_id=transition.to_chunk_id,
                from_symbol=transition.from_symbol, to_symbol=transition.to_symbol,
                from_path=transition.from_path, to_path=transition.to_path,
                similarity_score=transition.similarity_score, change=transition.change,
            ) for transition in track.transitions],
        ) for track in report.tracks]
        return EvolutionResponse(
            query=report.query, repository_id=catalog.repository_id,
            retrieval_mode=report.retrieval_mode,
            retrieval_latency_ms=(time.perf_counter() - started) * 1000,
            versions_searched=report.versions_searched, tracks=tracks,
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        case_results: list[EvaluationCaseResult] = []
        methods_used: set[str] = set()
        for case in request.cases:
            response = self.search(SearchRequest(
                query=case.query, repository=request.repository, top_k=request.top_k,
                retrieval_method=request.retrieval_method,
            ))
            methods_used.add(response.retrieval_method)
            retrieved = [item.chunk_id for item in response.results]
            grades = [case.relevance.get(chunk_id, 0) for chunk_id in retrieved]
            dcg = sum((2 ** grade - 1) / math.log2(rank + 1)
                      for rank, grade in enumerate(grades[:10], start=1))
            ideal = sorted(case.relevance.values(), reverse=True)[:10]
            idcg = sum((2 ** grade - 1) / math.log2(rank + 1)
                       for rank, grade in enumerate(ideal, start=1))
            ndcg = dcg / idcg if idcg else 0.0
            first_relevant = next((rank for rank, grade in enumerate(grades, start=1) if grade > 0), None)
            reciprocal_rank = 1.0 / first_relevant if first_relevant else 0.0
            case_results.append(EvaluationCaseResult(
                query=case.query, ndcg_at_10=ndcg, reciprocal_rank=reciprocal_rank,
                retrieved_chunk_ids=retrieved, retrieval_latency_ms=response.retrieval_latency_ms,
            ))
        mean_ndcg = sum(item.ndcg_at_10 for item in case_results) / len(case_results)
        mean_mrr = sum(item.reciprocal_rank for item in case_results) / len(case_results)
        mean_latency = sum(item.retrieval_latency_ms for item in case_results) / len(case_results)
        return EvaluationResponse(
            repository=request.repository,
            retrieval_method=next(iter(methods_used)) if len(methods_used) == 1 else "mixed",
            ndcg_at_10=mean_ndcg, mrr=mean_mrr, mean_latency_ms=mean_latency,
            cases=case_results,
        )
