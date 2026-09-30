from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.indexing.bm25_index import BM25Index
from app.indexing.embedding_index import SemanticIndex
from app.indexing.index_manager import index_repository, write_jsonl
from app.ingestion.repository_loader import RepositoryError, repository_id
from app.versions.git_loader import (
    changed_path_metadata, extract_git_snapshot, git_root, resolve_commit,
)


@dataclass(slots=True)
class VersionRecord:
    version_id: str
    label: str
    commit_hash: str
    parents: list[str]
    changed_paths: list[str]
    renames: dict[str, str]
    chunks_path: str
    bm25_index_path: str
    semantic_index_path: str | None
    number_of_files: int
    number_of_chunks: int
    model_name: str | None = None


@dataclass(slots=True)
class VersionCatalog:
    repository_id: str
    repository_path: str
    versions: list[VersionRecord] = field(default_factory=list)

    FORMAT_VERSION = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.FORMAT_VERSION,
            "repository_id": self.repository_id,
            "repository_path": self.repository_path,
            "versions": [asdict(version) for version in self.versions],
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "VersionCatalog":
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
            if data.get("format_version") != cls.FORMAT_VERSION:
                raise ValueError("unsupported version catalog format version")
            return cls(
                repository_id=data["repository_id"],
                repository_path=data["repository_path"],
                versions=[VersionRecord(**item) for item in data["versions"]],
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Unable to load version catalog {source}: {exc}") from exc


class VersionManager:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).expanduser().resolve()

    def index_git_version(self, repository_path: str | Path, revision: str,
                          label: str | None = None, include_semantic: bool = False,
                          model_name: str | None = None, batch_size: int = 32) -> tuple[Path, VersionRecord]:
        root = git_root(repository_path)
        commit, parents = resolve_commit(root, revision)
        repo_id = repository_id(root)
        repo_output = self.output_dir / repo_id
        catalog_path = repo_output / "catalog.json"
        if catalog_path.exists():
            catalog = VersionCatalog.load(catalog_path)
            if catalog.repository_id != repo_id:
                raise ValueError("Output catalog belongs to a different repository")
            existing = next((version for version in catalog.versions if version.commit_hash == commit), None)
            if existing is not None:
                return catalog_path, existing
        else:
            catalog = VersionCatalog(repo_id, str(root))

        version_label = label or revision
        if any(version.label == version_label for version in catalog.versions):
            raise ValueError(f"Version label '{version_label}' is already in this catalog")
        changed_paths, renames = changed_path_metadata(root, commit, parents)
        version_dir = repo_output / "versions" / commit
        version_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="codelens-git-snapshot-") as temporary:
            snapshot = Path(temporary) / "source"
            list(extract_git_snapshot(root, commit, snapshot))
            summary = index_repository(
                snapshot, repository_id_override=repo_id,
                version_id=commit, version_label=version_label, commit_hash=commit,
            )

        chunks_file = version_dir / "chunks.jsonl"
        bm25_file = version_dir / "bm25.json"
        write_jsonl(summary, chunks_file)
        BM25Index.build(summary.chunks).save(bm25_file)

        semantic_path: str | None = None
        used_model: str | None = None
        if include_semantic:
            from app.embeddings.sentence_transformer import DEFAULT_EMBEDDING_MODEL
            from app.indexing.embedding_index import SemanticIndex

            used_model = model_name or DEFAULT_EMBEDDING_MODEL
            semantic_file = version_dir / "semantic.faiss"
            SemanticIndex.build(summary.chunks, used_model, batch_size).save(semantic_file)
            semantic_path = semantic_file.relative_to(repo_output).as_posix()

        record = VersionRecord(
            version_id=commit, label=version_label, commit_hash=commit,
            parents=parents, changed_paths=changed_paths, renames=renames,
            chunks_path=chunks_file.relative_to(repo_output).as_posix(),
            bm25_index_path=bm25_file.relative_to(repo_output).as_posix(),
            semantic_index_path=semantic_path,
            number_of_files=summary.number_of_files,
            number_of_chunks=summary.number_of_chunks,
            model_name=used_model,
        )
        catalog.versions.append(record)
        catalog.save(catalog_path)
        return catalog_path, record
