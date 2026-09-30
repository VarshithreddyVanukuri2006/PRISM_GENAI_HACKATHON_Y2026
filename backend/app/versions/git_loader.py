from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Iterator

from app.config import IGNORED_DIRS, MAX_SOURCE_FILE_BYTES, SUPPORTED_EXTENSIONS
from app.ingestion.repository_loader import RepositoryError


def _git(repository_path: str | Path, *args: str, stdout=None) -> str:
    command = ["git", "-C", str(repository_path), *args]
    try:
        completed = subprocess.run(command, check=True, text=stdout is None,
                                   stdout=subprocess.PIPE if stdout is None else stdout,
                                   stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RepositoryError("Git executable was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        raise RepositoryError(f"Git command failed: {stderr.strip() or 'unknown Git error'}") from exc
    return completed.stdout.strip() if stdout is None else ""


def git_root(repository_path: str | Path) -> Path:
    root = _git(repository_path, "rev-parse", "--show-toplevel")
    return Path(root).resolve()


def resolve_commit(repository_path: str | Path, revision: str) -> tuple[str, list[str]]:
    if not revision or revision.startswith("-"):
        raise RepositoryError("Git revision must be a non-empty ref, tag, or commit ID")
    root = git_root(repository_path)
    try:
        commit = _git(root, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}")
        parents_line = _git(root, "rev-list", "--parents", "-n", "1", commit)
    except RepositoryError as exc:
        raise RepositoryError(f"Could not resolve Git revision '{revision}': {exc}") from exc
    parent_values = parents_line.split()
    return commit, parent_values[1:]


def changed_path_metadata(repository_path: str | Path, commit: str,
                          parents: list[str]) -> tuple[list[str], dict[str, str]]:
    if not parents:
        return [], {}
    output = _git(repository_path, "diff", "--find-renames", "--name-status",
                  parents[0], commit)
    changed: list[str] = []
    renames: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split("\t")
        if not fields:
            continue
        status = fields[0]
        if status.startswith("R") and len(fields) >= 3:
            changed.extend(fields[1:3])
            renames[fields[1]] = fields[2]
        elif len(fields) >= 2:
            changed.append(fields[1])
    return sorted(set(changed)), renames


def _eligible_archive_path(member_path: str) -> bool:
    path = PurePosixPath(member_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    if any(part.casefold() in {name.casefold() for name in IGNORED_DIRS} for part in path.parts[:-1]):
        return False
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    return True


def extract_git_snapshot(repository_path: str | Path, commit: str,
                         destination: str | Path) -> Iterator[Path]:
    """Extract only bounded, regular supported source files from a Git archive."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination.parent / f"{destination.name}.tar"
    try:
        with archive_path.open("wb") as archive_file:
            _git(repository_path, "archive", "--format=tar", commit, stdout=archive_file)
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive.getmembers():
                if not member.isfile() or member.size > MAX_SOURCE_FILE_BYTES:
                    continue
                if not _eligible_archive_path(member.name):
                    continue
                relative = PurePosixPath(member.name)
                target = (destination / Path(*relative.parts)).resolve()
                try:
                    target.relative_to(destination)
                except ValueError:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                try:
                    source = extracted.read().decode("utf-8")
                except UnicodeError:
                    continue
                target.write_text(source, encoding="utf-8")
        yield from sorted(destination.rglob("*"), key=lambda path: path.as_posix().casefold())
    finally:
        archive_path.unlink(missing_ok=True)
