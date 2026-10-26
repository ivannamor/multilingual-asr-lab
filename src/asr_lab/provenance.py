"""Information recorded with every run so that results can be traced and reproduced.

The git commit is read from the ``.git`` directory directly instead of running
``git``: the code never calls shell commands, so it behaves the same on Linux and
Windows. The drawback is that uncommitted changes cannot be detected.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks so large files do not fill memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(start: Path | None = None) -> str | None:
    """Commit hash checked out in the repository containing ``start`` (default: cwd).

    Returns None outside a git repository or if the reference cannot be resolved.
    """
    git_dir = _find_git_dir((start or Path.cwd()).resolve())
    if git_dir is None:
        return None
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head  # detached HEAD: the file contains the hash itself
    ref = head.removeprefix("ref: ")
    ref_file = git_dir / ref
    if ref_file.exists():
        return ref_file.read_text(encoding="utf-8").strip()
    return _packed_ref(git_dir, ref)


def utc_now() -> str:
    """Current time in UTC, ISO 8601, to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def environment() -> dict[str, Any]:
    """Python version and platform, which can affect speed and, rarely, results."""
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "executable_is_venv": sys.prefix != sys.base_prefix,
    }


def _find_git_dir(directory: Path) -> Path | None:
    for candidate in (directory, *directory.parents):
        dot_git = candidate / ".git"
        if dot_git.is_dir():
            return dot_git
        if dot_git.is_file():  # worktrees and submodules: ".git" is a file pointing elsewhere
            target = dot_git.read_text(encoding="utf-8").strip().removeprefix("gitdir: ")
            return (candidate / target).resolve()
    return None


def _packed_ref(git_dir: Path, ref: str) -> str | None:
    """Look up a reference in ``packed-refs`` (git packs old refs into one file)."""
    packed = git_dir / "packed-refs"
    if not packed.exists():
        return None
    for line in packed.read_text(encoding="utf-8").splitlines():
        if line.endswith(f" {ref}"):
            return line.split(" ", 1)[0]
    return None
