"""Tests for run provenance: file hashes and the git commit read from .git."""

from pathlib import Path

from asr_lab.provenance import environment, file_sha256, git_commit, utc_now

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _repo(tmp_path: Path) -> Path:
    git_dir = tmp_path / "repo" / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    return git_dir


def test_file_sha256_known_value(tmp_path: Path) -> None:
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_git_commit_from_branch_ref(tmp_path: Path) -> None:
    git_dir = _repo(tmp_path)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "refs" / "heads" / "main").write_text(COMMIT + "\n", encoding="utf-8")
    subdir = git_dir.parent / "src" / "pkg"
    subdir.mkdir(parents=True)
    assert git_commit(subdir) == COMMIT  # found by walking up from a subdirectory


def test_git_commit_from_packed_refs(tmp_path: Path) -> None:
    git_dir = _repo(tmp_path)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "packed-refs").write_text(
        f"# pack-refs with: peeled\n{COMMIT} refs/heads/main\n", encoding="utf-8"
    )
    assert git_commit(git_dir.parent) == COMMIT


def test_git_commit_with_detached_head(tmp_path: Path) -> None:
    git_dir = _repo(tmp_path)
    (git_dir / "HEAD").write_text(COMMIT + "\n", encoding="utf-8")
    assert git_commit(git_dir.parent) == COMMIT


def test_git_commit_outside_a_repository(tmp_path: Path) -> None:
    assert git_commit(tmp_path) is None


def test_utc_now_and_environment() -> None:
    assert utc_now().endswith("+00:00")
    assert set(environment()) == {"python", "platform", "executable_is_venv"}
