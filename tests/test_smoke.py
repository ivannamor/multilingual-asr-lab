"""Smoke test: the package is installed and importable."""

from asr_lab import cli


def test_package_is_importable() -> None:
    assert callable(cli.main)
