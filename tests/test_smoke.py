"""Smoke test: the package is installed and importable."""

import asr_lab


def test_package_is_importable() -> None:
    assert callable(asr_lab.main)
