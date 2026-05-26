"""Smoke test for evaluation module scaffolding."""

from __future__ import annotations


def test_evaluation_package_importable() -> None:
    import subtitles_ocr.evaluation  # noqa: F401


def test_pysubs2_importable() -> None:
    import pysubs2  # noqa: F401
