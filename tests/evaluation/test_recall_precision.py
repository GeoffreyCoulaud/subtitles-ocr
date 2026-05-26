"""Tests for recall and precision sub-scores (ADR-0006 §5.4-5.5)."""

from __future__ import annotations

import pytest

from subtitles_ocr.evaluation.recall_precision import precision, recall


def test_recall_full_match() -> None:
    assert recall(n_matched=10, n_ref=10) == 1.0


def test_recall_half_match() -> None:
    assert recall(n_matched=5, n_ref=10) == 0.5


def test_recall_raises_on_empty_reference() -> None:
    with pytest.raises(ValueError):
        recall(n_matched=0, n_ref=0)


def test_precision_full_match() -> None:
    assert precision(n_matched=10, n_out=10) == 1.0


def test_precision_half_match() -> None:
    assert precision(n_matched=5, n_out=10) == 0.5


def test_precision_empty_output_is_one() -> None:
    assert precision(n_matched=0, n_out=0) == 1.0
