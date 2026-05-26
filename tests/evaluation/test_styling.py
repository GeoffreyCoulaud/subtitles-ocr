"""Tests for styling sub-score (ADR-0006 §5.7)."""

from __future__ import annotations

import pytest

from subtitles_ocr.evaluation.styling import styling_pair_score, styling_score


def test_no_overrides_on_either_side_returns_none() -> None:
    assert styling_pair_score("foo", "foo") is None


def test_matching_italic_scores_one() -> None:
    assert styling_pair_score(r"{\i1}foo", r"{\i1}foo") == 1.0


def test_mismatched_italic_scores_zero() -> None:
    assert styling_pair_score(r"{\i1}foo", r"foo") == 0.0


def test_matching_colour_close_score_one() -> None:
    # Same colour both sides -> component score 1, mean 1.
    assert styling_pair_score(r"{\c&H000000FF&}foo", r"{\c&H000000FF&}foo") == 1.0


def test_font_size_ratio() -> None:
    # fs=20 vs fs=40 -> ratio 0.5.
    assert styling_pair_score(r"{\fs20}foo", r"{\fs40}foo") == 0.5


def test_rotation_difference_180_degrees_scores_zero() -> None:
    assert styling_pair_score(r"{\frz0}foo", r"{\frz180}foo") == 0.0


def test_multiple_components_averaged() -> None:
    # italic matches (1.0); font_size differs by ratio 0.5 -> mean 0.75.
    score = styling_pair_score(r"{\i1\fs20}foo", r"{\i1\fs40}foo")
    assert score == pytest.approx(0.75)


def test_styling_score_skips_none_pairs() -> None:
    # First pair has no overrides on either side -> skipped.
    # Second pair has matching italic -> 1.0.
    pairs = [("foo", "foo"), (r"{\i1}foo", r"{\i1}foo")]
    assert styling_score(pairs) == 1.0


def test_styling_score_returns_none_when_all_pairs_skip() -> None:
    assert styling_score([("foo", "foo"), ("bar", "bar")]) is None
