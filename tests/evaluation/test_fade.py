"""Tests for fade sub-score (ADR-0006 §5.9)."""

from __future__ import annotations

from fractions import Fraction

from subtitles_ocr.evaluation.fade import fade_pair_score, fade_score


def test_both_no_fade_excluded() -> None:
    assert fade_pair_score("foo", "foo", fps=Fraction(24)) is None


def test_one_side_fade_scores_zero() -> None:
    assert fade_pair_score(r"{\fad(200,300)}foo", r"foo", fps=Fraction(24)) == 0.0
    assert fade_pair_score(r"foo", r"{\fad(200,300)}foo", fps=Fraction(24)) == 0.0


def test_matching_fade_scores_one() -> None:
    out = r"{\fad(200,300)}foo"
    ref = r"{\fad(200,300)}foo"
    assert fade_pair_score(out, ref, fps=Fraction(24)) == 1.0


def test_close_fade_scores_high() -> None:
    out = r"{\fad(200,300)}foo"  # 200 ms ≈ 4.8 frames at 24 fps
    ref = r"{\fad(220,320)}foo"  # 220 ms ≈ 5.28 frames → delta ≈ 0.48 frames (sub-frame)
    # Sub-frame delta on both endpoints → endpoint scores 1.0 → pair score 1.0
    assert fade_pair_score(out, ref, fps=Fraction(24)) == 1.0


def test_far_fade_scores_lower() -> None:
    out = r"{\fad(0,0)}foo"
    ref = r"{\fad(500,500)}foo"  # 500 ms = 12 frames at 24 fps → beyond K=10 → 0
    assert fade_pair_score(out, ref, fps=Fraction(24)) == 0.0


def test_fade_score_excludes_none_pairs() -> None:
    pairs = [("foo", "foo"), (r"{\fad(0,0)}foo", r"{\fad(0,0)}foo")]
    assert fade_score(pairs, fps=Fraction(24)) == 1.0


def test_fade_score_returns_none_when_all_excluded() -> None:
    assert fade_score([("foo", "foo")], fps=Fraction(24)) is None
