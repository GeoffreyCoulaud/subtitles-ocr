"""Tests for position sub-score (ADR-0006 §5.8)."""

from __future__ import annotations

import pytest

from subtitles_ocr.evaluation.position import (
    anchor_to_centre,
    position_pair_score,
    position_score,
)


def test_anchor_to_centre_an2_default() -> None:
    # \an2 bottom-centre; bbox 100 x 40; anchor at (500, 1040)
    centre = anchor_to_centre(anchor=(500.0, 1040.0), an=2, width=100.0, height=40.0)
    assert centre == (500.0, 1020.0)


def test_anchor_to_centre_an5_is_identity() -> None:
    centre = anchor_to_centre(anchor=(960.0, 540.0), an=5, width=200.0, height=60.0)
    assert centre == (960.0, 540.0)


def test_anchor_to_centre_an8_top_centre() -> None:
    centre = anchor_to_centre(anchor=(960.0, 100.0), an=8, width=200.0, height=60.0)
    assert centre == (960.0, 130.0)


def test_pair_with_no_pos_or_move_is_excluded() -> None:
    result = position_pair_score(
        out_text="foo", ref_text="foo",
        play_res=(1920, 1080), default_font_size=40.0,
    )
    assert result is None


def test_identical_pos_scores_one() -> None:
    out = r"{\pos(960,1040)}foo"
    ref = r"{\pos(960,1040)}foo"
    score = position_pair_score(out, ref, play_res=(1920, 1080), default_font_size=40.0)
    assert score == 1.0


def test_pos_offset_by_10pct_of_diagonal_scores_zero() -> None:
    diag = (1920**2 + 1080**2) ** 0.5
    offset = 0.10 * diag
    out = rf"{{\pos(0,0)}}foo"
    ref = rf"{{\pos({offset},0)}}foo"
    score = position_pair_score(out, ref, play_res=(1920, 1080), default_font_size=40.0)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_different_an_same_screen_centre_scores_high() -> None:
    # Both sides describe text whose centre lands near (960, 540).
    # Output: \an5 + \pos(960, 540) → centre (960, 540).
    # Reference: \an2 + \pos(960, 560), bbox height 40 → centre (960, 540).
    out = r"{\an5\pos(960,540)\fs40}foo"
    ref = r"{\an2\pos(960,560)\fs40}foo"
    score = position_pair_score(out, ref, play_res=(1920, 1080), default_font_size=40.0)
    assert score == pytest.approx(1.0, abs=0.05)


def test_move_compared_at_both_endpoints() -> None:
    out = r"{\move(100,100,500,500)}foo"
    ref = r"{\move(100,100,500,500)}foo"
    score = position_pair_score(out, ref, play_res=(1920, 1080), default_font_size=40.0)
    assert score == 1.0


def test_position_score_excludes_none_pairs() -> None:
    pairs = [("foo", "foo"), (r"{\pos(960,540)}foo", r"{\pos(960,540)}foo")]
    assert position_score(pairs, play_res=(1920, 1080), default_font_size=40.0) == 1.0


def test_position_score_returns_none_when_all_pairs_skip() -> None:
    pairs = [("foo", "foo")]
    assert position_score(pairs, play_res=(1920, 1080), default_font_size=40.0) is None
