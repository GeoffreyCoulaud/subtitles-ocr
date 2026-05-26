"""Tests for the timing sub-score (ADR-0006 §5.3)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from subtitles_ocr.evaluation.alignment import Cue
from subtitles_ocr.evaluation.timing import endpoint_score, timing_pair_score, timing_score


def test_endpoint_score_is_one_at_zero_delta() -> None:
    assert endpoint_score(0.0, K=10) == 1.0


def test_endpoint_score_is_one_for_sub_frame_offset() -> None:
    assert endpoint_score(0.9, K=10) == 1.0


def test_endpoint_score_below_one_at_exactly_one_frame() -> None:
    score = endpoint_score(1.0, K=10)
    assert score < 1.0
    assert score == pytest.approx(0.9)


def test_endpoint_score_is_zero_at_K_frames() -> None:
    assert endpoint_score(10.0, K=10) == 0.0


def test_endpoint_score_clamps_to_zero_beyond_K() -> None:
    assert endpoint_score(50.0, K=10) == 0.0


def test_timing_pair_score_averages_start_and_end() -> None:
    fps = Fraction(24)
    # 1 second offset = 24 frames -> both endpoints 0 -> pair score 0
    ref = Cue(0.0, 1.0)
    out = Cue(1.0, 2.0)
    assert timing_pair_score(out, ref, fps=fps, K=10) == 0.0


def test_timing_pair_score_one_for_identical_cues() -> None:
    fps = Fraction(24)
    cue = Cue(0.5, 1.5)
    assert timing_pair_score(cue, cue, fps=fps, K=10) == 1.0


def test_timing_score_returns_none_for_empty_pairs() -> None:
    assert timing_score([], fps=Fraction(24)) is None


def test_timing_score_means_over_pairs() -> None:
    fps = Fraction(24)
    perfect = (Cue(0.0, 1.0), Cue(0.0, 1.0))
    half_second_off = (Cue(0.5, 1.5), Cue(0.0, 1.0))  # 12-frame deltas -> ep score clamped to 0
    score = timing_score([perfect, half_second_off], fps=fps)
    assert score == pytest.approx(0.5)
