"""Tests for line_breaks sub-score (ADR-0006 §5.6, v1 count-only)."""

from __future__ import annotations

from subtitles_ocr.evaluation.line_breaks import (
    count_breaks,
    line_breaks_pair_score,
    line_breaks_score,
)


def test_count_breaks_strips_override_blocks() -> None:
    assert count_breaks(r"{\i1}foo\Nbar{\i0}") == 1


def test_count_breaks_counts_capital_and_lower_n() -> None:
    assert count_breaks(r"a\Nb\nc") == 2


def test_count_breaks_zero_for_single_line() -> None:
    assert count_breaks("just words") == 0


def test_pair_score_one_when_counts_match() -> None:
    assert line_breaks_pair_score(r"a\Nb", r"a\Nb") == 1.0


def test_pair_score_zero_when_counts_differ() -> None:
    assert line_breaks_pair_score(r"a\Nb\Nc", r"a\Nb") == 0.0


def test_line_breaks_score_returns_none_for_empty_pairs() -> None:
    assert line_breaks_score([]) is None


def test_line_breaks_score_averages() -> None:
    pairs = [(r"a\Nb", r"a\Nb"), (r"a", r"a\Nb")]
    assert line_breaks_score(pairs) == 0.5
