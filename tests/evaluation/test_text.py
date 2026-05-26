"""Tests for text_plain and text_exact sub-scores (ADR-0006 §5.1-5.2)."""

from __future__ import annotations

import pytest

from subtitles_ocr.evaluation.text import (
    normalise_exact,
    normalise_plain,
    text_exact_pair_score,
    text_exact_score,
    text_plain_pair_score,
    text_plain_score,
)


def test_normalise_plain_strips_override_tags() -> None:
    assert normalise_plain(r"{\i1}Hello{\i0}") == "hello"


def test_normalise_plain_lowercases_and_strips_punctuation() -> None:
    assert normalise_plain("Hello, World!") == "hello world"


def test_normalise_plain_collapses_smart_quotes_and_ellipsis() -> None:
    assert normalise_plain("It's great…") == "its great"


def test_normalise_plain_handles_line_breaks() -> None:
    assert normalise_plain(r"line one\Nline two") == "line one line two"


def test_normalise_exact_preserves_case_and_punctuation() -> None:
    assert normalise_exact(r"{\i1}Hello, World!{\i0}") == "Hello, World!"


def test_text_plain_pair_score_identical() -> None:
    assert text_plain_pair_score("hello", "hello") == 1.0


def test_text_plain_pair_score_single_substitution() -> None:
    # "hello" vs "jello": 1 edit, max length 5 -> score 0.8
    assert text_plain_pair_score("hello", "jello") == pytest.approx(0.8)


def test_text_plain_pair_score_empty_strings_score_one() -> None:
    assert text_plain_pair_score("", "") == 1.0


def test_text_plain_score_averages_over_pairs() -> None:
    score = text_plain_score([("hello", "hello"), ("world", "wxrld")])
    assert score == pytest.approx((1.0 + 0.8) / 2)


def test_text_plain_score_returns_none_when_no_pairs() -> None:
    assert text_plain_score([]) is None


def test_text_exact_score_is_stricter_than_plain() -> None:
    pairs = [("Hello!", "hello")]
    assert text_exact_score(pairs) < text_plain_score(pairs)
