"""Tests for evaluation Pydantic models (ADR-0006 §7.2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from subtitles_ocr.evaluation.report import (
    MatchedPair,
    ScoreReport,
    Weights,
    default_weights,
)


def test_weights_accepts_non_negative_integers() -> None:
    w = Weights(
        text_plain=25, text_exact=5, timing=20, recall=15, precision=10,
        line_breaks=5, styling=5, position=5, fade=5,
    )
    assert w.text_plain == 25
    assert w.fade == 5


def test_weights_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        Weights(
            text_plain=-1, text_exact=5, timing=20, recall=15, precision=10,
            line_breaks=5, styling=5, position=5, fade=5,
        )


def test_default_weights_match_adr_0006() -> None:
    w = default_weights()
    assert w.text_plain == 25
    assert w.text_exact == 5
    assert w.timing == 20
    assert w.recall == 15
    assert w.precision == 10
    assert w.line_breaks == 5
    assert w.styling == 5
    assert w.position == 5
    assert w.fade == 5


def test_matched_pair_allows_nullable_styling_position_fade() -> None:
    pair = MatchedPair(
        ref_index=0, out_index=0, iou=1.0,
        text_plain=1.0, text_exact=1.0, timing=1.0, line_breaks=1.0,
        styling=None, position=None, fade=None,
    )
    assert pair.styling is None


def test_score_report_serialises_to_json() -> None:
    report = ScoreReport(
        final=0.87,
        sub_scores={"text_plain": 0.9, "fade": None},
        effective_weights={"text_plain": 25, "fade": 0},
        n_ref=10, n_out=10, n_matched=10,
        matched_pairs=[],
        warnings=[],
    )
    payload = report.model_dump_json()
    assert '"final":0.87' in payload
    assert '"fade":null' in payload
