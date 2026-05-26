"""Subtitle output scoring engine (ADR-0006)."""

from __future__ import annotations

from subtitles_ocr.evaluation.report import (
    MatchedPair,
    ScoreReport,
    Weights,
    default_weights,
)
from subtitles_ocr.evaluation.score import score

__all__ = ["score", "ScoreReport", "Weights", "MatchedPair", "default_weights"]
