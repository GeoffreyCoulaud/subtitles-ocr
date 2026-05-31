"""Pydantic v2 data models for the scoring report (ADR-0006 §7.2, ADR-0008)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Weights(BaseModel):
    text_plain: int = Field(ge=0)
    text_exact: int = Field(ge=0)
    timing: int = Field(ge=0)
    recall: int = Field(ge=0)
    precision: int = Field(ge=0)
    line_breaks: int = Field(ge=0)
    styling: int = Field(ge=0)
    # ADR-0008: position pillar split into three independent axes.
    position: int = Field(ge=0)
    anchor: int = Field(ge=0)
    intent: int = Field(ge=0)
    fade: int = Field(ge=0)


def default_weights() -> Weights:
    return Weights(
        text_plain=25,
        text_exact=5,
        timing=20,
        recall=15,
        precision=10,
        line_breaks=5,
        styling=5,
        position=6,
        anchor=3,
        intent=1,
        fade=5,
    )


class MatchedPair(BaseModel):
    ref_index: int
    out_index: int
    iou: float
    text_plain: float
    text_exact: float
    timing: float
    line_breaks: float
    styling: float | None = None
    position: float | None = None
    anchor: float | None = None
    intent: float | None = None
    fade: float | None = None


class ScoreReport(BaseModel):
    final: float
    sub_scores: dict[str, float | None]
    effective_weights: dict[str, float]
    n_ref: int
    n_out: int
    n_matched: int
    matched_pairs: list[MatchedPair]
    warnings: list[str]
