"""Recall and precision (ADR-0006 §5.4-5.5)."""

from __future__ import annotations


def recall(n_matched: int, n_ref: int) -> float:
    if n_ref <= 0:
        raise ValueError("Recall is undefined when the reference has no cues.")
    return n_matched / n_ref


def precision(n_matched: int, n_out: int) -> float:
    if n_out <= 0:
        return 1.0
    return n_matched / n_out
