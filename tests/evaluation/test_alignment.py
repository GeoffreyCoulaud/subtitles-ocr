"""Tests for temporal-IoU alignment (ADR-0006 §4)."""

from __future__ import annotations

from subtitles_ocr.evaluation.alignment import AlignmentResult, Cue, align_by_iou


def _cue(start: float, end: float) -> Cue:
    return Cue(start_s=start, end_s=end)


def test_identical_cue_lists_pair_one_to_one() -> None:
    refs = [_cue(0.0, 1.0), _cue(2.0, 3.0)]
    outs = [_cue(0.0, 1.0), _cue(2.0, 3.0)]
    result = align_by_iou(refs, outs)
    assert result.pairs == [(0, 0, 1.0), (1, 1, 1.0)]
    assert result.unmatched_ref == []
    assert result.unmatched_out == []


def test_disjoint_cues_produce_no_pairs() -> None:
    refs = [_cue(0.0, 1.0)]
    outs = [_cue(5.0, 6.0)]
    result = align_by_iou(refs, outs)
    assert result.pairs == []
    assert result.unmatched_ref == [0]
    assert result.unmatched_out == [0]


def test_partial_overlap_matches_with_correct_iou() -> None:
    refs = [_cue(0.0, 2.0)]
    outs = [_cue(1.0, 3.0)]
    # overlap = 1.0, union = 3.0 → IoU = 1/3
    result = align_by_iou(refs, outs)
    assert len(result.pairs) == 1
    ref_i, out_i, iou = result.pairs[0]
    assert (ref_i, out_i) == (0, 0)
    assert iou == 1 / 3


def test_contention_resolved_by_highest_iou() -> None:
    # Two refs both overlap a single output cue.
    refs = [_cue(0.0, 1.0), _cue(0.5, 1.5)]
    outs = [_cue(0.4, 1.4)]
    result = align_by_iou(refs, outs)
    # ref 1 has IoU 0.8/1.1 ≈ 0.727; ref 0 has 0.6/1.4 ≈ 0.428. ref 1 wins.
    assert len(result.pairs) == 1
    assert result.pairs[0][0] == 1
    assert result.unmatched_ref == [0]


def test_empty_reference_returns_empty_alignment() -> None:
    result = align_by_iou([], [_cue(0.0, 1.0)])
    assert result.pairs == []
    assert result.unmatched_ref == []
    assert result.unmatched_out == [0]


def test_empty_output_returns_all_refs_unmatched() -> None:
    result = align_by_iou([_cue(0.0, 1.0), _cue(2.0, 3.0)], [])
    assert result.pairs == []
    assert result.unmatched_ref == [0, 1]
    assert result.unmatched_out == []
