"""Top-level score() orchestration (ADR-0006 §6, §7; ADR-0008)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pysubs2

from subtitles_ocr.evaluation.alignment import Cue, align_by_iou
from subtitles_ocr.evaluation.fade import fade_pair_score, fade_score
from subtitles_ocr.evaluation.line_breaks import line_breaks_pair_score, line_breaks_score
from subtitles_ocr.evaluation.position import (
    EffectiveAnchor,
    anchor_pair_score,
    anchor_score,
    effective_anchor,
    intent_pair_score,
    intent_score,
    position_pair_score,
    position_score,
)
from subtitles_ocr.evaluation.recall_precision import precision, recall
from subtitles_ocr.evaluation.report import MatchedPair, ScoreReport, Weights
from subtitles_ocr.evaluation.styling import styling_pair_score, styling_score
from subtitles_ocr.evaluation.text import (
    text_exact_pair_score,
    text_exact_score,
    text_plain_pair_score,
    text_plain_score,
)
from subtitles_ocr.evaluation.timing import timing_pair_score, timing_score


def _load_dialogue(path: Path) -> tuple[list[pysubs2.SSAEvent], pysubs2.SSAFile]:
    subs = pysubs2.load(str(path))
    dialogue = [e for e in subs.events if e.type == "Dialogue"]
    return dialogue, subs


def _cue_of(event: pysubs2.SSAEvent) -> Cue:
    return Cue(start_s=event.start / 1000.0, end_s=event.end / 1000.0)


def _play_res(subs: pysubs2.SSAFile) -> tuple[int, int]:
    info = subs.info if isinstance(subs.info, dict) else {}
    try:
        x = int(info.get("PlayResX", 1920))
        y = int(info.get("PlayResY", 1080))
        return (x, y)
    except (TypeError, ValueError):
        return (1920, 1080)


def score(
    output_path: Path,
    reference_path: Path,
    weights: Weights,
    fps: Fraction,
    K: int = 10,
) -> ScoreReport:
    ref_events, ref_subs = _load_dialogue(reference_path)
    out_events, out_subs = _load_dialogue(output_path)

    if not ref_events:
        raise ValueError("Reference .ass has no dialogue events; scoring is undefined.")

    ref_cues = [_cue_of(e) for e in ref_events]
    out_cues = [_cue_of(e) for e in out_events]
    alignment = align_by_iou(ref_cues, out_cues)

    play_res = _play_res(ref_subs)

    # Pre-resolve effective anchors once per event.
    ref_anchors: list[EffectiveAnchor] = [effective_anchor(e, ref_subs) for e in ref_events]
    out_anchors: list[EffectiveAnchor] = [effective_anchor(e, out_subs) for e in out_events]

    text_pairs: list[tuple[str, str]] = []
    timing_pairs: list[tuple[Cue, Cue]] = []
    anchor_pairs: list[tuple[EffectiveAnchor, EffectiveAnchor]] = []
    matched_pairs: list[MatchedPair] = []
    warnings: list[str] = []

    for ref_i, out_i, iou in alignment.pairs:
        ref_e = ref_events[ref_i]
        out_e = out_events[out_i]
        out_eff = out_anchors[out_i]
        ref_eff = ref_anchors[ref_i]
        text_pairs.append((out_e.text, ref_e.text))
        timing_pairs.append((_cue_of(out_e), _cue_of(ref_e)))
        anchor_pairs.append((out_eff, ref_eff))
        matched_pairs.append(
            MatchedPair(
                ref_index=ref_i,
                out_index=out_i,
                iou=iou,
                text_plain=text_plain_pair_score(out_e.text, ref_e.text),
                text_exact=text_exact_pair_score(out_e.text, ref_e.text),
                timing=timing_pair_score(_cue_of(out_e), _cue_of(ref_e), fps=fps, K=K),
                line_breaks=line_breaks_pair_score(out_e.text, ref_e.text),
                styling=styling_pair_score(out_e.text, ref_e.text),
                position=position_pair_score(out_eff, ref_eff, play_res),
                anchor=anchor_pair_score(out_eff, ref_eff),
                intent=intent_pair_score(out_eff, ref_eff),
                fade=fade_pair_score(out_e.text, ref_e.text, fps=fps, K=K),
            )
        )

    n_matched = len(matched_pairs)
    n_ref = len(ref_events)
    n_out = len(out_events)

    if n_out == 0:
        warnings.append("Output has no dialogue events; precision is reported as 1.0 by convention.")

    sub_scores: dict[str, float | None] = {
        "text_plain": text_plain_score(text_pairs),
        "text_exact": text_exact_score(text_pairs),
        "timing": timing_score(timing_pairs, fps=fps, K=K),
        "recall": recall(n_matched, n_ref),
        "precision": precision(n_matched, n_out),
        "line_breaks": line_breaks_score(text_pairs),
        "styling": styling_score(text_pairs),
        "position": position_score(anchor_pairs, play_res),
        "anchor": anchor_score(anchor_pairs),
        "intent": intent_score(anchor_pairs),
        "fade": fade_score(text_pairs, fps=fps, K=K),
    }

    # Weighted sum with null-rescaling.
    weight_map = weights.model_dump()
    numerator = 0.0
    denominator = 0
    effective_weights: dict[str, float] = {}
    for key, value in sub_scores.items():
        w = weight_map[key]
        if value is None:
            effective_weights[key] = 0
            continue
        numerator += w * value
        denominator += w
        effective_weights[key] = w
    final = numerator / denominator if denominator > 0 else 0.0

    return ScoreReport(
        final=final,
        sub_scores=sub_scores,
        effective_weights=effective_weights,
        n_ref=n_ref,
        n_out=n_out,
        n_matched=n_matched,
        matched_pairs=matched_pairs,
        warnings=warnings,
    )
