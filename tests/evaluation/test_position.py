"""Tests for position pillar sub-scores (ADR-0008).

Three independent axes inside the position pillar:
- position: numeric distance between (x, y) effective anchors (continuous).
- anchor:   binary match of resolved alignment direction (\\an).
- intent:   asymmetric — fires only when ref has an inline \\pos or \\move.

Effective anchor is resolved from inline \\pos / \\move when present, else
derived from style + alignment + margins + PlayResX/Y.
"""

from __future__ import annotations

import math

import pysubs2
import pytest

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


def _make_subs(
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    styles: dict[str, dict] | None = None,
) -> pysubs2.SSAFile:
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(play_res_x)
    subs.info["PlayResY"] = str(play_res_y)
    styles = styles or {"Default": dict(alignment=2, marginl=0, marginr=0, marginv=10)}
    for name, kwargs in styles.items():
        style = pysubs2.SSAStyle()
        for k, v in kwargs.items():
            setattr(style, k, v)
        subs.styles[name] = style
    return subs


def _event(text: str, style: str = "Default") -> pysubs2.SSAEvent:
    return pysubs2.SSAEvent(start=0, end=1000, text=text, style=style)


# ---------------------------------------------------------------------------
# effective_anchor: inline \pos
# ---------------------------------------------------------------------------


def test_effective_anchor_inline_pos_uses_style_alignment_when_no_an() -> None:
    subs = _make_subs(styles={"Default": dict(alignment=2, marginl=0, marginr=0, marginv=10)})
    eff = effective_anchor(_event(r"{\pos(960,540)}hello"), subs)
    assert eff.points == [(960.0, 540.0)]
    assert eff.alignment == 2
    assert eff.source == "pos"


def test_effective_anchor_inline_an_overrides_style_alignment() -> None:
    subs = _make_subs(styles={"Default": dict(alignment=2, marginl=0, marginr=0, marginv=10)})
    eff = effective_anchor(_event(r"{\an8\pos(960,100)}hello"), subs)
    assert eff.alignment == 8
    assert eff.source == "pos"


# ---------------------------------------------------------------------------
# effective_anchor: inline \move
# ---------------------------------------------------------------------------


def test_effective_anchor_inline_move_returns_two_points() -> None:
    subs = _make_subs()
    eff = effective_anchor(_event(r"{\move(100,100,500,500)}hello"), subs)
    assert eff.points == [(100.0, 100.0), (500.0, 500.0)]
    assert eff.source == "move"


# ---------------------------------------------------------------------------
# effective_anchor: pure style (no inline override)
# ---------------------------------------------------------------------------


def test_effective_anchor_pure_style_bottom_centre_uses_margins() -> None:
    subs = _make_subs(styles={"Default": dict(alignment=2, marginl=0, marginr=0, marginv=17)})
    eff = effective_anchor(_event("hello"), subs)
    assert eff.points == [(960.0, 1063.0)]
    assert eff.alignment == 2
    assert eff.source == "style"


def test_effective_anchor_pure_style_top_left_uses_margins() -> None:
    subs = _make_subs(styles={"Default": dict(alignment=7, marginl=20, marginr=0, marginv=30)})
    eff = effective_anchor(_event("hello"), subs)
    assert eff.points == [(20.0, 30.0)]


def test_effective_anchor_pure_style_top_right_uses_margins() -> None:
    subs = _make_subs(styles={"Default": dict(alignment=9, marginl=0, marginr=40, marginv=30)})
    eff = effective_anchor(_event("hello"), subs)
    assert eff.points == [(1880.0, 30.0)]


def test_effective_anchor_pure_style_middle_centre_ignores_marginv() -> None:
    subs = _make_subs(styles={"Default": dict(alignment=5, marginl=0, marginr=0, marginv=100)})
    eff = effective_anchor(_event("hello"), subs)
    assert eff.points == [(960.0, 540.0)]


def test_effective_anchor_pure_style_bottom_centre_with_horizontal_margins() -> None:
    # Centre alignment with non-zero horizontal margins still centres between
    # MarginL and PlayResX - MarginR.
    subs = _make_subs(styles={"Default": dict(alignment=2, marginl=100, marginr=200, marginv=17)})
    eff = effective_anchor(_event("hello"), subs)
    # x = (100 + (1920 - 200)) / 2 = 910
    assert eff.points == [(910.0, 1063.0)]


def test_effective_anchor_inline_an_overrides_alignment_but_keeps_style_margins() -> None:
    subs = _make_subs(styles={"Default": dict(alignment=2, marginl=0, marginr=0, marginv=17)})
    eff = effective_anchor(_event(r"{\an8}hello"), subs)
    # \an8 = top-centre, so y = MarginV = 17, x = PlayResX/2 = 960.
    assert eff.points == [(960.0, 17.0)]
    assert eff.alignment == 8
    assert eff.source == "style"


# ---------------------------------------------------------------------------
# position_pair_score
# ---------------------------------------------------------------------------


def _anchor(point: tuple[float, float], alignment: int = 2, source: str = "pos") -> EffectiveAnchor:
    return EffectiveAnchor(points=[point], alignment=alignment, source=source)


def test_position_pair_score_identical_anchors_score_one() -> None:
    a = _anchor((960.0, 1063.0))
    b = _anchor((960.0, 1063.0))
    assert position_pair_score(a, b, play_res=(1920, 1080)) == 1.0


def test_position_pair_score_beyond_d_max_scores_zero() -> None:
    diag = math.hypot(1920, 1080)
    offset = 0.10 * diag + 1.0  # just past the cliff
    a = _anchor((0.0, 0.0))
    b = _anchor((offset, 0.0))
    assert position_pair_score(a, b, play_res=(1920, 1080)) == 0.0


def test_position_pair_score_at_half_cliff_scores_half() -> None:
    diag = math.hypot(1920, 1080)
    offset = 0.05 * diag  # half of d_max
    a = _anchor((0.0, 0.0))
    b = _anchor((offset, 0.0))
    score = position_pair_score(a, b, play_res=(1920, 1080))
    assert score == pytest.approx(0.5, abs=1e-6)


def test_position_pair_score_both_move_per_endpoint_average() -> None:
    # Ref: (100,100)→(500,500); Out: (100,100)→(500,500). Identical.
    a = EffectiveAnchor(points=[(100.0, 100.0), (500.0, 500.0)], alignment=2, source="move")
    b = EffectiveAnchor(points=[(100.0, 100.0), (500.0, 500.0)], alignment=2, source="move")
    assert position_pair_score(a, b, play_res=(1920, 1080)) == 1.0


def test_position_pair_score_static_vs_move_treats_static_as_degenerate() -> None:
    # Ref \move from (100,100) to (500,500); out static at (100,100).
    # Start endpoint matches perfectly; end endpoint distance = √(400²+400²).
    play_res = (1920, 1080)
    diag = math.hypot(*play_res)
    d_max = 0.10 * diag
    d_end = math.hypot(400, 400)
    expected_end = max(0.0, 1.0 - d_end / d_max)
    expected = (1.0 + expected_end) / 2.0

    ref = EffectiveAnchor(points=[(100.0, 100.0), (500.0, 500.0)], alignment=2, source="move")
    out = _anchor((100.0, 100.0))
    assert position_pair_score(out, ref, play_res) == pytest.approx(expected)


def test_position_pair_score_move_vs_static_symmetric() -> None:
    # Same as previous but the static side is reversed (ref static, out \move).
    play_res = (1920, 1080)
    diag = math.hypot(*play_res)
    d_max = 0.10 * diag
    d_end = math.hypot(400, 400)
    expected_end = max(0.0, 1.0 - d_end / d_max)
    expected = (1.0 + expected_end) / 2.0

    out = EffectiveAnchor(points=[(100.0, 100.0), (500.0, 500.0)], alignment=2, source="move")
    ref = _anchor((100.0, 100.0))
    assert position_pair_score(out, ref, play_res) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# anchor_pair_score
# ---------------------------------------------------------------------------


def test_anchor_pair_score_matching_alignment_returns_one() -> None:
    a = _anchor((0.0, 0.0), alignment=2)
    b = _anchor((0.0, 0.0), alignment=2)
    assert anchor_pair_score(a, b) == 1.0


def test_anchor_pair_score_mismatching_alignment_returns_zero() -> None:
    a = _anchor((0.0, 0.0), alignment=2)
    b = _anchor((0.0, 0.0), alignment=8)
    assert anchor_pair_score(a, b) == 0.0


def test_anchor_pair_score_source_agnostic_inline_an_vs_style_alignment() -> None:
    # Same resolved alignment (2) regardless of how it was set — match.
    a = _anchor((0.0, 0.0), alignment=2, source="pos")
    b = _anchor((0.0, 0.0), alignment=2, source="style")
    assert anchor_pair_score(a, b) == 1.0


# ---------------------------------------------------------------------------
# intent_pair_score
# ---------------------------------------------------------------------------


def test_intent_pair_score_ref_pure_style_returns_none() -> None:
    out = _anchor((0.0, 0.0), source="pos")
    ref = _anchor((0.0, 0.0), source="style")
    assert intent_pair_score(out, ref) is None


def test_intent_pair_score_ref_pos_and_out_pos_returns_one() -> None:
    out = _anchor((0.0, 0.0), source="pos")
    ref = _anchor((0.0, 0.0), source="pos")
    assert intent_pair_score(out, ref) == 1.0


def test_intent_pair_score_ref_pos_and_out_style_returns_zero() -> None:
    out = _anchor((0.0, 0.0), source="style")
    ref = _anchor((0.0, 0.0), source="pos")
    assert intent_pair_score(out, ref) == 0.0


def test_intent_pair_score_ref_move_and_out_pos_returns_one() -> None:
    # Both have inline override (different kinds, but both qualify).
    out = _anchor((0.0, 0.0), source="pos")
    ref = EffectiveAnchor(points=[(0.0, 0.0), (1.0, 1.0)], alignment=2, source="move")
    assert intent_pair_score(out, ref) == 1.0


def test_intent_pair_score_ref_move_and_out_move_returns_one() -> None:
    out = EffectiveAnchor(points=[(0.0, 0.0), (1.0, 1.0)], alignment=2, source="move")
    ref = EffectiveAnchor(points=[(0.0, 0.0), (1.0, 1.0)], alignment=2, source="move")
    assert intent_pair_score(out, ref) == 1.0


# ---------------------------------------------------------------------------
# aggregate scores
# ---------------------------------------------------------------------------


def test_position_score_is_mean_of_pair_scores() -> None:
    play_res = (1920, 1080)
    diag = math.hypot(*play_res)
    half_offset = 0.05 * diag  # halfway through d_max
    pair_full = (_anchor((0.0, 0.0)), _anchor((0.0, 0.0)))
    pair_half = (_anchor((0.0, 0.0)), _anchor((half_offset, 0.0)))
    assert position_score([pair_full, pair_half], play_res) == pytest.approx(0.75, abs=1e-6)


def test_position_score_empty_returns_none() -> None:
    assert position_score([], (1920, 1080)) is None


def test_anchor_score_is_mean_of_pair_scores() -> None:
    a = _anchor((0.0, 0.0), alignment=2)
    b_match = _anchor((0.0, 0.0), alignment=2)
    b_mismatch = _anchor((0.0, 0.0), alignment=8)
    assert anchor_score([(a, b_match), (a, b_mismatch)]) == 0.5


def test_anchor_score_empty_returns_none() -> None:
    assert anchor_score([]) is None


def test_intent_score_excludes_none_pairs() -> None:
    out_pos = _anchor((0.0, 0.0), source="pos")
    ref_pos = _anchor((0.0, 0.0), source="pos")
    out_style = _anchor((0.0, 0.0), source="style")
    ref_style = _anchor((0.0, 0.0), source="style")
    # Pair 1 ref=style → None (excluded); pair 2 ref=pos, out=pos → 1.0
    assert intent_score([(out_style, ref_style), (out_pos, ref_pos)]) == 1.0


def test_intent_score_all_none_returns_none() -> None:
    out = _anchor((0.0, 0.0), source="style")
    ref = _anchor((0.0, 0.0), source="style")
    assert intent_score([(out, ref), (out, ref)]) is None


def test_intent_score_empty_returns_none() -> None:
    assert intent_score([]) is None
