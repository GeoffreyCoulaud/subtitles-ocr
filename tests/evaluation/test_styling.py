"""Tests for styling sub-score with source-agnostic resolution.

ADR-0008 split anchor + position; the same source-agnostic pattern is
extended here: styling components (italic/bold/underline/strikeout,
colors, font size, rotation) are resolved from inline tags *or* the
event's Style. The pair score compares resolved values, so a ref-Style-
derived italic vs an output-Style-derived italic of the same value
matches at 1.0 instead of being silently ignored.
"""

from __future__ import annotations

import math

import pysubs2
import pytest

from subtitles_ocr.evaluation.styling import (
    ResolvedStyling,
    resolve_styling,
    styling_pair_score,
    styling_score,
)


def _make_subs(styles: dict[str, dict] | None = None) -> pysubs2.SSAFile:
    subs = pysubs2.SSAFile()
    styles = styles or {"Default": {}}
    for name, kwargs in styles.items():
        style = pysubs2.SSAStyle()
        for k, v in kwargs.items():
            setattr(style, k, v)
        subs.styles[name] = style
    return subs


def _event(text: str, style: str = "Default") -> pysubs2.SSAEvent:
    return pysubs2.SSAEvent(start=0, end=1000, text=text, style=style)


def _white() -> tuple[int, int, int]:
    return (255, 255, 255)


def _black() -> tuple[int, int, int]:
    return (0, 0, 0)


# ---------------------------------------------------------------------------
# resolve_styling
# ---------------------------------------------------------------------------


def test_resolve_styling_inline_italic_overrides_style() -> None:
    subs = _make_subs(styles={"Default": dict(italic=False)})
    rs = resolve_styling(_event(r"{\i1}foo"), subs)
    assert rs.italic is True


def test_resolve_styling_style_italic_when_no_inline() -> None:
    subs = _make_subs(styles={"Default": dict(italic=True)})
    rs = resolve_styling(_event("foo"), subs)
    assert rs.italic is True


def test_resolve_styling_no_italic_anywhere() -> None:
    subs = _make_subs(styles={"Default": dict(italic=False)})
    rs = resolve_styling(_event("foo"), subs)
    assert rs.italic is False


def test_resolve_styling_inline_bold() -> None:
    subs = _make_subs()
    rs = resolve_styling(_event(r"{\b1}foo"), subs)
    assert rs.bold is True


def test_resolve_styling_style_font_size() -> None:
    subs = _make_subs(styles={"Default": dict(fontsize=50.0)})
    rs = resolve_styling(_event("foo"), subs)
    assert rs.font_size == 50.0


def test_resolve_styling_inline_font_size_overrides_style() -> None:
    subs = _make_subs(styles={"Default": dict(fontsize=50.0)})
    rs = resolve_styling(_event(r"{\fs80}foo"), subs)
    assert rs.font_size == 80.0


def test_resolve_styling_style_primary_colour_when_no_inline() -> None:
    subs = _make_subs(styles={
        "Default": dict(primarycolor=pysubs2.Color(10, 20, 30, 0)),
    })
    rs = resolve_styling(_event("foo"), subs)
    assert rs.primary_colour == (10, 20, 30)


def test_resolve_styling_inline_primary_colour_overrides_style() -> None:
    subs = _make_subs(styles={
        "Default": dict(primarycolor=pysubs2.Color(10, 20, 30, 0)),
    })
    rs = resolve_styling(_event(r"{\c&H00FF8040&}foo"), subs)
    # ASS hex is &HAABBGGRR: alpha=00, B=FF, G=80, R=40 → RGB=(0x40, 0x80, 0xFF)
    assert rs.primary_colour == (0x40, 0x80, 0xFF)


def test_resolve_styling_inline_rotation_overrides_style_angle() -> None:
    subs = _make_subs(styles={"Default": dict(angle=10.0)})
    rs = resolve_styling(_event(r"{\frz45}foo"), subs)
    assert rs.rotation_z == 45.0


def test_resolve_styling_style_angle_when_no_inline_frz() -> None:
    subs = _make_subs(styles={"Default": dict(angle=15.0)})
    rs = resolve_styling(_event("foo"), subs)
    assert rs.rotation_z == 15.0


def test_resolve_styling_default_rotation_is_zero() -> None:
    subs = _make_subs()
    rs = resolve_styling(_event("foo"), subs)
    assert rs.rotation_z == 0.0


# ---------------------------------------------------------------------------
# styling_pair_score
# ---------------------------------------------------------------------------


def _styling(
    italic: bool = False,
    bold: bool = False,
    underline: bool = False,
    strikeout: bool = False,
    primary: tuple[int, int, int] | None = None,
    outline: tuple[int, int, int] | None = None,
    font_size: float = 34.0,
    rotation_z: float = 0.0,
) -> ResolvedStyling:
    return ResolvedStyling(
        italic=italic,
        bold=bold,
        underline=underline,
        strikeout=strikeout,
        primary_colour=primary or _white(),
        outline_colour=outline or _black(),
        font_size=font_size,
        rotation_z=rotation_z,
    )


def test_styling_pair_score_all_matching_defaults_scores_one() -> None:
    a = _styling()
    b = _styling()
    assert styling_pair_score(a, b) == 1.0


def test_styling_pair_score_italic_mismatch_drops_below_one() -> None:
    a = _styling(italic=True)
    b = _styling(italic=False)
    # 1 of 8 components fails (italic). Components scoring 1.0: 7. Mean: 7/8.
    assert styling_pair_score(a, b) == pytest.approx(7 / 8)


def test_styling_pair_score_font_size_half_drops_one_component_to_half() -> None:
    a = _styling(font_size=20.0)
    b = _styling(font_size=40.0)
    # font_size ratio = 0.5; other 7 components match. Mean = (7 + 0.5) / 8.
    assert styling_pair_score(a, b) == pytest.approx(7.5 / 8)


def test_styling_pair_score_rotation_180_drops_one_component_to_zero() -> None:
    a = _styling(rotation_z=0.0)
    b = _styling(rotation_z=180.0)
    # rotation distance = 180; score = 0.0; others 1.0. Mean = 7/8.
    assert styling_pair_score(a, b) == pytest.approx(7 / 8)


def test_styling_pair_score_rotation_small_difference_close_to_one() -> None:
    a = _styling(rotation_z=0.0)
    b = _styling(rotation_z=10.0)
    # rotation similarity = 1 - 10/180 ≈ 0.944; mean ≈ (7 + 0.944) / 8
    assert styling_pair_score(a, b) == pytest.approx((7 + (1 - 10 / 180)) / 8)


def test_styling_pair_score_primary_colour_difference() -> None:
    a = _styling(primary=(255, 255, 255))
    b = _styling(primary=(0, 0, 0))
    # Max colour distance → primary component ≤ 0.0; other 7 match.
    # styling pair score ≤ 7/8.
    score = styling_pair_score(a, b)
    assert score is not None
    assert score <= 7 / 8


# ---------------------------------------------------------------------------
# styling_score aggregation
# ---------------------------------------------------------------------------


def test_styling_score_mean_of_pairs() -> None:
    perfect = (_styling(), _styling())
    italic_mismatch = (_styling(italic=True), _styling(italic=False))
    # First pair: 1.0; second: 7/8. Mean: (1.0 + 7/8) / 2 = 15/16
    score = styling_score([perfect, italic_mismatch])
    assert score == pytest.approx(15 / 16)


def test_styling_score_empty_returns_none() -> None:
    assert styling_score([]) is None
