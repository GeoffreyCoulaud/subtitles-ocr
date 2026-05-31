"""Styling sub-score with source-agnostic resolution (ADR-0008-style).

Each styling attribute (italic / bold / underline / strikeout, primary
colour, outline colour, font size, rotation z) is resolved per event from
either an inline override tag *or* the event's Style. The pair score
compares resolved values, so a ref-Style-derived italic matched against
an output-Style-derived italic of the same value scores 1.0 — fixing the
asymmetry that ADR-0006 §5.7's inline-only logic introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import pysubs2

from subtitles_ocr.evaluation._colour import colour_score
from subtitles_ocr.evaluation._tags import parse_event_text


@dataclass(frozen=True)
class ResolvedStyling:
    italic: bool
    bold: bool
    underline: bool
    strikeout: bool
    primary_colour: tuple[int, int, int]
    outline_colour: tuple[int, int, int]
    font_size: float
    rotation_z: float


def _colour_tuple(c: pysubs2.Color | None, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if c is None:
        return fallback
    return (int(c.r), int(c.g), int(c.b))


def resolve_styling(event: pysubs2.SSAEvent, subs: pysubs2.SSAFile) -> ResolvedStyling:
    parsed = parse_event_text(event.text)
    style = subs.styles.get(event.style) or pysubs2.SSAStyle()

    # Binary: inline ON wins. (Explicit inline OFF like \i0 is rare in
    # practice and not distinguished here; the resolver treats False as
    # "no inline override" and falls back to the Style attribute.)
    italic = parsed.italic or bool(style.italic)
    bold = parsed.bold or bool(style.bold)
    underline = parsed.underline or bool(style.underline)
    strikeout = parsed.strikeout or bool(style.strikeout)

    primary = parsed.primary_colour
    if primary is None:
        primary = _colour_tuple(style.primarycolor, (255, 255, 255))

    outline = parsed.outline_colour
    if outline is None:
        outline = _colour_tuple(style.outlinecolor, (0, 0, 0))

    font_size = parsed.font_size if parsed.font_size is not None else float(style.fontsize)

    rotation_z = (
        parsed.rotation_z if parsed.rotation_z is not None else float(style.angle)
    )

    return ResolvedStyling(
        italic=italic,
        bold=bold,
        underline=underline,
        strikeout=strikeout,
        primary_colour=primary,
        outline_colour=outline,
        font_size=font_size,
        rotation_z=rotation_z,
    )


def _binary(a: bool, b: bool) -> float:
    return 1.0 if a == b else 0.0


def _ratio(a: float, b: float) -> float:
    if max(a, b) == 0.0:
        return 1.0
    return min(a, b) / max(a, b)


def _rotation(a: float, b: float) -> float:
    return max(0.0, 1.0 - abs(a - b) / 180.0)


def styling_pair_score(out: ResolvedStyling, ref: ResolvedStyling) -> float:
    """Mean across all 8 components on resolved values.

    Every paired event has a defined value for every component (style
    fallback ensures that), so the pair score is always defined.
    """
    components = [
        _binary(out.italic, ref.italic),
        _binary(out.bold, ref.bold),
        _binary(out.underline, ref.underline),
        _binary(out.strikeout, ref.strikeout),
        colour_score(out.primary_colour, ref.primary_colour),
        colour_score(out.outline_colour, ref.outline_colour),
        _ratio(out.font_size, ref.font_size),
        _rotation(out.rotation_z, ref.rotation_z),
    ]
    return sum(components) / len(components)


def styling_score(
    pairs: list[tuple[ResolvedStyling, ResolvedStyling]],
) -> float | None:
    if not pairs:
        return None
    scores = [styling_pair_score(o, r) for o, r in pairs]
    return sum(scores) / len(scores)
