"""Position pillar sub-scores (ADR-0008).

Three independent axes inside the position pillar:

- ``position`` — continuous score on the (x, y) distance between two effective
  anchor points. Applies to every paired event. Cliff-linear similarity,
  ``d_max = diagonal × 0.10``.
- ``anchor`` — binary match on the resolved alignment direction (\\an).
  Source-agnostic (inline \\an and ``Style.Alignment`` interchangeable).
- ``intent`` — binary, asymmetric. Fires only when ref has an inline ``\\pos``
  or ``\\move``; rewards output emitting any override (``\\pos`` or ``\\move``).
  ``None`` otherwise, so the case "output adds a superfluous \\pos" is
  unpunished (output added redundant information, ref had none to lose).

``effective_anchor`` resolves an event to (points, alignment, source) using
inline tags when present, falling back to the style's alignment + margins +
``PlayResX/Y``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pysubs2

from subtitles_ocr.evaluation._tags import parse_event_text


Source = Literal["pos", "move", "style"]


@dataclass(frozen=True)
class EffectiveAnchor:
    points: list[tuple[float, float]]
    alignment: int
    source: Source


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _play_res(subs: pysubs2.SSAFile) -> tuple[int, int]:
    info = subs.info if isinstance(subs.info, dict) else {}
    try:
        x = int(info.get("PlayResX", 1920))
        y = int(info.get("PlayResY", 1080))
        return (x, y)
    except (TypeError, ValueError):
        return (1920, 1080)


def _anchor_from_style(
    play_res: tuple[int, int],
    alignment: int,
    margin_l: float,
    margin_r: float,
    margin_v: float,
) -> tuple[float, float]:
    width, height = play_res
    # Horizontal: 1/4/7 left, 2/5/8 centre, 3/6/9 right.
    horiz = alignment % 3
    if horiz == 1:  # left
        x = margin_l
    elif horiz == 0:  # right (3, 6, 9)
        x = width - margin_r
    else:  # centre (2, 5, 8)
        x = (margin_l + (width - margin_r)) / 2.0
    # Vertical: 1-3 bottom, 4-6 middle (margins ignored), 7-9 top.
    if 1 <= alignment <= 3:
        y = height - margin_v
    elif 7 <= alignment <= 9:
        y = margin_v
    else:
        y = height / 2.0
    return (float(x), float(y))


def effective_anchor(event: pysubs2.SSAEvent, subs: pysubs2.SSAFile) -> EffectiveAnchor:
    parsed = parse_event_text(event.text)
    style = subs.styles.get(event.style)
    if style is None:
        # Fall back to pysubs2's built-in Default if the named style is missing.
        style = pysubs2.SSAStyle()
    alignment = parsed.alignment if parsed.alignment is not None else int(style.alignment)
    play_res = _play_res(subs)

    if parsed.pos is not None:
        return EffectiveAnchor(points=[parsed.pos], alignment=alignment, source="pos")
    if parsed.move is not None:
        m = parsed.move
        return EffectiveAnchor(
            points=[(m[0], m[1]), (m[2], m[3])],
            alignment=alignment,
            source="move",
        )
    point = _anchor_from_style(
        play_res,
        alignment,
        float(style.marginl),
        float(style.marginr),
        float(style.marginv),
    )
    return EffectiveAnchor(points=[point], alignment=alignment, source="style")


# ---------------------------------------------------------------------------
# Pair scores
# ---------------------------------------------------------------------------


def _endpoint_pair(eff: EffectiveAnchor) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(eff.points) >= 2:
        return eff.points[0], eff.points[1]
    p = eff.points[0]
    return p, p


def _cliff(distance: float, diag: float) -> float:
    if diag <= 0:
        return 1.0 if distance == 0 else 0.0
    d_max = diag * 0.10
    return max(0.0, 1.0 - distance / d_max)


def position_pair_score(
    out: EffectiveAnchor,
    ref: EffectiveAnchor,
    play_res: tuple[int, int],
) -> float:
    out_start, out_end = _endpoint_pair(out)
    ref_start, ref_end = _endpoint_pair(ref)
    diag = math.hypot(play_res[0], play_res[1])
    s_start = _cliff(math.hypot(out_start[0] - ref_start[0], out_start[1] - ref_start[1]), diag)
    s_end = _cliff(math.hypot(out_end[0] - ref_end[0], out_end[1] - ref_end[1]), diag)
    return (s_start + s_end) / 2.0


def anchor_pair_score(out: EffectiveAnchor, ref: EffectiveAnchor) -> float:
    return 1.0 if out.alignment == ref.alignment else 0.0


def intent_pair_score(out: EffectiveAnchor, ref: EffectiveAnchor) -> float | None:
    if ref.source not in ("pos", "move"):
        return None
    return 1.0 if out.source in ("pos", "move") else 0.0


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def position_score(
    pairs: list[tuple[EffectiveAnchor, EffectiveAnchor]],
    play_res: tuple[int, int],
) -> float | None:
    if not pairs:
        return None
    scores = [position_pair_score(o, r, play_res) for o, r in pairs]
    return sum(scores) / len(scores)


def anchor_score(pairs: list[tuple[EffectiveAnchor, EffectiveAnchor]]) -> float | None:
    if not pairs:
        return None
    scores = [anchor_pair_score(o, r) for o, r in pairs]
    return sum(scores) / len(scores)


def intent_score(pairs: list[tuple[EffectiveAnchor, EffectiveAnchor]]) -> float | None:
    scored = [s for s in (intent_pair_score(o, r) for o, r in pairs) if s is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)
