"""Position sub-score (ADR-0006 §5.8)."""

from __future__ import annotations

import math
import re

from subtitles_ocr.evaluation._tags import ParsedEvent, parse_event_text


_ANCHOR_OFFSET = {
    1: (+0.5, -0.5),
    2: (0.0, -0.5),
    3: (-0.5, -0.5),
    4: (+0.5, 0.0),
    5: (0.0, 0.0),
    6: (-0.5, 0.0),
    7: (+0.5, +0.5),
    8: (0.0, +0.5),
    9: (-0.5, +0.5),
}

_LINE_BREAK = re.compile(r"\\[Nn]")


def anchor_to_centre(
    anchor: tuple[float, float],
    an: int,
    width: float,
    height: float,
) -> tuple[float, float]:
    fx, fy = _ANCHOR_OFFSET[an]
    return (anchor[0] + fx * width, anchor[1] + fy * height)


def _estimate_bbox(parsed: ParsedEvent, default_font_size: float) -> tuple[float, float]:
    fs = parsed.font_size if parsed.font_size is not None else default_font_size
    lines = _LINE_BREAK.split(parsed.plain_text)
    max_chars = max((len(line) for line in lines), default=1)
    width = fs * 0.55 * max_chars
    height = fs * len(lines)
    return (width, height)


def _centre_for(
    parsed: ParsedEvent,
    default_font_size: float,
    anchor_override: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    if anchor_override is None:
        if parsed.pos is None:
            return None
        anchor = parsed.pos
    else:
        anchor = anchor_override
    an = parsed.alignment if parsed.alignment is not None else 2
    width, height = _estimate_bbox(parsed, default_font_size)
    return anchor_to_centre(anchor, an, width, height)


def _score_distance(
    a: tuple[float, float],
    b: tuple[float, float],
    play_res: tuple[int, int],
) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    distance = math.sqrt(dx * dx + dy * dy)
    diag = math.sqrt(play_res[0] ** 2 + play_res[1] ** 2)
    normalised = distance / diag if diag > 0 else 0.0
    d_max = 0.10
    return max(0.0, min(1.0, 1.0 - normalised / d_max))


def position_pair_score(
    out_text: str,
    ref_text: str,
    play_res: tuple[int, int],
    default_font_size: float,
) -> float | None:
    out = parse_event_text(out_text)
    ref = parse_event_text(ref_text)

    # \move case: compare endpoint centres
    out_move = out.move
    ref_move = ref.move
    if out_move is not None or ref_move is not None:
        if out_move is None or ref_move is None:
            return 0.0
        o_start_centre = _centre_for(out, default_font_size, anchor_override=(out_move[0], out_move[1]))
        o_end_centre = _centre_for(out, default_font_size, anchor_override=(out_move[2], out_move[3]))
        r_start_centre = _centre_for(ref, default_font_size, anchor_override=(ref_move[0], ref_move[1]))
        r_end_centre = _centre_for(ref, default_font_size, anchor_override=(ref_move[2], ref_move[3]))
        assert o_start_centre and o_end_centre and r_start_centre and r_end_centre
        s_start = _score_distance(o_start_centre, r_start_centre, play_res)
        s_end = _score_distance(o_end_centre, r_end_centre, play_res)
        return (s_start + s_end) / 2.0

    # \pos case: at least one side must specify \pos to enter the comparison
    if out.pos is None and ref.pos is None:
        return None
    o_centre = _centre_for(out, default_font_size)
    r_centre = _centre_for(ref, default_font_size)
    if o_centre is None or r_centre is None:
        return 0.0
    return _score_distance(o_centre, r_centre, play_res)


def position_score(
    pairs: list[tuple[str, str]],
    play_res: tuple[int, int],
    default_font_size: float,
) -> float | None:
    scored = [
        s
        for s in (position_pair_score(o, r, play_res, default_font_size) for o, r in pairs)
        if s is not None
    ]
    if not scored:
        return None
    return sum(scored) / len(scored)
