"""ASS override tag parsing for the evaluation engine (ADR-0006 §5.7-5.9)."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedEvent:
    plain_text: str
    italic: bool
    bold: bool
    underline: bool
    strikeout: bool
    primary_colour: tuple[int, int, int] | None
    outline_colour: tuple[int, int, int] | None
    font_size: float | None
    rotation_z: float | None
    alignment: int | None
    pos: tuple[float, float] | None
    move: tuple[float, float, float, float, int | None, int | None] | None
    fade: tuple[int, int] | None


_OVERRIDE_BLOCK = re.compile(r"\{([^{}]*)\}")
_COLOUR_VALUE = re.compile(r"&H?([0-9A-Fa-f]{1,8})&?")


def _parse_colour(raw: str) -> tuple[int, int, int] | None:
    m = _COLOUR_VALUE.search(raw)
    if not m:
        return None
    hex_str = m.group(1).rjust(8, "0")
    # &HAABBGGRR — alpha then BGR
    b = int(hex_str[-6:-4], 16)
    g = int(hex_str[-4:-2], 16)
    r = int(hex_str[-2:], 16)
    return (r, g, b)


def _find_first_value(block_text: str, pattern: str) -> str | None:
    m = re.search(pattern, block_text)
    return m.group(1) if m else None


def parse_event_text(raw: str) -> ParsedEvent:
    blocks = _OVERRIDE_BLOCK.findall(raw)
    plain = _OVERRIDE_BLOCK.sub("", raw)

    joined = " ".join(blocks)

    italic = bool(re.search(r"\\i1\b", joined))
    bold = bool(re.search(r"\\b1\b", joined))
    underline = bool(re.search(r"\\u1\b", joined))
    strikeout = bool(re.search(r"\\s1\b", joined))

    primary_raw = _find_first_value(joined, r"\\1?c\s*(&H[0-9A-Fa-f]+&?|&H?[0-9A-Fa-f]+)")
    primary = _parse_colour(primary_raw) if primary_raw else None

    outline_raw = _find_first_value(joined, r"\\3c\s*(&H[0-9A-Fa-f]+&?|&H?[0-9A-Fa-f]+)")
    outline = _parse_colour(outline_raw) if outline_raw else None

    fs_raw = _find_first_value(joined, r"\\fs([-\d.]+)")
    font_size = float(fs_raw) if fs_raw else None

    frz_raw = _find_first_value(joined, r"\\frz([-\d.]+)")
    rotation_z = float(frz_raw) if frz_raw else None

    an_raw = _find_first_value(joined, r"\\an(\d)")
    alignment = int(an_raw) if an_raw else None

    pos_m = re.search(r"\\pos\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", joined)
    pos = (float(pos_m.group(1)), float(pos_m.group(2))) if pos_m else None

    move_m = re.search(
        r"\\move\("
        r"\s*([-\d.]+)\s*,\s*([-\d.]+)\s*"
        r",\s*([-\d.]+)\s*,\s*([-\d.]+)\s*"
        r"(?:,\s*([-\d]+)\s*,\s*([-\d]+)\s*)?"
        r"\)",
        joined,
    )
    if move_m:
        t1 = int(move_m.group(5)) if move_m.group(5) else None
        t2 = int(move_m.group(6)) if move_m.group(6) else None
        move = (
            float(move_m.group(1)),
            float(move_m.group(2)),
            float(move_m.group(3)),
            float(move_m.group(4)),
            t1,
            t2,
        )
    else:
        move = None

    fade_m = re.search(r"\\fad\(\s*([-\d]+)\s*,\s*([-\d]+)\s*\)", joined)
    fade = (int(fade_m.group(1)), int(fade_m.group(2))) if fade_m else None

    return ParsedEvent(
        plain_text=plain,
        italic=italic,
        bold=bold,
        underline=underline,
        strikeout=strikeout,
        primary_colour=primary,
        outline_colour=outline,
        font_size=font_size,
        rotation_z=rotation_z,
        alignment=alignment,
        pos=pos,
        move=move,
        fade=fade,
    )
