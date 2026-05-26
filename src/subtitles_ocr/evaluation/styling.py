"""Styling sub-score (ADR-0006 §5.7)."""

from __future__ import annotations

from subtitles_ocr.evaluation._colour import colour_score
from subtitles_ocr.evaluation._tags import ParsedEvent, parse_event_text


def _binary(a: bool, b: bool) -> float:
    return 1.0 if a == b else 0.0


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None or b is None:
        return 0.0
    if max(a, b) == 0.0:
        return 1.0
    return min(a, b) / max(a, b)


def _rotation(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None or b is None:
        return 0.0
    return max(0.0, 1.0 - abs(a - b) / 180.0)


def _colour_component(
    a: tuple[int, int, int] | None,
    b: tuple[int, int, int] | None,
) -> float | None:
    if a is None and b is None:
        return None
    if a is None or b is None:
        return 0.0
    return colour_score(a, b)


def _components_for(pa: ParsedEvent, pb: ParsedEvent) -> list[float]:
    # Binary components are always considered "specified" — default False is meaningful.
    # We include them only when at least one side asserts True (i.e. styling is being applied).
    components: list[float] = []

    if pa.italic or pb.italic:
        components.append(_binary(pa.italic, pb.italic))
    if pa.bold or pb.bold:
        components.append(_binary(pa.bold, pb.bold))
    if pa.underline or pb.underline:
        components.append(_binary(pa.underline, pb.underline))
    if pa.strikeout or pb.strikeout:
        components.append(_binary(pa.strikeout, pb.strikeout))

    pc = _colour_component(pa.primary_colour, pb.primary_colour)
    if pc is not None:
        components.append(pc)

    oc = _colour_component(pa.outline_colour, pb.outline_colour)
    if oc is not None:
        components.append(oc)

    fs = _ratio(pa.font_size, pb.font_size)
    if fs is not None:
        components.append(fs)

    rz = _rotation(pa.rotation_z, pb.rotation_z)
    if rz is not None:
        components.append(rz)

    return components


def styling_pair_score(out_text: str, ref_text: str) -> float | None:
    pa = parse_event_text(out_text)
    pb = parse_event_text(ref_text)
    comps = _components_for(pa, pb)
    if not comps:
        return None
    return sum(comps) / len(comps)


def styling_score(pairs: list[tuple[str, str]]) -> float | None:
    scored = [s for s in (styling_pair_score(o, r) for o, r in pairs) if s is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)
