"""Stage 12 — ASS export (ADR-0002 §3 Stage 11, ADR-0003 §4.4)."""

from __future__ import annotations

import math
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import ClassVar, Literal

import pysubs2
from pydantic import BaseModel
from pysubs2 import Alignment, Color, SSAEvent, SSAFile, SSAStyle

from subtitles_ocr.config import ExportConfig, PipelineGlobals
from subtitles_ocr.pipeline.animation import AnimatedEvent, AnimationAnalysisResult
from subtitles_ocr.pipeline.color import ColorExtractionResult, EventColors
from subtitles_ocr.pipeline.normalize import NormalizeResult
from subtitles_ocr.timing import frame_to_ms

STAGE_VERSION: int = 3

# ADR-0002 §3 Stage 11
_FRZ_OMIT_THRESHOLD_DEG: float = 0.5
_NONLINEAR_COMMENT: str = "{!sign: animation non reconstruite!}"

PositionClass = Literal["Bottom", "Top", "Sign"]


class ExportResult(BaseModel):
    out_path_written: str
    event_count: int


class ExportStage:
    CONFIG_FIELD: ClassVar[str] = "export"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = (
        "workdir",
        "out_path",
        "fps",
        "fansub_width",
        "fansub_height",
    )

    def __init__(self) -> None:
        pass

    def run(self, globals: PipelineGlobals, config: ExportConfig) -> ExportResult:
        animation = _load_animation(globals.workdir)
        colors = _load_colors(globals.workdir)
        doc = _load_doc(globals.workdir)

        colors_by_id = {c.event_id: c for c in colors.events}
        text_by_id = {e.event_id: e.cleaned_text for e in doc.events}

        # Build a working list of (event, EventColors, position, cleaned_text).
        # Events absent from text_by_id were pruned by the normalize stage
        # (OCR noise, ADR-0005) — skip them silently. Events with
        # style_supported=False are routed to the Default group, the other
        # events are clustered by color within their position.
        min_duration_ms = config.min_event_duration_ms
        min_mean_conf = config.min_event_mean_confidence
        per_event: list[_PreparedEvent] = []
        for ev in animation.events:
            text = text_by_id.get(ev.event_id)
            if text is None:
                continue
            start_ms = frame_to_ms(ev.fansub_frame_start, globals.fps)
            end_ms = frame_to_ms(ev.fansub_frame_end, globals.fps)
            if (end_ms - start_ms) < min_duration_ms:
                continue
            confs = ev.raw_ocr_confidences
            if confs:
                mean_conf = sum(confs) / len(confs)
                if mean_conf < min_mean_conf:
                    continue
            ec = colors_by_id[ev.event_id]
            position = _classify_position(
                ev.quad_median, globals.fansub_width, globals.fansub_height
            )
            per_event.append(
                _PreparedEvent(event=ev, colors=ec, position=position, cleaned_text=text)
            )

        per_event = _merge_wrapped_lines(per_event, globals.fps)

        style_assignments = _synthesize_styles(
            per_event, threshold=config.color_cluster_threshold
        )

        subs = _build_ssa_file(
            per_event=per_event,
            style_assignments=style_assignments,
            globals=globals,
            config=config,
        )

        out_path = globals.out_path
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Force ASS format regardless of out_path suffix; encoding UTF-8 (no BOM).
        subs.save(str(tmp_path), format_="ass", encoding="utf-8")
        try:
            os.rename(tmp_path, out_path)
        except OSError:
            # Atomicity: out_path must not exist on failure. Clean up the .tmp
            # leftover so the workdir does not accumulate partial artefacts.
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

        return ExportResult(out_path_written=str(out_path), event_count=len(per_event))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_animation(workdir: Path) -> AnimationAnalysisResult:
    raw = (workdir / "08_animation" / "animation.json").read_text(encoding="utf-8")
    return AnimationAnalysisResult.model_validate_json(raw)


def _load_colors(workdir: Path) -> ColorExtractionResult:
    raw = (workdir / "09_color" / "colors.json").read_text(encoding="utf-8")
    return ColorExtractionResult.model_validate_json(raw)


def _load_doc(workdir: Path) -> NormalizeResult:
    raw = (workdir / "11_normalize" / "normalized.json").read_text(encoding="utf-8")
    return NormalizeResult.model_validate_json(raw)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _centroid(quad: list[tuple[int, int]]) -> tuple[float, float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


_POSITION_HCENTER_TOLERANCE_FRAC: float = 0.20  # ±20 % of width center
_POSITION_ROTATION_TOLERANCE_DEG: float = 2.0  # above this, classify as Sign


def _merge_wrapped_lines(
    events: list["_PreparedEvent"], fps: Fraction
) -> list["_PreparedEvent"]:
    """Merge OCR events that look like top/bottom halves of a wrapped subtitle.

    Long dialogue lines render on screen as two visual lines but fansub .ass
    files store them as a single string. PaddleOCR detects each visual line
    as a separate text box, which becomes two trajectories and two events
    here. Merging them into a single event with `\\N` between the texts both
    improves text_plain (each ref event gets the full text instead of half)
    and precision (one event matches instead of one matched + one unmatched).

    Merge criterion: two events overlap in time (≥ 80 %), are horizontally
    adjacent (x-centre within 25 % of frame width), and are stacked
    vertically with the closer pair touching within ~1.5 line heights.
    """
    if len(events) < 2:
        return events

    pairs = sorted(
        enumerate(events),
        key=lambda iev: iev[1].event.fansub_frame_start,
    )
    consumed: set[int] = set()
    merged: list[_PreparedEvent] = []
    by_idx = {i: pe for i, pe in pairs}

    def centroid(pe: _PreparedEvent) -> tuple[float, float]:
        return _centroid(pe.event.quad_median)

    def quad_height(pe: _PreparedEvent) -> float:
        ys = [y for _, y in pe.event.quad_median]
        return max(ys) - min(ys)

    for i, pe in pairs:
        if i in consumed:
            continue
        best_j: int | None = None
        best_gap = float("inf")
        s_a = pe.event.fansub_frame_start
        e_a = pe.event.fansub_frame_end
        cx_a, cy_a = centroid(pe)
        h_a = quad_height(pe)
        for j, qe in pairs:
            if j == i or j in consumed:
                continue
            s_b = qe.event.fansub_frame_start
            e_b = qe.event.fansub_frame_end
            overlap = max(0, min(e_a, e_b) - max(s_a, s_b))
            duration = max(e_a, e_b) - min(s_a, s_b)
            if duration == 0 or overlap / duration < 0.80:
                continue
            cx_b, cy_b = centroid(qe)
            if abs(cx_a - cx_b) > 0.30 * 640:  # play-res ≈ frame width
                continue
            gap = abs(cy_a - cy_b)
            h_b = quad_height(qe)
            # Stacked: gap roughly one line tall (between 0.5 × and 1.6 × the
            # taller line's height). Co-located events (same y) are not
            # wrapped pairs but duplicate detections — leave them alone.
            min_gap = 0.5 * max(h_a, h_b)
            max_gap = 1.6 * max(h_a, h_b)
            if not (min_gap <= gap <= max_gap):
                continue
            if gap < best_gap:
                best_gap = gap
                best_j = j
        if best_j is None:
            merged.append(pe)
            continue
        partner = by_idx[best_j]
        top, bottom = (pe, partner) if cy_a <= centroid(partner)[1] else (partner, pe)
        text = top.cleaned_text + "\n" + bottom.cleaned_text
        # Use the longer trajectory's event as the carrier (more reliable
        # quad_median / frame range).
        carrier_ev = top.event if len(top.event.member_frame_indices) >= len(
            bottom.event.member_frame_indices
        ) else bottom.event
        merged.append(
            _PreparedEvent(
                event=carrier_ev,
                colors=top.colors,
                position=top.position,
                cleaned_text=text,
            )
        )
        consumed.add(i)
        consumed.add(best_j)
    return merged


def _classify_position(
    quad: list[tuple[int, int]], width: int, height: int
) -> PositionClass:
    cx, cy = _centroid(quad)
    in_top_third = cy < height / 3
    in_bottom_third = cy > 2 * height / 3
    horizontally_centered = abs(cx - width / 2) <= _POSITION_HCENTER_TOLERANCE_FRAC * width
    # Dialogue OCR quads are essentially axis-aligned (sub-degree noise).
    # A clearly rotated quad signals a sign / overlay regardless of where the
    # centroid happens to fall.
    rotated = abs(_rotation_angle_deg(quad)) > _POSITION_ROTATION_TOLERANCE_DEG
    if rotated:
        return "Sign"
    if in_bottom_third and horizontally_centered:
        return "Bottom"
    if in_top_third and horizontally_centered:
        return "Top"
    return "Sign"


def _rotation_angle_deg(quad: list[tuple[int, int]]) -> float:
    # TL→TR edge orientation. ASS convention: clockwise positive. Image y axis
    # grows downward, atan2(dy, dx) already returns clockwise-positive in that
    # frame of reference, so no sign flip is needed.
    (x_tl, y_tl), (x_tr, y_tr) = quad[0], quad[1]
    return math.degrees(math.atan2(y_tr - y_tl, x_tr - x_tl))


# ---------------------------------------------------------------------------
# Color clustering — ΔE76 in CIE LAB
# ---------------------------------------------------------------------------


def _srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1 / 2.4)) - 0.055


def _rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    # sRGB → linear RGB → XYZ (D65) → LAB (D65)
    r, g, b = (_srgb_to_linear(v / 255.0) for v in rgb)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    # D65 reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    fx = _lab_f(x / xn)
    fy = _lab_f(y / yn)
    fz = _lab_f(z / zn)
    L = 116 * fy - 16
    a_ = 500 * (fx - fy)
    b_ = 200 * (fy - fz)
    return (L, a_, b_)


def _lab_f(t: float) -> float:
    delta = 6 / 29
    if t > delta**3:
        return t ** (1 / 3)
    return t / (3 * delta**2) + 4 / 29


def _lab_to_rgb(lab: tuple[float, float, float]) -> tuple[int, int, int]:
    L, a, b = lab
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    x = xn * _lab_f_inv(fx)
    y = yn * _lab_f_inv(fy)
    z = zn * _lab_f_inv(fz)
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b_ = x * 0.0556434 + y * -0.2040259 + z * 1.0572252
    r8 = _clip8(_linear_to_srgb(r))
    g8 = _clip8(_linear_to_srgb(g))
    b8 = _clip8(_linear_to_srgb(b_))
    return (r8, g8, b8)


def _lab_f_inv(t: float) -> float:
    delta = 6 / 29
    if t > delta:
        return t**3
    return 3 * delta**2 * (t - 4 / 29)


def _clip8(v: float) -> int:
    return max(0, min(255, int(round(v * 255))))


def _delta_e76(lab_a: tuple[float, float, float], lab_b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab_a, lab_b, strict=True)))


# ---------------------------------------------------------------------------
# Style synthesis
# ---------------------------------------------------------------------------


class _PreparedEvent:
    __slots__ = ("event", "colors", "position", "cleaned_text")

    def __init__(
        self,
        event: AnimatedEvent,
        colors: EventColors,
        position: PositionClass,
        cleaned_text: str,
    ) -> None:
        self.event = event
        self.colors = colors
        self.position = position
        self.cleaned_text = cleaned_text


class _Cluster:
    __slots__ = ("fill_labs", "outline_labs", "member_indices")

    def __init__(self) -> None:
        self.fill_labs: list[tuple[float, float, float]] = []
        self.outline_labs: list[tuple[float, float, float]] = []
        self.member_indices: list[int] = []

    def fill_centroid(self) -> tuple[float, float, float]:
        return _mean_lab(self.fill_labs)

    def outline_centroid(self) -> tuple[float, float, float]:
        return _mean_lab(self.outline_labs)

    def distance_to(
        self, fill_lab: tuple[float, float, float], outline_lab: tuple[float, float, float]
    ) -> float:
        return max(
            _delta_e76(self.fill_centroid(), fill_lab),
            _delta_e76(self.outline_centroid(), outline_lab),
        )

    def add(
        self,
        fill_lab: tuple[float, float, float],
        outline_lab: tuple[float, float, float],
        idx: int,
    ) -> None:
        self.fill_labs.append(fill_lab)
        self.outline_labs.append(outline_lab)
        self.member_indices.append(idx)


def _mean_lab(labs: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    n = len(labs)
    return (
        sum(c[0] for c in labs) / n,
        sum(c[1] for c in labs) / n,
        sum(c[2] for c in labs) / n,
    )


def _synthesize_styles(
    per_event: list[_PreparedEvent], *, threshold: float
) -> list[str]:
    """Return parallel list of style name per event in `per_event`."""
    style_per_idx: list[str | None] = [None] * len(per_event)

    # Default-group events are immediate.
    for i, pe in enumerate(per_event):
        if not pe.colors.style_supported:
            style_per_idx[i] = f"{pe.position}-Default"

    # Cluster supported events per position.
    for position in ("Bottom", "Top", "Sign"):
        supported = [
            i
            for i, pe in enumerate(per_event)
            if pe.position == position and pe.colors.style_supported
        ]
        if not supported:
            continue
        clusters: list[_Cluster] = []
        for i in supported:
            ec = per_event[i].colors
            assert ec.fill_color is not None and ec.outline_color is not None
            fill_lab = _rgb_to_lab(ec.fill_color)
            outline_lab = _rgb_to_lab(ec.outline_color)
            chosen: _Cluster | None = None
            best_distance = float("inf")
            for cluster in clusters:
                d = cluster.distance_to(fill_lab, outline_lab)
                if d < threshold and d < best_distance:
                    chosen = cluster
                    best_distance = d
            if chosen is None:
                chosen = _Cluster()
                clusters.append(chosen)
            chosen.add(fill_lab, outline_lab, i)

        # Rank clusters by descending member count to assign indices; idx 0 is
        # the dominant color cluster of the position.
        order = sorted(range(len(clusters)), key=lambda k: -len(clusters[k].member_indices))
        for rank, cluster_idx in enumerate(order):
            cluster = clusters[cluster_idx]
            style_name = f"{position}-{rank}"
            for member in cluster.member_indices:
                style_per_idx[member] = style_name

    result: list[str] = []
    for s in style_per_idx:
        assert s is not None
        result.append(s)
    return result


# ---------------------------------------------------------------------------
# SSA / ASS construction
# ---------------------------------------------------------------------------


def _build_ssa_file(
    *,
    per_event: list[_PreparedEvent],
    style_assignments: list[str],
    globals: PipelineGlobals,
    config: ExportConfig,
) -> SSAFile:
    subs = SSAFile()
    subs.info["Title"] = "Default Aegisub file"
    subs.info["ScriptType"] = "v4.00+"
    subs.info["PlayResX"] = str(globals.fansub_width)
    subs.info["PlayResY"] = str(globals.fansub_height)
    subs.info["WrapStyle"] = "0"
    subs.info["ScaledBorderAndShadow"] = "yes"

    # Synthesize styles dictionary.
    subs.styles = {}
    # Track which (position, color-cluster idx) -> canonical (fill, outline) RGB
    canonical_colors = _canonical_colors_by_style(per_event, style_assignments)
    used_styles = list(dict.fromkeys(style_assignments))  # stable, dedup
    for style_name in used_styles:
        if style_name.endswith("-Default"):
            subs.styles[style_name] = _make_default_style(config)
        else:
            fill_rgb, outline_rgb = canonical_colors[style_name]
            subs.styles[style_name] = _make_style(
                fill_rgb=fill_rgb,
                outline_rgb=outline_rgb,
                position=style_name.split("-", 1)[0],
                config=config,
            )

    for i, pe in enumerate(per_event):
        style_name = style_assignments[i]
        ev = pe.event
        start_ms = frame_to_ms(ev.fansub_frame_start, globals.fps)
        end_ms = frame_to_ms(ev.fansub_frame_end, globals.fps)

        text = pe.cleaned_text.replace("\n", "\\N")
        position_class: PositionClass = pe.position  # type: ignore[assignment]

        # Inline tag order: position + rotation + animation + (colors NEVER inline)
        tag_parts: list[str] = []
        if position_class == "Sign":
            cx, cy = _centroid(ev.quad_median)
            tag_parts.append(f"\\pos({int(round(cx))},{int(round(cy))})")
            # Rotation: only emitted on Sign-class events (mid-screen overlays
            # like in-frame signs and rotated annotations). Regular dialogue
            # OCR quads have sub-degree rotation noise that would create
            # styling-component mismatches. ASS spec: `\frzNUMBER` (no parens).
            angle = _rotation_angle_deg(ev.quad_median)
            if abs(angle) >= _FRZ_OMIT_THRESHOLD_DEG:
                tag_parts.append(f"\\frz{_format_float(angle)}")

        motion = ev.motion
        if motion is not None and motion.get("type") == "linear":
            x1, y1 = motion["start"]
            x2, y2 = motion["end"]
            tag_parts.append(f"\\move({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

        # Animation-detected fades are noisy on real material; we trust them
        # only when explicitly enabled via config.
        if config.emit_animation_fades and (
            ev.fade_in_ms > 0 or ev.fade_out_ms > 0
        ):
            tag_parts.append(f"\\fad({ev.fade_in_ms},{ev.fade_out_ms})")

        if tag_parts:
            inline = "{" + "".join(tag_parts) + "}"
            text = inline + text

        if motion is not None and motion.get("type") == "nonlinear_flagged":
            text = _NONLINEAR_COMMENT + text

        subs.append(
            SSAEvent(
                start=start_ms,
                end=end_ms,
                text=text,
                style=style_name,
            )
        )
    return subs


def _canonical_colors_by_style(
    per_event: list[_PreparedEvent], style_assignments: list[str]
) -> dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """For each non-default style, mean of member fills/outlines in LAB → RGB."""
    members: dict[str, list[int]] = {}
    for i, name in enumerate(style_assignments):
        if name.endswith("-Default"):
            continue
        members.setdefault(name, []).append(i)
    result: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {}
    for name, idxs in members.items():
        fills_lab = [_rgb_to_lab(per_event[i].colors.fill_color) for i in idxs]  # type: ignore[arg-type]
        outlines_lab = [_rgb_to_lab(per_event[i].colors.outline_color) for i in idxs]  # type: ignore[arg-type]
        fill_rgb = _lab_to_rgb(_mean_lab(fills_lab))
        outline_rgb = _lab_to_rgb(_mean_lab(outlines_lab))
        result[name] = (fill_rgb, outline_rgb)
    return result


def _alignment_for(position: str) -> Alignment:
    if position == "Top":
        return Alignment.TOP_CENTER
    if position == "Sign":
        # \pos overrides alignment but pick something neutral (center).
        return Alignment.MIDDLE_CENTER
    return Alignment.BOTTOM_CENTER


def _make_style(
    *,
    fill_rgb: tuple[int, int, int],
    outline_rgb: tuple[int, int, int],
    position: str,
    config: ExportConfig,
) -> SSAStyle:
    return SSAStyle(
        fontname=config.default_font,
        fontsize=float(config.default_font_size),
        primarycolor=Color(fill_rgb[0], fill_rgb[1], fill_rgb[2], 0),
        outlinecolor=Color(outline_rgb[0], outline_rgb[1], outline_rgb[2], 0),
        outline=2.0,
        alignment=_alignment_for(position),
    )


def _make_default_style(config: ExportConfig) -> SSAStyle:
    return SSAStyle(
        fontname=config.default_font,
        fontsize=float(config.default_font_size),
        primarycolor=Color(255, 255, 255, 0),
        outlinecolor=Color(0, 0, 0, 0),
        outline=2.0,
        alignment=Alignment.BOTTOM_CENTER,
    )


def _format_float(x: float) -> str:
    # ASS accepts plain numeric tokens; trim trailing zeros for readability.
    s = f"{x:.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


__all__ = [
    "ExportConfig",
    "ExportResult",
    "ExportStage",
    "STAGE_VERSION",
    "pysubs2",
]
