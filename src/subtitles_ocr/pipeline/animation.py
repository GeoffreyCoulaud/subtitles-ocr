"""Stage 8 — animation analysis (`\\move` + `\\fad`), ADR-0003 §4.2.

Two sub-stages applied in order:

* **A1 (intra-event move)** — linear regression on centroid trajectories.
* **A2 (inter-event merge)** — fragmented chains stitched when gap + text +
  trajectory R² all agree.
* **B (fade)** — diff-intensity scoring in pre/post search windows, anchored
  linear fit, extrapolation to ``score=0``.

Fade detection requires a ``diff_source`` (Protocol exposing
``mean_intensity(frame_idx, bbox) -> float``). When absent, fades are silently
treated as 0 — useful for tests/integration where no real frames exist.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel
from rapidfuzz.distance import Levenshtein

from subtitles_ocr.config import AnimationConfig, PipelineGlobals
from subtitles_ocr.meta import (
    BaseMeta,
    FileFingerprint,
    cache_invalidating_dict,
    fingerprint,
)
from subtitles_ocr.pipeline.group import GroupResult, SubtitleEvent
from subtitles_ocr.timing import ms_to_frame

logger = logging.getLogger(__name__)

STAGE_VERSION: int = 3

STAGE_NAME = "08_animation"


# ---------------------------------------------------------------------------
# Public schemas
# ---------------------------------------------------------------------------


class AnimatedEvent(BaseModel):
    event_id: int
    fansub_frame_start: int
    fansub_frame_end: int
    raw_ocr_texts: list[str]
    raw_ocr_confidences: list[float]
    quads_per_frame: dict[int, list[tuple[int, int]]]
    quad_median: list[tuple[int, int]]
    member_frame_indices: list[int]
    # motion = {"type": "linear", "start": (x, y), "end": (x, y)}
    #        | {"type": "nonlinear_flagged"}
    #        | None  (static)
    motion: dict | None
    fade_in_ms: int
    fade_out_ms: int


class AnimationAnalysisResult(BaseModel):
    events: list[AnimatedEvent]
    stats: dict


class DiffIntensitySource(Protocol):
    """Provides mean diff-intensity (LCN+Sobel, see ADR-0002 §9) within bbox.

    Bbox is ``(x_min, y_min, x_max, y_max)`` in fansub pixel coordinates,
    half-open (max exclusive)."""

    def mean_intensity(
        self, frame_idx: int, bbox: tuple[int, int, int, int]
    ) -> float: ...


_EMPTY_STATS: dict[str, int] = {
    "static": 0,
    "linear_move": 0,
    "flagged_nonlinear": 0,
    "fade_in_only": 0,
    "fade_out_only": 0,
    "full_fade": 0,
}


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class AnimationStage:
    CONFIG_FIELD: ClassVar[str] = "animation"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = (
        "workdir",
        "fps",
        "fansub_total_frames",
    )

    def __init__(self, diff_source: DiffIntensitySource | None = None) -> None:
        self.diff_source = diff_source

    def run(
        self, globals: PipelineGlobals, config: AnimationConfig
    ) -> AnimationAnalysisResult:
        # Lazy auto-load: in production the OCR stage writes a per-frame diff
        # sidecar at 06_ocr/diff_grid.npz. If no caller injected a diff source
        # and that sidecar exists, wire a PersistedDiffSource so fade detection
        # runs. Tests that pre-inject a diff source (or that omit the sidecar)
        # are unaffected.
        if self.diff_source is None:
            diff_sidecar = globals.workdir / "06_ocr" / "diff_grid.npz"
            if diff_sidecar.exists():
                from subtitles_ocr.pipeline.frame_processing.diff_intensity import (
                    PersistedDiffSource,
                )

                self.diff_source = PersistedDiffSource(diff_sidecar)

        out_dir = globals.workdir / STAGE_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "animation.json"
        meta_path = out_dir / "animation.meta.json"

        group_path = globals.workdir / "07_group" / "events.json"
        expected_meta = self._build_meta(globals, config, group_path)

        if out_path.exists() and meta_path.exists():
            try:
                cached_meta = BaseMeta.model_validate_json(meta_path.read_text())
            except ValueError:
                cached_meta = None
            expected_for_compare = BaseMeta.model_validate_json(
                expected_meta.model_dump_json()
            )
            if cached_meta is not None and cached_meta.matches(expected_for_compare):
                logger.info("animation stage: cache hit, skipping recompute")
                return AnimationAnalysisResult.model_validate_json(out_path.read_text())

        group_result = GroupResult.model_validate_json(group_path.read_text())
        events = list(group_result.events)

        events = [_intra_event_move(ev, config) for ev in events]
        events = _merge_inter_events(events, config, globals)
        animated = _detect_fades(
            events,
            config,
            globals,
            self.diff_source,
            group_result.fansub_total_frames,
        )

        stats = _summarize(animated)
        result = AnimationAnalysisResult(events=animated, stats=stats)

        _atomic_write_text(out_path, result.model_dump_json())
        _atomic_write_text(meta_path, expected_meta.model_dump_json())
        logger.info(
            "animation stage: wrote %d events (stats=%s)",
            len(result.events),
            result.stats,
        )
        return result

    def _build_meta(
        self,
        globals: PipelineGlobals,
        config: AnimationConfig,
        group_path: Path,
    ) -> BaseMeta:
        globals_subset = {
            "fps": f"{globals.fps.numerator}/{globals.fps.denominator}",
            "fansub_total_frames": globals.fansub_total_frames,
            "workdir": str(globals.workdir),
        }
        inputs: dict[str, FileFingerprint] = {}
        if group_path.exists():
            inputs["group_events"] = fingerprint(group_path, treat_as_intermediate=True)
        return BaseMeta(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config=cache_invalidating_dict(config),
            globals_subset=globals_subset,
            input_fingerprints=inputs,
            written_at=datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# Working dataclass-ish state (mutable between sub-stages)
# ---------------------------------------------------------------------------


class _WorkingEvent:
    """Mutable carry between sub-stages A1 / A2 / B."""

    __slots__ = (
        "event_id",
        "fansub_frame_start",
        "fansub_frame_end",
        "raw_ocr_texts",
        "raw_ocr_confidences",
        "quads_per_frame",
        "quad_median",
        "member_frame_indices",
        "motion",
        "fade_in_ms",
        "fade_out_ms",
    )

    def __init__(
        self,
        *,
        event_id: int,
        fansub_frame_start: int,
        fansub_frame_end: int,
        raw_ocr_texts: list[str],
        raw_ocr_confidences: list[float],
        quads_per_frame: dict[int, list[tuple[int, int]]],
        quad_median: list[tuple[int, int]],
        member_frame_indices: list[int],
        motion: dict | None = None,
        fade_in_ms: int = 0,
        fade_out_ms: int = 0,
    ) -> None:
        self.event_id = event_id
        self.fansub_frame_start = fansub_frame_start
        self.fansub_frame_end = fansub_frame_end
        self.raw_ocr_texts = raw_ocr_texts
        self.raw_ocr_confidences = raw_ocr_confidences
        self.quads_per_frame = quads_per_frame
        self.quad_median = quad_median
        self.member_frame_indices = member_frame_indices
        self.motion = motion
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms

    @classmethod
    def from_subtitle_event(cls, ev: SubtitleEvent) -> "_WorkingEvent":
        return cls(
            event_id=ev.event_id,
            fansub_frame_start=ev.fansub_frame_start,
            fansub_frame_end=ev.fansub_frame_end,
            raw_ocr_texts=list(ev.raw_ocr_texts),
            raw_ocr_confidences=list(ev.raw_ocr_confidences),
            quads_per_frame=dict(ev.quads_per_frame),
            quad_median=list(ev.quad_median),
            member_frame_indices=list(ev.member_frame_indices),
        )

    def to_animated_event(self) -> AnimatedEvent:
        return AnimatedEvent(
            event_id=self.event_id,
            fansub_frame_start=self.fansub_frame_start,
            fansub_frame_end=self.fansub_frame_end,
            raw_ocr_texts=self.raw_ocr_texts,
            raw_ocr_confidences=self.raw_ocr_confidences,
            quads_per_frame=self.quads_per_frame,
            quad_median=self.quad_median,
            member_frame_indices=self.member_frame_indices,
            motion=self.motion,
            fade_in_ms=self.fade_in_ms,
            fade_out_ms=self.fade_out_ms,
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _centroid(quad: list[tuple[int, int]]) -> tuple[float, float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _bbox_of(quad: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Ordinary least squares ``y = a*x + b``; returns (a, b, R²).

    For constant ``y`` we return R²=1 (any line through the mean fits exactly).
    For len < 2 returns (0, mean, 1.0).
    """
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0, 1.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    if den_x == 0.0:
        return 0.0, mean_y, 1.0
    slope = num / den_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    if ss_tot == 0.0:
        return slope, intercept, 1.0
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot
    return slope, intercept, r2


def _linfit_through_anchor(
    xs: list[float],
    ys: list[float],
    *,
    anchor_x: float,
    anchor_y: float,
) -> tuple[float, float]:
    """Constrained linear fit passing through ``(anchor_x, anchor_y)``.

    Returns (slope, r2). Solves min Σ(y_i - (a·(x_i - anchor_x) + anchor_y))².
    """
    if not xs:
        return 0.0, 0.0
    num = sum((x - anchor_x) * (y - anchor_y) for x, y in zip(xs, ys))
    den = sum((x - anchor_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0, 0.0
    slope = num / den
    mean_y = sum(ys) / len(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    if ss_tot == 0.0:
        # All ys equal — anchor consistency determines R².
        return slope, 1.0 if abs(ys[0] - anchor_y) < 1e-12 else 0.0
    ss_res = sum(
        (y - (slope * (x - anchor_x) + anchor_y)) ** 2 for x, y in zip(xs, ys)
    )
    return slope, 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# A1 — intra-event move
# ---------------------------------------------------------------------------


def _intra_event_move(ev: SubtitleEvent, config: AnimationConfig) -> _WorkingEvent:
    we = _WorkingEvent.from_subtitle_event(ev)
    frames = sorted(we.quads_per_frame.keys())
    if len(frames) < 2:
        return we

    centroids = [_centroid(we.quads_per_frame[f]) for f in frames]
    xs = [float(f) for f in frames]
    cxs = [c[0] for c in centroids]
    cys = [c[1] for c in centroids]

    slope_x, intercept_x, r2_x = _linfit(xs, cxs)
    slope_y, intercept_y, r2_y = _linfit(xs, cys)

    start_x = slope_x * xs[0] + intercept_x
    end_x = slope_x * xs[-1] + intercept_x
    start_y = slope_y * xs[0] + intercept_y
    end_y = slope_y * xs[-1] + intercept_y
    displacement = math.hypot(end_x - start_x, end_y - start_y)

    if displacement < config.min_move_displacement_px:
        return we

    if r2_x >= config.move_r2_threshold and r2_y >= config.move_r2_threshold:
        we.motion = {
            "type": "linear",
            "start": (int(round(start_x)), int(round(start_y))),
            "end": (int(round(end_x)), int(round(end_y))),
        }
    else:
        we.motion = {"type": "nonlinear_flagged"}
    return we


# ---------------------------------------------------------------------------
# A2 — inter-event merge
# ---------------------------------------------------------------------------


def _normalized_lev(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


def _best_text(texts: list[str], confs: list[float]) -> str:
    if not texts:
        return ""
    best_idx = max(range(len(texts)), key=lambda i: confs[i] if i < len(confs) else 0.0)
    return texts[best_idx]


def _merge_inter_events(
    events: list[_WorkingEvent],
    config: AnimationConfig,
    globals: PipelineGlobals,
) -> list[_WorkingEvent]:
    if len(events) < 2:
        return events

    gap_tol_frames = ms_to_frame(config.move_gap_tolerance_ms, globals.fps)

    # Pass: walk events and greedily extend a chain when criteria hold.
    out: list[_WorkingEvent] = []
    i = 0
    while i < len(events):
        chain = [events[i]]
        j = i + 1
        while j < len(events):
            gap = events[j].fansub_frame_start - chain[-1].fansub_frame_end
            if gap > gap_tol_frames or gap < 0:
                break
            text_a = _best_text(chain[-1].raw_ocr_texts, chain[-1].raw_ocr_confidences)
            text_b = _best_text(events[j].raw_ocr_texts, events[j].raw_ocr_confidences)
            if _normalized_lev(text_a, text_b) >= config.move_text_levenshtein_max:
                break
            chain.append(events[j])
            j += 1

        if len(chain) == 1:
            out.append(events[i])
            i += 1
            continue

        # Fit combined trajectory and decide linear vs flagged
        combined_frames: list[int] = []
        combined_xs: list[float] = []
        combined_ys: list[float] = []
        for ev in chain:
            for f in sorted(ev.quads_per_frame.keys()):
                cx, cy = _centroid(ev.quads_per_frame[f])
                combined_frames.append(f)
                combined_xs.append(cx)
                combined_ys.append(cy)

        if len(combined_frames) >= 2:
            xs = [float(f) for f in combined_frames]
            slope_x, intercept_x, r2_x = _linfit(xs, combined_xs)
            slope_y, intercept_y, r2_y = _linfit(xs, combined_ys)
            linear_ok = (
                r2_x >= config.move_r2_threshold
                and r2_y >= config.move_r2_threshold
            )
        else:
            slope_x = slope_y = intercept_x = intercept_y = 0.0
            linear_ok = False

        merged = _merge_chain(chain)
        if len(combined_frames) >= 2 and linear_ok:
            start_x = slope_x * xs[0] + intercept_x
            end_x = slope_x * xs[-1] + intercept_x
            start_y = slope_y * xs[0] + intercept_y
            end_y = slope_y * xs[-1] + intercept_y
            merged.motion = {
                "type": "linear",
                "start": (int(round(start_x)), int(round(start_y))),
                "end": (int(round(end_x)), int(round(end_y))),
            }
        else:
            merged.motion = {"type": "nonlinear_flagged"}
        out.append(merged)
        i = j

    return out


def _merge_chain(chain: list[_WorkingEvent]) -> _WorkingEvent:
    first = chain[0]
    quads_per_frame: dict[int, list[tuple[int, int]]] = {}
    texts: list[str] = []
    confs: list[float] = []
    member_frames: list[int] = []
    all_quads: list[list[tuple[int, int]]] = []
    for ev in chain:
        for f in sorted(ev.quads_per_frame.keys()):
            quads_per_frame[f] = ev.quads_per_frame[f]
            all_quads.append(ev.quads_per_frame[f])
        texts.extend(ev.raw_ocr_texts)
        confs.extend(ev.raw_ocr_confidences)
        member_frames.extend(ev.member_frame_indices)
    return _WorkingEvent(
        event_id=first.event_id,
        fansub_frame_start=first.fansub_frame_start,
        fansub_frame_end=chain[-1].fansub_frame_end,
        raw_ocr_texts=texts,
        raw_ocr_confidences=confs,
        quads_per_frame=quads_per_frame,
        quad_median=_quad_per_coord_median(all_quads) if all_quads else first.quad_median,
        member_frame_indices=member_frames,
    )


def _quad_per_coord_median(
    quads: list[list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    from statistics import median_low

    out: list[tuple[int, int]] = []
    for i in range(4):
        xs = [q[i][0] for q in quads]
        ys = [q[i][1] for q in quads]
        out.append((median_low(xs), median_low(ys)))
    return out


# ---------------------------------------------------------------------------
# B — fade detection
# ---------------------------------------------------------------------------


def _bbox_at_frame_for_event(
    we: _WorkingEvent, frame_idx: int
) -> tuple[int, int, int, int]:
    motion = we.motion
    if motion is not None and motion.get("type") == "linear":
        # Extrapolate centroid linearly using motion.start / motion.end and
        # keep the quad geometry of the closest-known frame (start of event).
        known_frames = sorted(we.quads_per_frame.keys())
        if not known_frames:
            return _bbox_of(we.quad_median)
        ref_frame = known_frames[0]
        ref_quad = we.quads_per_frame[ref_frame]
        ref_cx, ref_cy = _centroid(ref_quad)
        sx, sy = motion["start"]
        ex, ey = motion["end"]
        n_start, n_end = known_frames[0], known_frames[-1]
        if n_end == n_start:
            cx = float(sx)
            cy = float(sy)
        else:
            t = (frame_idx - n_start) / (n_end - n_start)
            cx = sx + t * (ex - sx)
            cy = sy + t * (ey - sy)
        dx = cx - ref_cx
        dy = cy - ref_cy
        shifted = [(int(round(x + dx)), int(round(y + dy))) for x, y in ref_quad]
        return _bbox_of(shifted)
    return _bbox_of(we.quad_median)


def _occupied_frames(events: list[_WorkingEvent]) -> dict[int, int]:
    """Map frame_idx → owning event_id (only frames inside [start, end))."""
    out: dict[int, int] = {}
    for ev in events:
        for f in range(ev.fansub_frame_start, ev.fansub_frame_end):
            out[f] = ev.event_id
    return out


def _detect_fades(
    events: list[_WorkingEvent],
    config: AnimationConfig,
    globals: PipelineGlobals,
    source: DiffIntensitySource | None,
    total_frames: int,
) -> list[AnimatedEvent]:
    if source is None:
        logger.info(
            "animation stage: no diff_source provided, fade detection skipped"
        )
        return [ev.to_animated_event() for ev in events]

    window_frames = max(1, ms_to_frame(config.fade_search_window_ms, globals.fps))
    min_frames = max(1, math.ceil(
        config.min_fade_duration_ms / (1000.0 / float(globals.fps))
    ))
    occupied = _occupied_frames(events)

    for ev in events:
        if ev.motion is not None and ev.motion.get("type") == "nonlinear_flagged":
            continue
        if ev.fansub_frame_end <= ev.fansub_frame_start:
            continue

        # Anchor scoring: score(K) normalized by mean diff intensity at the
        # event-start frame (using its bbox).
        start_bbox = _bbox_at_frame_for_event(ev, ev.fansub_frame_start)
        anchor_intensity = source.mean_intensity(ev.fansub_frame_start, start_bbox)
        if anchor_intensity <= 0:
            continue

        # Fade-in side: pre-window frames in [start - W, start)
        pre_frames = [
            f
            for f in range(
                max(0, ev.fansub_frame_start - window_frames), ev.fansub_frame_start
            )
            if occupied.get(f, ev.event_id) == ev.event_id
        ]
        t_in_ms = _fit_fade(
            ev,
            pre_frames,
            anchor_intensity,
            anchor_frame=ev.fansub_frame_start,
            config=config,
            globals=globals,
            source=source,
            min_frames=min_frames,
            side="in",
        )

        # Fade-out side: post-window frames in [end, end + W)
        post_frames = [
            f
            for f in range(
                ev.fansub_frame_end,
                min(total_frames, ev.fansub_frame_end + window_frames),
            )
            if occupied.get(f, ev.event_id) == ev.event_id
        ]
        t_out_ms = _fit_fade(
            ev,
            post_frames,
            anchor_intensity,
            anchor_frame=ev.fansub_frame_end - 1,
            config=config,
            globals=globals,
            source=source,
            min_frames=min_frames,
            side="out",
        )

        ev.fade_in_ms = t_in_ms
        ev.fade_out_ms = t_out_ms

        # Consistency: t_in + t_out must fit within (extended) event duration
        duration_ms = int(
            round(
                (ev.fansub_frame_end - ev.fansub_frame_start)
                * 1000.0
                / float(globals.fps)
            )
        )
        if ev.fade_in_ms + ev.fade_out_ms > duration_ms:
            logger.warning(
                "event %d: partial fade (in=%dms + out=%dms > duration=%dms); reverting",
                ev.event_id,
                ev.fade_in_ms,
                ev.fade_out_ms,
                duration_ms,
            )
            ev.fade_in_ms = 0
            ev.fade_out_ms = 0
            continue

        # Extend boundaries to cover the fade ramps
        if ev.fade_in_ms > 0:
            extend = ms_to_frame(ev.fade_in_ms, globals.fps)
            ev.fansub_frame_start = max(0, ev.fansub_frame_start - extend)
        if ev.fade_out_ms > 0:
            extend = ms_to_frame(ev.fade_out_ms, globals.fps)
            ev.fansub_frame_end = min(total_frames, ev.fansub_frame_end + extend)

    return [ev.to_animated_event() for ev in events]


def _fit_fade(
    ev: _WorkingEvent,
    frames: list[int],
    anchor_intensity: float,
    *,
    anchor_frame: int,
    config: AnimationConfig,
    globals: PipelineGlobals,
    source: DiffIntensitySource,
    min_frames: int,
    side: str,
) -> int:
    score_lo, score_hi = config.fade_score_fit_range
    obs_xs: list[float] = []
    obs_ys: list[float] = []
    for f in frames:
        bbox = _bbox_at_frame_for_event(ev, f)
        s = source.mean_intensity(f, bbox) / anchor_intensity
        if score_lo <= s <= score_hi:
            obs_xs.append(float(f))
            obs_ys.append(s)

    if len(obs_xs) < min_frames:
        return 0

    slope, r2 = _linfit_through_anchor(
        obs_xs, obs_ys, anchor_x=float(anchor_frame), anchor_y=1.0
    )
    if r2 < config.fade_fit_r2_threshold:
        logger.warning(
            "event %d (%s-fade): R²=%.3f < %.2f; reverting",
            ev.event_id,
            side,
            r2,
            config.fade_fit_r2_threshold,
        )
        return 0
    if slope == 0:
        return 0
    # Extrapolate y = slope*(x - anchor_x) + 1 → 0 at x = anchor_x - 1/slope
    fade_origin = -1.0 / slope  # frames offset from anchor where score=0
    t_frames = abs(fade_origin)
    fps_f = float(globals.fps)
    t_ms = int(round(t_frames * 1000.0 / fps_f))

    if t_ms < config.min_fade_duration_ms:
        return 0
    if t_ms > config.fade_duration_cap_ms:
        logger.warning(
            "event %d (%s-fade): extrapolated %dms > cap %dms; reverting",
            ev.event_id,
            side,
            t_ms,
            config.fade_duration_cap_ms,
        )
        return 0
    return t_ms


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _summarize(events: list[AnimatedEvent]) -> dict[str, int]:
    stats = dict(_EMPTY_STATS)
    for ev in events:
        has_fade_in = ev.fade_in_ms > 0
        has_fade_out = ev.fade_out_ms > 0
        if has_fade_in and has_fade_out:
            stats["full_fade"] += 1
        elif has_fade_in:
            stats["fade_in_only"] += 1
        elif has_fade_out:
            stats["fade_out_only"] += 1

        if ev.motion is None:
            stats["static"] += 1
        elif ev.motion.get("type") == "linear":
            stats["linear_move"] += 1
        elif ev.motion.get("type") == "nonlinear_flagged":
            stats["flagged_nonlinear"] += 1
    return stats


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
