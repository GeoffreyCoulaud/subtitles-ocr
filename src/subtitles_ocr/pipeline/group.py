"""Stage 7 — per-quad trajectory grouping (ADR-0002 §3 Stage 7, ADR-0003 §4.1)."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median_low
from typing import ClassVar

from pydantic import BaseModel
from rapidfuzz.distance import Levenshtein

from subtitles_ocr.config import GroupConfig, PipelineGlobals
from subtitles_ocr.io import JsonlWriter
from subtitles_ocr.meta import BaseMeta, cache_invalidating_dict, fingerprint
from subtitles_ocr.pipeline.alignment.stage import AlignmentResult
from subtitles_ocr.pipeline.ocr import FrameOcrResult, OcrDetection

logger = logging.getLogger(__name__)

STAGE_VERSION: int = 2

_STAGE_NAME = "07_group"


class SubtitleEvent(BaseModel):
    event_id: int
    fansub_frame_start: int
    fansub_frame_end: int
    raw_ocr_texts: list[str]
    raw_ocr_confidences: list[float]
    # ADR-0003 §4.1: per-frame quads required for animation analysis.
    # Pydantic v2 serializes int keys as strings in JSON and converts back at validation time.
    quads_per_frame: dict[int, list[tuple[int, int]]]
    quad_median: list[tuple[int, int]]
    member_frame_indices: list[int]


class GroupResult(BaseModel):
    fansub_total_frames: int
    events: list[SubtitleEvent]
    stats: dict


# ----------------------------- internals -----------------------------


@dataclass
class _Trajectory:
    """Mutable in-flight trajectory. Finalized into a SubtitleEvent at break time.

    `stale_frames` counts consecutive ALIGNED frames since the last match. The
    trajectory is finalized when this exceeds `max_gap_frames`; the event's
    end_exclusive is then `last_matched_frame + 1` (the gap frames are not
    part of the event's lifetime).
    """

    start_frame: int
    last_matched_frame: int
    stale_frames: int = 0
    members: list[tuple[int, OcrDetection]] = field(default_factory=list)

    def extend(self, frame_idx: int, det: OcrDetection) -> None:
        self.members.append((frame_idx, det))
        self.last_matched_frame = frame_idx
        self.stale_frames = 0

    def last_detection(self) -> OcrDetection:
        return self.members[-1][1]

    def last_frame(self) -> int:
        return self.members[-1][0]


def _normalized_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _sutherland_hodgman(subject: list[tuple[float, float]], clip: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Convex polygon clipping. Both polygons assumed convex with consistent winding."""
    # Ensure clip is counter-clockwise so the "inside" half-plane test is uniform.
    output = list(subject)
    if _signed_area(clip) < 0:
        clip = list(reversed(clip))
    if _signed_area(output) < 0:
        output = list(reversed(output))

    n_clip = len(clip)
    for i in range(n_clip):
        if not output:
            break
        a = clip[i]
        b = clip[(i + 1) % n_clip]
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if _is_inside(e, a, b):
                if not _is_inside(s, a, b):
                    output.append(_segment_intersection(s, e, a, b))
                output.append(e)
            elif _is_inside(s, a, b):
                output.append(_segment_intersection(s, e, a, b))
            s = e
    return output


def _signed_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _is_inside(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> bool:
    # Left side of directed edge a→b (CCW winding ⇒ "inside")
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0


def _segment_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return p2  # parallel; fall back to endpoint, intersection set is empty in practice
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _quad_iou(qa: list[tuple[int, int]], qb: list[tuple[int, int]]) -> float:
    a = [(float(x), float(y)) for x, y in qa]
    b = [(float(x), float(y)) for x, y in qb]
    area_a = _polygon_area(a)
    area_b = _polygon_area(b)
    if area_a == 0.0 or area_b == 0.0:
        return 0.0
    inter = _sutherland_hodgman(a, b)
    area_inter = _polygon_area(inter)
    union = area_a + area_b - area_inter
    if union <= 0.0:
        return 0.0
    return area_inter / union


def _match_continuation(
    traj: _Trajectory,
    det: OcrDetection,
    config: GroupConfig,
) -> tuple[bool, float]:
    """Return (matches, score). Score = IoU; only meaningful when matches=True."""
    last = traj.last_detection()
    lev = _normalized_levenshtein(last.text, det.text)
    if lev >= config.text_levenshtein_max:
        return False, 0.0
    iou = _quad_iou(last.quad, det.quad)
    if iou <= config.quad_iou_min:
        return False, 0.0
    return True, iou


def _finalize(traj: _Trajectory, event_id: int, end_exclusive: int) -> SubtitleEvent:
    frames = [f for f, _ in traj.members]
    texts = [d.text for _, d in traj.members]
    confs = [d.confidence for _, d in traj.members]
    quads_per_frame = {f: list(d.quad) for f, d in traj.members}
    quad_median = _per_coord_median([d.quad for _, d in traj.members])
    return SubtitleEvent(
        event_id=event_id,
        fansub_frame_start=traj.start_frame,
        fansub_frame_end=end_exclusive,
        raw_ocr_texts=texts,
        raw_ocr_confidences=confs,
        quads_per_frame=quads_per_frame,
        quad_median=quad_median,
        member_frame_indices=frames,
    )


def _per_coord_median(quads: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    # Each quad has 4 vertices; median per vertex per coordinate.
    out: list[tuple[int, int]] = []
    for i in range(4):
        xs = [q[i][0] for q in quads]
        ys = [q[i][1] for q in quads]
        out.append((median_low(xs), median_low(ys)))
    return out


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _build_meta(
    globs: PipelineGlobals,
    config: GroupConfig,
    ocr_path: Path,
    alignment_path: Path,
) -> BaseMeta:
    return BaseMeta(
        stage_name=_STAGE_NAME,
        stage_version=STAGE_VERSION,
        config=cache_invalidating_dict(config),
        globals_subset={k: getattr(globs, k) for k in GroupStage.GLOBALS_USED if k != "workdir"},
        input_fingerprints={
            "ocr_results": fingerprint(ocr_path, treat_as_intermediate=True),
            "alignment": fingerprint(alignment_path, treat_as_intermediate=True),
        },
        written_at=datetime.now(timezone.utc),
    )


# ----------------------------- stage -----------------------------


class GroupStage:
    CONFIG_FIELD: ClassVar[str] = "group"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = ("workdir", "fansub_total_frames")

    def __init__(self) -> None:
        pass

    def run(self, globals: PipelineGlobals, config: GroupConfig) -> GroupResult:
        workdir = globals.workdir
        out_dir = workdir / "07_group"
        events_path = out_dir / "events.json"
        meta_path = out_dir / "events.meta.json"
        ocr_path = workdir / "06_ocr" / "results.jsonl"
        alignment_path = workdir / "02_alignment" / "alignment.json"

        candidate_meta = _build_meta(globals, config, ocr_path, alignment_path)

        # Resume: if events.json + matching sidecar already exist, return cached result.
        if events_path.exists() and meta_path.exists():
            try:
                persisted_meta = BaseMeta.model_validate_json(
                    meta_path.read_text(encoding="utf-8")
                )
            except ValueError:
                persisted_meta = None
            if persisted_meta is not None and persisted_meta.matches(candidate_meta):
                logger.info("cache hit; reusing %s", events_path)
                return GroupResult.model_validate_json(
                    events_path.read_text(encoding="utf-8")
                )

        alignment = AlignmentResult.model_validate_json(
            alignment_path.read_text(encoding="utf-8")
        )
        status_by_frame = self._status_by_frame(alignment)
        ocr_by_frame = self._read_ocr(ocr_path)

        events = self._group(globals, config, status_by_frame, ocr_by_frame)

        result = GroupResult(
            fansub_total_frames=globals.fansub_total_frames,
            events=events,
            stats={
                "events": len(events),
                "aligned_frames": sum(1 for s in status_by_frame.values() if s == "ALIGNED"),
            },
        )

        _atomic_write_text(events_path, result.model_dump_json())
        _atomic_write_text(meta_path, candidate_meta.model_dump_json())
        return result

    @staticmethod
    def _status_by_frame(alignment: AlignmentResult) -> dict[int, str]:
        out: dict[int, str] = {}
        for seg in alignment.segments:
            for idx in range(seg.fansub_frame_start, seg.fansub_frame_end):
                out[idx] = seg.status
        return out

    @staticmethod
    def _read_ocr(path: Path) -> dict[int, FrameOcrResult]:
        if not path.exists():
            return {}
        out: dict[int, FrameOcrResult] = {}
        with JsonlWriter(path, FrameOcrResult) as reader:
            for item in reader.iter_persisted():
                out[item.fansub_frame_idx] = item
        return out

    @staticmethod
    def _group(
        globs: PipelineGlobals,
        config: GroupConfig,
        status_by_frame: dict[int, str],
        ocr_by_frame: dict[int, FrameOcrResult],
    ) -> list[SubtitleEvent]:
        active: list[_Trajectory] = []
        finalized: list[SubtitleEvent] = []
        next_event_id = 0

        def finalize_traj(traj: _Trajectory) -> None:
            nonlocal next_event_id
            finalized.append(
                _finalize(traj, next_event_id, traj.last_matched_frame + 1)
            )
            next_event_id += 1

        for idx in range(globs.fansub_total_frames):
            status = status_by_frame.get(idx, "ORPHAN")

            if status != "ALIGNED":
                # ORPHAN / USER_SKIPPED: trajectories carry over untouched.
                # Stale counter is NOT incremented (non-ALIGNED frames are
                # neutral per design — they neither match nor break trajectories).
                continue

            frame_result = ocr_by_frame.get(idx)
            detections = list(frame_result.detections) if frame_result is not None else []

            # Greedy 1-1 matching: for each active trajectory, pick the best
            # unassigned detection (highest IoU among those satisfying both
            # Levenshtein and IoU thresholds). Deterministic across input order.
            assigned_det_idx: set[int] = set()
            extended_traj_indices: set[int] = set()

            for t_i, traj in enumerate(active):
                best_d: int | None = None
                best_score = -1.0
                for d_i, det in enumerate(detections):
                    if d_i in assigned_det_idx:
                        continue
                    ok, score = _match_continuation(traj, det, config)
                    if ok and score > best_score:
                        best_score = score
                        best_d = d_i
                if best_d is not None:
                    traj.extend(idx, detections[best_d])
                    assigned_det_idx.add(best_d)
                    extended_traj_indices.add(t_i)

            # Non-extended active trajectories: increment stale; finalize only
            # if the gap budget is exceeded.
            survivors: list[_Trajectory] = []
            for t_i, traj in enumerate(active):
                if t_i in extended_traj_indices:
                    survivors.append(traj)
                    continue
                traj.stale_frames += 1
                if traj.stale_frames > config.max_gap_frames:
                    finalize_traj(traj)
                else:
                    survivors.append(traj)
            active = survivors

            # Unmatched detections → start new trajectories.
            for d_i, det in enumerate(detections):
                if d_i in assigned_det_idx:
                    continue
                new_traj = _Trajectory(start_frame=idx, last_matched_frame=idx)
                new_traj.extend(idx, det)
                active.append(new_traj)

        # Flush remaining trajectories at the video end: each ends at its own
        # last_matched_frame + 1, NOT at fansub_total_frames.
        for traj in active:
            finalize_traj(traj)
        return finalized
