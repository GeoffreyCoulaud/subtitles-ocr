"""Stage 9 — per-event color extraction (ADR-0002 §3 Stage 8, ADR-0003 §4.3)."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Protocol

import cv2
import numpy as np
from pydantic import BaseModel
from scipy.ndimage import distance_transform_edt

from subtitles_ocr.config import ColorConfig, PipelineGlobals
from subtitles_ocr.meta import BaseMeta, cache_invalidating_dict
from subtitles_ocr.pipeline.animation import AnimationAnalysisResult
from subtitles_ocr.timing import ms_to_frame

STAGE_VERSION: int = 1

_STAGE_NAME = "09_color"


class EventColors(BaseModel):
    event_id: int
    fill_color: tuple[int, int, int] | None
    outline_color: tuple[int, int, int] | None
    style_supported: bool
    stroke_width_px: float


class ColorExtractionResult(BaseModel):
    events: list[EventColors]
    stats: dict


class FrameReader(Protocol):
    """Reads a single decoded RGB uint8 frame from a video at a given index.

    Production impl will wrap ffmpeg / OpenCV; tests inject in-memory fakes.
    """

    def read(self, path: Path, frame_idx: int) -> np.ndarray: ...


class ColorStage:
    CONFIG_FIELD: ClassVar[str] = "color"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = (
        "workdir",
        "fps",
        "hardsub_path",
    )

    def __init__(self, frame_reader: FrameReader | None = None) -> None:
        self.frame_reader = frame_reader

    def run(self, globals: PipelineGlobals, config: ColorConfig) -> ColorExtractionResult:
        workdir = globals.workdir
        out_path = workdir / _STAGE_NAME / "colors.json"
        meta_path = workdir / _STAGE_NAME / "colors.meta.json"
        animation_path = workdir / "08_animation" / "animation.json"

        candidate_meta = self._build_meta(globals, config, animation_path)
        if out_path.exists() and meta_path.exists():
            try:
                persisted = BaseMeta.model_validate_json(meta_path.read_text())
            except ValueError:
                persisted = None
            if persisted is not None and persisted.matches(candidate_meta):
                return ColorExtractionResult.model_validate_json(out_path.read_text())

        if self.frame_reader is None:
            raise RuntimeError(
                "ColorStage requires a FrameReader to compute fresh results; "
                "none was injected."
            )

        animation = AnimationAnalysisResult.model_validate_json(
            animation_path.read_text()
        )

        events_out: list[EventColors] = []
        n_supported = 0
        for event in animation.events:
            ec = self._process_event(event, globals, config)
            if ec.style_supported:
                n_supported += 1
            events_out.append(ec)

        result = ColorExtractionResult(
            events=events_out,
            stats={"supported": n_supported, "total": len(events_out)},
        )
        self._atomic_write(out_path, result.model_dump_json())
        self._atomic_write(meta_path, candidate_meta.model_dump_json())
        return result

    # ---------------- per-event ----------------

    def _process_event(
        self,
        event,
        globals: PipelineGlobals,
        config: ColorConfig,
    ) -> EventColors:
        # Exclude fade-in / fade-out frames by absolute frame index.
        # Stage 7 gap tolerance may produce sparse member_frame_indices, so
        # filtering by list position would drop the wrong frames.
        fade_in_frames = ms_to_frame(event.fade_in_ms, globals.fps)
        fade_out_frames = ms_to_frame(event.fade_out_ms, globals.fps)
        start = event.fansub_frame_start
        end = event.fansub_frame_end
        members = [
            m for m in event.member_frame_indices
            if start + fade_in_frames <= m < end - fade_out_frames
        ]

        if len(members) < 3:
            return EventColors(
                event_id=event.event_id,
                fill_color=None,
                outline_color=None,
                style_supported=False,
                stroke_width_px=0.0,
            )

        # Determine canonical canvas size from median of per-frame quad sizes.
        per_frame_quads = event.quads_per_frame
        widths: list[float] = []
        heights: list[float] = []
        for fidx in members:
            quad = per_frame_quads.get(fidx, event.quad_median)
            w, h = _quad_wh(quad)
            widths.append(w)
            heights.append(h)
        W_canon = max(8, int(round(float(np.median(widths)))))
        H_canon = max(8, int(round(float(np.median(heights)))))

        warped: list[np.ndarray] = []
        for fidx in members:
            frame = self.frame_reader.read(globals.hardsub_path, fidx)
            quad = per_frame_quads.get(fidx, event.quad_median)
            padded_quad = _expand_quad(quad, config.crop_padding_pct)
            warp = _rectify(frame, padded_quad, W_canon, H_canon)
            warped.append(warp)

        stack = np.stack(warped, axis=0)
        median_img = np.median(stack, axis=0).astype(np.uint8)

        gray = cv2.cvtColor(median_img, cv2.COLOR_RGB2GRAY)
        # Otsu global on the canonical crop. THRESH_BINARY: glyph (bright) → 255.
        _, glyph_mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        if int(glyph_mask.sum()) == 0:
            return EventColors(
                event_id=event.event_id,
                fill_color=None,
                outline_color=None,
                style_supported=False,
                stroke_width_px=0.0,
            )

        bool_mask = glyph_mask > 0
        # distance_transform_edt returns the distance from True to nearest False.
        distances = distance_transform_edt(bool_mask)
        glyph_dists = distances[bool_mask]
        stroke_width = float(np.percentile(glyph_dists, 95))

        radius = int(math.floor(config.erosion_factor * stroke_width))
        if radius < 1:
            radius = 1
        kernel_size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # Interior pool = deep core of the glyph mask (purely fill pixels).
        eroded = cv2.erode(glyph_mask, kernel, iterations=1)
        interior_mask = eroded > 0
        # Outline pool = annular band just *outside* the glyph mask, captured by
        # dilation minus the original mask. This is the band where the outline
        # color physically lives in the typical hardsub style (fill brighter
        # than bg; outline darker than fill but possibly close to bg). Using the
        # OUTER band rather than the inner ring of the fill mask is the only way
        # to recover the outline color when fill and outline are on opposite
        # sides of Otsu's split (the dominant real case).
        dilated = cv2.dilate(glyph_mask, kernel, iterations=1)
        outline_mask = (dilated > 0) & ~bool_mask

        interior_pool_size = int(interior_mask.sum())
        outline_pool_size = int(outline_mask.sum())

        if interior_pool_size < config.interior_min_pixels:
            return EventColors(
                event_id=event.event_id,
                fill_color=None,
                outline_color=None,
                style_supported=False,
                stroke_width_px=stroke_width,
            )

        if outline_pool_size == 0:
            return EventColors(
                event_id=event.event_id,
                fill_color=None,
                outline_color=None,
                style_supported=False,
                stroke_width_px=stroke_width,
            )

        ratio = interior_pool_size / outline_pool_size
        if ratio < config.pool_ratio_min or ratio > config.pool_ratio_max:
            return EventColors(
                event_id=event.event_id,
                fill_color=None,
                outline_color=None,
                style_supported=False,
                stroke_width_px=stroke_width,
            )

        # Convert to HSV (uint8 OpenCV scale: H in [0,179], S/V in [0,255]).
        hsv_img = cv2.cvtColor(median_img, cv2.COLOR_RGB2HSV)

        interior_pixels = hsv_img[interior_mask]
        outline_pixels = hsv_img[outline_mask]

        interior_h_var = _circular_hue_variance(interior_pixels[:, 0])
        outline_h_var = _circular_hue_variance(outline_pixels[:, 0])

        if interior_h_var > config.interior_hue_var_max:
            return EventColors(
                event_id=event.event_id,
                fill_color=None,
                outline_color=None,
                style_supported=False,
                stroke_width_px=stroke_width,
            )

        if outline_h_var > config.outline_hue_var_max:
            return EventColors(
                event_id=event.event_id,
                fill_color=None,
                outline_color=None,
                style_supported=False,
                stroke_width_px=stroke_width,
            )

        fill_color = _hsv_mode_to_rgb(interior_pixels, bins=config.hsv_bins)
        outline_color = _hsv_mode_to_rgb(outline_pixels, bins=config.hsv_bins)

        return EventColors(
            event_id=event.event_id,
            fill_color=fill_color,
            outline_color=outline_color,
            style_supported=True,
            stroke_width_px=stroke_width,
        )

    # ---------------- sidecar / atomic write ----------------

    def _build_meta(
        self,
        globals: PipelineGlobals,
        config: ColorConfig,
        animation_path: Path,
    ) -> BaseMeta:
        globals_subset = {
            "fps": f"{globals.fps.numerator}/{globals.fps.denominator}",
            "hardsub_path": str(globals.hardsub_path),
        }
        input_fps = {}
        if animation_path.exists():
            from subtitles_ocr.meta import fingerprint

            input_fps["animation"] = fingerprint(
                animation_path, treat_as_intermediate=True
            )
        return BaseMeta(
            stage_name=_STAGE_NAME,
            stage_version=STAGE_VERSION,
            config=cache_invalidating_dict(config),
            globals_subset=globals_subset,
            input_fingerprints=input_fps,
            written_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


# ---------------- geometry helpers ----------------


def _quad_wh(quad: list[tuple[int, int]]) -> tuple[float, float]:
    """Width = avg(TL-TR, BL-BR) length; height = avg(TL-BL, TR-BR) length."""
    pts = np.asarray(quad, dtype=np.float32)
    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return float(w), float(h)


def _expand_quad(
    quad: list[tuple[int, int]], padding_pct: float
) -> list[tuple[float, float]]:
    """Expand the quad outward by `padding_pct * diagonal` along each vertex's
    direction from the centroid.
    """
    pts = np.asarray(quad, dtype=np.float32)
    centroid = pts.mean(axis=0)
    # Diagonal: average of TL-BR and TR-BL
    diag = (np.linalg.norm(pts[2] - pts[0]) + np.linalg.norm(pts[3] - pts[1])) / 2.0
    pad = padding_pct * diag

    expanded: list[tuple[float, float]] = []
    for p in pts:
        direction = p - centroid
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            expanded.append((float(p[0]), float(p[1])))
            continue
        unit = direction / norm
        new_p = p + unit * pad
        expanded.append((float(new_p[0]), float(new_p[1])))
    return expanded


def _rectify(
    img: np.ndarray,
    quad: list[tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    src = np.asarray(quad, dtype=np.float32)
    dst = np.asarray(
        [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
    )


# ---------------- color clustering helpers ----------------


def _circular_hue_variance(hues: np.ndarray) -> float:
    """Circular variance on OpenCV hue (0..179). Returns 1 - R where R is the
    mean resultant length. 0 = perfectly concentrated, 1 = fully dispersed.
    """
    if hues.size == 0:
        return 1.0
    angles = (hues.astype(np.float64) / 180.0) * 2.0 * math.pi
    s = np.sin(angles).mean()
    c = np.cos(angles).mean()
    r = math.sqrt(s * s + c * c)
    return 1.0 - r


def _hsv_mode_to_rgb(
    hsv_pixels: np.ndarray, bins: int
) -> tuple[int, int, int]:
    """Mode in HSV with `bins`-bucket quantization, re-converted to RGB."""
    if hsv_pixels.shape[0] == 0:
        return (0, 0, 0)
    # Quantize each channel into `bins` buckets.
    h_bin_size = 180 / bins
    sv_bin_size = 256 / bins
    h_idx = np.minimum((hsv_pixels[:, 0] / h_bin_size).astype(np.int32), bins - 1)
    s_idx = np.minimum((hsv_pixels[:, 1] / sv_bin_size).astype(np.int32), bins - 1)
    v_idx = np.minimum((hsv_pixels[:, 2] / sv_bin_size).astype(np.int32), bins - 1)
    keys = (h_idx * bins * bins) + (s_idx * bins) + v_idx
    uniq, counts = np.unique(keys, return_counts=True)
    winner = uniq[int(np.argmax(counts))]
    h_w = winner // (bins * bins)
    rest = winner % (bins * bins)
    s_w = rest // bins
    v_w = rest % bins
    # Take bin centers and convert one pixel back to RGB.
    h_val = int(round((h_w + 0.5) * h_bin_size))
    h_val = min(h_val, 179)
    s_val = int(round((s_w + 0.5) * sv_bin_size))
    s_val = min(s_val, 255)
    v_val = int(round((v_w + 0.5) * sv_bin_size))
    v_val = min(v_val, 255)
    one = np.array([[[h_val, s_val, v_val]]], dtype=np.uint8)
    rgb = cv2.cvtColor(one, cv2.COLOR_HSV2RGB)[0, 0]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
