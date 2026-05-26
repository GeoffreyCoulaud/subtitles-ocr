"""Stages 3-5 — diff + mask + compose, exposed as a streaming iterator.

`ComposedFrame` is a runtime-only frozen dataclass (never persisted), per
ADR-0004 §3.2. Pydantic is reserved for persisted boundaries.

Frame reading uses dependency injection via the `FrameReader` Protocol so
tests never touch real video files (per ADR-0004 §13.5 — no monkeypatching of
external deps). The default implementation uses OpenCV (cv2.VideoCapture);
PyAV is acceptable as a future swap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from subtitles_ocr.config import FrameProcessingConfig, PipelineGlobals
from subtitles_ocr.pipeline.alignment.stage import AlignmentResult
from subtitles_ocr.pipeline.frame_processing.compose import compose
from subtitles_ocr.pipeline.frame_processing.diff import compute_diff
from subtitles_ocr.pipeline.frame_processing.mask import make_mask

STAGE_VERSION: int = 1


@dataclass(frozen=True)
class ComposedFrame:
    fansub_frame_idx: int
    image: np.ndarray  # RGB uint8, H×W×3


class FrameReader(Protocol):
    def read(self, path: Path, frame_idx: int) -> np.ndarray: ...


class _OpenCvFrameReader:
    """Cached sequential reader. Re-opens on backward seeks to stay simple."""

    def __init__(self) -> None:
        self._caps: dict[Path, tuple[object, int]] = {}

    def _get(self, path: Path) -> tuple[object, int]:
        if path not in self._caps:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV failed to open video: {path}")
            self._caps[path] = (cap, -1)
        return self._caps[path]

    def read(self, path: Path, frame_idx: int) -> np.ndarray:
        cap, last_idx = self._get(path)
        if frame_idx != last_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # type: ignore[attr-defined]
        ok, bgr = cap.read()  # type: ignore[attr-defined]
        if not ok or bgr is None:
            raise RuntimeError(f"OpenCV failed to read frame {frame_idx} from {path}")
        self._caps[path] = (cap, frame_idx)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_debug_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        # Scale float32 diff to uint8 for visualization.
        if image.dtype != np.uint8:
            vmax = float(np.max(image)) or 1.0
            image = (np.clip(image / vmax, 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(str(path), image)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)


def iter_composed_frames(
    globals: PipelineGlobals,
    alignment_result: AlignmentResult,
    config: FrameProcessingConfig,
    start_at_fansub_idx: int = 0,
    *,
    frame_reader: FrameReader | None = None,
) -> Iterator[ComposedFrame]:
    reader = frame_reader if frame_reader is not None else _OpenCvFrameReader()
    debug = globals.debug_images
    workdir = globals.workdir
    for seg in alignment_result.segments:
        if seg.status != "ALIGNED":
            continue
        if seg.raw_frame_start is None or seg.offset_frames is None:
            continue
        for fansub_idx in range(seg.fansub_frame_start, seg.fansub_frame_end):
            if fansub_idx < start_at_fansub_idx:
                continue
            raw_idx = fansub_idx + seg.offset_frames
            fansub_img = reader.read(globals.hardsub_path, fansub_idx)
            raw_img = reader.read(globals.raw_path, raw_idx)
            diff = compute_diff(fansub_img, raw_img, config)
            mask = make_mask(diff, config)
            composed = compose(fansub_img, mask)
            if debug:
                stem = f"{fansub_idx:08d}.png"
                _write_debug_image(workdir / "03_diff" / "debug" / stem, diff)
                _write_debug_image(workdir / "04_mask" / "frames" / stem, mask)
                _write_debug_image(workdir / "05_compose" / "frames" / stem, composed)
            yield ComposedFrame(fansub_frame_idx=fansub_idx, image=composed)
