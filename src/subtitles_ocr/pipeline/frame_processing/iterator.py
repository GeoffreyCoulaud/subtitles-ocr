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
    """Default frame reader, backed by PyAV (libavcodec) for broad codec
    support (AV1, HEVC, …) that opencv's bundled ffmpeg may lack. Cached per
    path with a sequential fast path and re-seek on jumps. The class keeps
    its historical name for API stability with existing imports."""

    def __init__(self) -> None:
        # path -> (av container, video stream, frame iterator, next_idx)
        self._state: dict[Path, dict] = {}

    def _open(self, path: Path) -> dict:
        import av

        container = av.open(str(path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        state = {
            "container": container,
            "stream": stream,
            "iter": container.decode(stream),
            "next_idx": 0,
        }
        self._state[path] = state
        return state

    def _seek(self, path: Path, frame_idx: int) -> None:
        state = self._state[path]
        stream = state["stream"]
        target_pts = int(frame_idx / stream.average_rate / stream.time_base)
        state["container"].seek(target_pts, any_frame=False, backward=True, stream=stream)
        state["iter"] = state["container"].decode(stream)
        state["next_idx"] = -1

    def read(self, path: Path, frame_idx: int) -> np.ndarray:
        if path not in self._state:
            self._open(path)
        state = self._state[path]
        if frame_idx < state["next_idx"] or frame_idx > state["next_idx"] + 256:
            self._seek(path, frame_idx)
        stream = state["stream"]
        target_pts = int(frame_idx / stream.average_rate / stream.time_base)
        for frame in state["iter"]:
            if frame.pts is None:
                continue
            if frame.pts < target_pts:
                continue
            state["next_idx"] = frame_idx + 1
            return frame.to_ndarray(format="rgb24")
        raise RuntimeError(f"PyAV reached EOF before frame {frame_idx} in {path}")


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
    # ConformStage writes the AR/codec-normalized raw to `01_conform/raw.mkv`;
    # diff/mask require fansub and raw at the same resolution, so we must read
    # the conformed copy, not `globals.raw_path` (which still points at the
    # source bluray).
    conformed_raw_path = workdir / "01_conform" / "raw.mkv"
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
            raw_img = reader.read(conformed_raw_path, raw_idx)
            diff = compute_diff(fansub_img, raw_img, config)
            mask = make_mask(diff, config)
            composed = compose(fansub_img, mask)
            if debug:
                stem = f"{fansub_idx:08d}.png"
                _write_debug_image(workdir / "03_diff" / "debug" / stem, diff)
                _write_debug_image(workdir / "04_mask" / "frames" / stem, mask)
                _write_debug_image(workdir / "05_compose" / "frames" / stem, composed)
            yield ComposedFrame(fansub_frame_idx=fansub_idx, image=composed)
