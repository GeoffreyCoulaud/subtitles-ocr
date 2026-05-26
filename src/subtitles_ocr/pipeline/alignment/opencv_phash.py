"""PyAV-backed default `FrameSource` for AlignmentStage phash.

Uses libavcodec (PyAV) for decoding so AV1 / HEVC / etc. all work, regardless
of OpenCV's bundled ffmpeg capabilities. PHash is still computed via
``cv2.img_hash.PHash``.
"""

from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np


class OpenCvPhashFrameSource:
    def __init__(self, fansub_path: Path, raw_path: Path) -> None:
        self._paths = {"fansub": Path(fansub_path), "raw": Path(raw_path)}
        self._containers: dict[str, av.container.InputContainer] = {}
        self._streams: dict[str, av.video.stream.VideoStream] = {}
        self._iters: dict[str, object] = {}
        self._next_idx: dict[str, int] = {}
        self._hasher = cv2.img_hash.PHash_create()

    def _open(self, source: str) -> None:
        container = av.open(str(self._paths[source]))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        self._containers[source] = container
        self._streams[source] = stream
        self._iters[source] = container.decode(stream)
        self._next_idx[source] = 0

    def _seek(self, source: str, frame_idx: int) -> None:
        stream = self._streams[source]
        # seek by PTS in stream time-base; we want frame_idx → timestamp
        # average_rate gives Fraction frames/sec
        rate = stream.average_rate
        pts = int(frame_idx / rate / stream.time_base)
        self._containers[source].seek(pts, any_frame=False, backward=True, stream=stream)
        self._iters[source] = self._containers[source].decode(stream)
        self._next_idx[source] = -1  # unknown until we advance

    def get_phash(self, source: str, frame_idx: int) -> int:
        if source not in self._containers:
            self._open(source)
        # if requested frame is in the past or far ahead, seek
        if frame_idx < self._next_idx[source] or frame_idx > self._next_idx[source] + 64:
            self._seek(source, frame_idx)

        decoded = None
        # advance until we land on or past the desired index, then identify by counting
        # PyAV doesn't surface canonical frame indices for all formats, so we use
        # a counting strategy after seek: count frames decoded from this point.
        iterator = self._iters[source]
        # decoder yields frames in DTS order after seek; pick the first whose
        # frame timestamp matches frame_idx
        rate = self._streams[source].average_rate
        target_pts = int(frame_idx / rate / self._streams[source].time_base)
        for frame in iterator:
            if frame.pts is None:
                continue
            if frame.pts < target_pts:
                continue
            decoded = frame
            break
        if decoded is None:
            raise RuntimeError(
                f"OpenCvPhashFrameSource: decode reached EOF source={source} idx={frame_idx} path={self._paths[source]}"
            )
        self._next_idx[source] = frame_idx + 1

        rgb = decoded.to_ndarray(format="bgr24")
        hash_bytes = self._hasher.compute(rgb)
        return int.from_bytes(np.asarray(hash_bytes).tobytes(), byteorder="little", signed=False)
