"""Fakes and fixtures for cross-stage integration tests.

Per ADR-0004 §13.4 every external dep (ffmpeg, OCR engine, LLM, frame reader)
is fronted by a Protocol and tests inject a fake via the stage constructor.
No monkeypatching anywhere.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from subtitles_ocr.config import PipelineGlobals
from subtitles_ocr.ffmpeg.protocol import TranscodeArgs, VideoMetadata
from subtitles_ocr.pipeline.ocr import OcrDetection

__all__ = [
    "FakeFfmpeg",
    "FakeFrameReader",
    "FakeFrameSource",
    "FakeLlm",
    "FakeOcrEngine",
    "make_video_metadata",
]


# ---------------------------------------------------------------------------
# Fake ffmpeg runner (Conform + Alignment)
# ---------------------------------------------------------------------------


@dataclass
class FakeFfmpeg:
    """In-memory FfmpegRunner. Records calls; produces stub output files.

    The conformed output file is a deterministic 5-byte payload so re-runs that
    invalidate do not perturb downstream fingerprints in surprising ways.
    """

    probe_returns: dict[Path, VideoMetadata] = field(default_factory=dict)
    transcode_calls: list[TranscodeArgs] = field(default_factory=list)
    probe_calls: list[Path] = field(default_factory=list)
    extract_calls: list[tuple[Path, int, Path]] = field(default_factory=list)

    def probe(self, path: Path) -> VideoMetadata:
        self.probe_calls.append(path)
        return self.probe_returns[path]

    def transcode(self, args: TranscodeArgs) -> None:
        self.transcode_calls.append(args)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_bytes(b"FAKE_CONFORMED_MKV")

    def extract_audio(self, path: Path, track_index: int, out: Path) -> None:
        self.extract_calls.append((path, track_index, out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"FAKE_WAV")


# ---------------------------------------------------------------------------
# Fake phash frame source (Alignment phash-only branch)
# ---------------------------------------------------------------------------


@dataclass
class FakeFrameSource:
    """Returns a deterministic phash per (source, frame_idx).

    To produce a perfect 1:1 alignment (no orphans) we return the same hash for
    `("fansub", i)` and `("raw", i)` so phash_fallback resolves offset=0 with
    distance=0 for every frame.
    """

    def get_phash(self, source: str, frame_idx: int) -> int:
        # Identical hashes between sources at the same idx → offset 0, distance 0.
        return (frame_idx * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Fake OCR engine
# ---------------------------------------------------------------------------


@dataclass
class FakeOcrEngine:
    """OcrEngine returning a fixed detection per frame.

    Tracks call count to assert cache hit/miss across stage re-runs.
    """

    text: str = "hello"
    confidence: float = 0.9
    quad: tuple = ((100, 800), (300, 800), (300, 850), (100, 850))
    call_count: int = 0

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        self.call_count += 1
        return [
            OcrDetection(
                text=self.text,
                confidence=self.confidence,
                quad=[(int(x), int(y)) for x, y in self.quad],
            )
        ]


# ---------------------------------------------------------------------------
# Fake LLM client
# ---------------------------------------------------------------------------


@dataclass
class FakeLlm:
    """LlmClient that returns a constructed BaseModel of the requested schema.

    Tracks call count; responses are produced by a callable so we can return
    either CleanedEvent or DocCleanupResult depending on what the stage asked
    for, without needing two separate fakes.
    """

    response_factory: object = None  # callable(schema, prompt) -> BaseModel
    call_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def complete(self, prompt: str, response_schema: type[BaseModel], *, model: str):
        with self._lock:
            self.call_count += 1
        assert callable(self.response_factory), "response_factory must be a callable"
        return self.response_factory(response_schema, prompt)


# ---------------------------------------------------------------------------
# Fake frame reader (Color stage)
# ---------------------------------------------------------------------------


@dataclass
class FakeFrameReader:
    """FrameReader serving a deterministic synthetic frame for any idx.

    The frame is a small white-text-on-dark-bg image inside the canonical quad
    region so ColorStage finds the supported style branch.
    """

    width: int = 1920
    height: int = 1080
    call_count: int = 0
    _cached_frame: np.ndarray | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _build_frame(self) -> np.ndarray:
        import cv2

        img = np.full((self.height, self.width, 3), 16, dtype=np.uint8)
        # Glyph drawn in the quad region used by FakeOcrEngine ((100, 800) - (300, 850))
        org = (110, 840)
        cv2.putText(img, "Hi", org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, "Hi", org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def read(self, path: Path, frame_idx: int) -> np.ndarray:
        with self._lock:
            self.call_count += 1
            if self._cached_frame is None:
                self._cached_frame = self._build_frame()
            return self._cached_frame


# ---------------------------------------------------------------------------
# Workspace + globals fixtures specific to integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def real_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Real on-disk hardsub/raw files so Conform can fingerprint them."""
    hardsub = tmp_path / "fansub.mkv"
    raw = tmp_path / "raw.mkv"
    hardsub.write_bytes(b"FAKE_HARDSUB_SOURCE_BYTES")
    raw.write_bytes(b"FAKE_RAW_SOURCE_BYTES")
    return hardsub, raw


@pytest.fixture
def integration_workdir(tmp_path: Path) -> Path:
    wd = tmp_path / "workdir"
    wd.mkdir(parents=True, exist_ok=True)
    return wd


@pytest.fixture
def integration_globals(
    integration_workdir: Path, real_inputs: tuple[Path, Path]
) -> PipelineGlobals:
    hardsub, raw = real_inputs
    return PipelineGlobals(
        workdir=integration_workdir,
        hardsub_path=hardsub,
        raw_path=raw,
        out_path=integration_workdir / "out.ass",
        fps=Fraction(24, 1),
        fansub_width=1920,
        fansub_height=1080,
        fansub_total_frames=5,
        debug_images=False,
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline runner used by every integration test
# ---------------------------------------------------------------------------


def make_video_metadata(width: int = 1920, height: int = 1080) -> VideoMetadata:
    return VideoMetadata(
        width=width,
        height=height,
        fps_num=24,
        fps_den=1,
        total_frames=5,
        duration_s=5 / 24,
        pix_fmt="yuv420p",
        colorspace="bt709",
    )
