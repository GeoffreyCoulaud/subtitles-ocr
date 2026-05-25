"""Stage 1 — spatial conform (ADR-0002 §3 Stage 1)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ValidationError

from subtitles_ocr.config import ConformConfig, PipelineGlobals
from subtitles_ocr.exceptions import AspectRatioMismatch
from subtitles_ocr.ffmpeg.protocol import FfmpegRunner, TranscodeArgs, VideoMetadata
from subtitles_ocr.ffmpeg.subprocess_runner import SubprocessFfmpegRunner
from subtitles_ocr.meta import BaseMeta, cache_invalidating_dict, fingerprint

STAGE_VERSION: int = 1

STAGE_NAME = "01_conform"
OUTPUT_FILENAME = "raw.mkv"
SIDECAR_FILENAME = "raw.meta.json"

logger = logging.getLogger(__name__)


class ConformResult(BaseModel):
    raw_conformed_path: Path
    target_width: int
    target_height: int
    pix_fmt: str


def _aspect_ratio(width: int, height: int) -> float:
    return width / height


def _ar_close(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) <= tol * max(a, b)


def _build_filter_chain(target_width: int, target_height: int) -> str:
    # ADR-0002 §3 Stage 1:
    #   - scale=W:H:flags=area  (area resampling → no ringing on text edges)
    #   - format=yuv420p        (truncation, no dither — uncorrelated noise would
    #                            degrade the downstream diff)
    return f"scale={target_width}:{target_height}:flags=area,format=yuv420p"


class ConformStage:
    CONFIG_FIELD: ClassVar[str] = "conform"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = (
        "workdir",
        "hardsub_path",
        "raw_path",
        "fansub_width",
        "fansub_height",
    )

    def __init__(self, ffmpeg: FfmpegRunner | None = None) -> None:
        self.ffmpeg: FfmpegRunner = ffmpeg if ffmpeg is not None else SubprocessFfmpegRunner()

    def run(self, globals: PipelineGlobals, config: ConformConfig) -> ConformResult:
        out_dir = globals.workdir / STAGE_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / OUTPUT_FILENAME
        sidecar_path = out_dir / SIDECAR_FILENAME

        fansub_meta = self.ffmpeg.probe(globals.hardsub_path)
        raw_meta = self.ffmpeg.probe(globals.raw_path)

        ar_fansub = _aspect_ratio(fansub_meta.width, fansub_meta.height)
        ar_raw = _aspect_ratio(raw_meta.width, raw_meta.height)
        if not _ar_close(ar_fansub, ar_raw):
            if config.ar_strategy == "error":
                raise AspectRatioMismatch(
                    f"fansub AR {ar_fansub:.4f} ({fansub_meta.width}x{fansub_meta.height}) "
                    f"does not match raw AR {ar_raw:.4f} ({raw_meta.width}x{raw_meta.height})",
                    stage=STAGE_NAME,
                    hint=(
                        "Pass --ar-strategy letterbox or --ar-strategy crop to accept "
                        "the mismatch, or verify the source pair."
                    ),
                )
            # letterbox/crop strategies deferred — not in MVP scope per ADR-0002.
            raise NotImplementedError(
                f"ar_strategy={config.ar_strategy!r} not implemented yet"
            )

        target_width = globals.fansub_width
        target_height = globals.fansub_height
        filter_chain = _build_filter_chain(target_width, target_height)

        candidate_meta = self._build_meta(
            globals=globals,
            config=config,
            fansub_meta=fansub_meta,
            raw_meta=raw_meta,
        )

        if out_path.exists() and sidecar_path.exists():
            persisted = self._read_sidecar(sidecar_path)
            if persisted is not None and persisted.matches(candidate_meta):
                logger.info("conform cache hit; skipping transcode")
                return ConformResult(
                    raw_conformed_path=out_path,
                    target_width=target_width,
                    target_height=target_height,
                    pix_fmt="yuv420p",
                )

        args = TranscodeArgs(
            input_path=globals.raw_path,
            output_path=out_path,
            filter_chain=filter_chain,
            target_width=target_width,
            target_height=target_height,
            target_pix_fmt="yuv420p",
        )
        self.ffmpeg.transcode(args)

        # Persist sidecar AFTER transcode so a crash leaves a stale-or-missing
        # sidecar (cache miss next run), never a sidecar pointing at half-written
        # output.
        sidecar_path.write_text(candidate_meta.model_dump_json())

        return ConformResult(
            raw_conformed_path=out_path,
            target_width=target_width,
            target_height=target_height,
            pix_fmt="yuv420p",
        )

    @staticmethod
    def _read_sidecar(path: Path) -> BaseMeta | None:
        try:
            return BaseMeta.model_validate_json(path.read_text())
        except (ValidationError, ValueError):
            return None

    @staticmethod
    def _build_meta(
        *,
        globals: PipelineGlobals,
        config: ConformConfig,
        fansub_meta: VideoMetadata,
        raw_meta: VideoMetadata,
    ) -> BaseMeta:
        return BaseMeta(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config=cache_invalidating_dict(config),
            globals_subset={
                "fansub_width": globals.fansub_width,
                "fansub_height": globals.fansub_height,
            },
            input_fingerprints={
                "hardsub": fingerprint(globals.hardsub_path),
                "raw": fingerprint(globals.raw_path),
            },
            written_at=datetime.now(timezone.utc),
        )
