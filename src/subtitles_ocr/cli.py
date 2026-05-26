from __future__ import annotations

import argparse
import logging
import shlex
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from subtitles_ocr.config import (
    FrameProcessingConfig,
    PipelineConfig,
    PipelineGlobals,
    section_for,
)
from subtitles_ocr.exceptions import PipelineError
from subtitles_ocr.ffmpeg.protocol import FfmpegRunner
from subtitles_ocr.ffmpeg.subprocess_runner import SubprocessFfmpegRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="subtitles-ocr")

    p.add_argument("--hardsub", required=True, type=Path)
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workdir", required=True, type=Path)

    p.add_argument("--language", default="latin")
    p.add_argument("--synopsis", type=Path, default=None)
    p.add_argument("--debug-images", action="store_true")
    p.add_argument(
        "--ar-strategy",
        choices=["error", "letterbox", "crop"],
        default="error",
    )
    p.add_argument("--hardsub-audio-track", type=int, default=None)
    p.add_argument("--raw-audio-track", type=int, default=None)
    p.add_argument("--hardsub-skip", action="append", default=None)
    p.add_argument("--raw-skip", action="append", default=None)
    p.add_argument(
        "--ocr-device",
        choices=["auto", "cuda", "rocm", "cpu"],
        default="auto",
    )
    p.add_argument("--event-cleanup-model", default=None)
    p.add_argument("--event-cleanup-parallelism", type=int, default=1)
    p.add_argument("--doc-cleanup-model", default=None)
    p.add_argument("--doc-cleanup-parallelism", type=int, default=1)
    p.add_argument("--color-cluster-threshold", type=float, default=10.0)
    p.add_argument("--debug", action="store_true")

    return p


def _build_globals(ns: argparse.Namespace, ffmpeg: FfmpegRunner) -> PipelineGlobals:
    """Probe the hardsub at boot to derive fps/width/height/total_frames.

    The fansub video is the canonical reference frame space throughout the
    pipeline (ADR-0002 §3 Stage 1). Probing it once at boot avoids the
    transitory --fps-num/--fps-den/--fansub-* CLI flags that earlier phases
    of P1 had relied on.
    """
    meta = ffmpeg.probe(ns.hardsub)
    return PipelineGlobals(
        workdir=ns.workdir,
        hardsub_path=ns.hardsub,
        raw_path=ns.raw,
        out_path=ns.out,
        fps=Fraction(meta.fps_num, meta.fps_den),
        fansub_width=meta.width,
        fansub_height=meta.height,
        fansub_total_frames=meta.total_frames,
        debug_images=ns.debug_images,
    )


def _build_config(ns: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(
        ar_strategy=ns.ar_strategy,
        hardsub_audio_track=ns.hardsub_audio_track,
        raw_audio_track=ns.raw_audio_track,
        hardsub_skip_ranges=list(ns.hardsub_skip or []),
        raw_skip_ranges=list(ns.raw_skip or []),
    )
    # Route per-stage flags into their owning sub-configs (ADR-0002 §4).
    config.conform.ar_strategy = ns.ar_strategy
    config.ocr.language = ns.language
    config.ocr.device = ns.ocr_device
    config.event_cleanup.model = ns.event_cleanup_model
    config.event_cleanup.parallelism = ns.event_cleanup_parallelism
    config.doc_cleanup.model = ns.doc_cleanup_model
    config.doc_cleanup.parallelism = ns.doc_cleanup_parallelism
    config.doc_cleanup.synopsis_path = ns.synopsis
    config.export.color_cluster_threshold = ns.color_cluster_threshold
    return config


def parse_args(
    argv: list[str] | None = None,
    *,
    ffmpeg: FfmpegRunner | None = None,
) -> tuple[PipelineGlobals, PipelineConfig, bool]:
    ns = _build_parser().parse_args(argv)
    runner: FfmpegRunner = ffmpeg if ffmpeg is not None else SubprocessFfmpegRunner()
    globals_ = _build_globals(ns, runner)
    config = _build_config(ns)
    return globals_, config, ns.debug


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


class TzAwareFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return (
            datetime.fromtimestamp(record.created)
            .astimezone()
            .isoformat(timespec="milliseconds")
        )


class ShortNameFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.short_name = record.name.rsplit(".", 1)[-1]
        return True


def setup_logging(stdout_level: int, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] [%(short_name)s] %(message)s"
    formatter = TzAwareFormatter(fmt)
    short_filter = ShortNameFilter()

    stdout_h = logging.StreamHandler(sys.stdout)
    stdout_h.setLevel(stdout_level)
    stdout_h.setFormatter(formatter)
    stdout_h.addFilter(short_filter)

    file_h = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)
    file_h.addFilter(short_filter)

    root = logging.getLogger("subtitles_ocr")
    root.setLevel(logging.DEBUG)
    root.handlers = [stdout_h, file_h]
    root.propagate = False

    header = (
        f'==== run started {datetime.now().astimezone().isoformat(timespec="seconds")} '
        f'cmd="{shlex.join(sys.argv)}" ===='
    )
    file_h.stream.write(header + "\n")
    file_h.stream.flush()


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def build_stages(config: PipelineConfig) -> list:
    """Return the 9 MVP stages in pipeline order (ADR-0002 §2).

    Stages that depend on extra constructor wiring (root-level audio tracks,
    skip ranges) are instantiated here. The pure-config stages take no
    arguments.
    """
    from subtitles_ocr.pipeline.alignment.stage import AlignmentStage
    from subtitles_ocr.pipeline.animation import AnimationStage
    from subtitles_ocr.pipeline.color import ColorStage
    from subtitles_ocr.pipeline.conform import ConformStage
    from subtitles_ocr.pipeline.doc_cleanup import DocCleanupStage
    from subtitles_ocr.pipeline.event_cleanup import EventCleanupStage
    from subtitles_ocr.pipeline.export import ExportStage
    from subtitles_ocr.pipeline.group import GroupStage
    from subtitles_ocr.pipeline.ocr import OcrStage

    return [
        ConformStage(),
        AlignmentStage(
            hardsub_audio_track=config.hardsub_audio_track,
            raw_audio_track=config.raw_audio_track,
            hardsub_skip_ranges=list(config.hardsub_skip_ranges),
            raw_skip_ranges=list(config.raw_skip_ranges),
        ),
        OcrStage(),
        GroupStage(),
        AnimationStage(),
        ColorStage(),
        EventCleanupStage(),
        DocCleanupStage(),
        ExportStage(),
    ]


def run_pipeline(
    globals: PipelineGlobals,
    config: PipelineConfig,
    stages: list | None = None,
) -> None:
    actual_stages = stages if stages is not None else build_stages(config)
    results: dict[str, object] = {}
    for stage in actual_stages:
        stage_config = section_for(config, stage)
        kwargs: dict[str, object] = {}
        # OcrStage owns frame_processing and (in production) consumes the
        # previous stage's AlignmentResult to drive the diff/mask/compose loop.
        # Tests inject `composed_frames` directly and bypass this branch.
        if type(stage).__name__ == "OcrStage":
            alignment_result = results.get("alignment")
            if alignment_result is not None:
                kwargs["alignment_result"] = alignment_result
                kwargs["frame_processing_config"] = config.frame_processing
        result = stage.run(globals, stage_config, **kwargs)
        results[stage.CONFIG_FIELD] = result
    # Touch FrameProcessingConfig import so static checkers don't drop it.
    _ = FrameProcessingConfig


def main(
    argv: list[str] | None = None,
    *,
    ffmpeg: FfmpegRunner | None = None,
    stages_override: list | None = None,
) -> int:
    globals_, config, debug = parse_args(argv, ffmpeg=ffmpeg)
    globals_.workdir.mkdir(parents=True, exist_ok=True)
    stdout_level = logging.DEBUG if debug else logging.INFO
    setup_logging(stdout_level, globals_.workdir / "pipeline.log")
    try:
        run_pipeline(globals_, config, stages=stages_override)
    except PipelineError as e:
        logger.error(
            "[stage %s] %s: %s. Hint: %s",
            e.stage,
            type(e).__name__,
            e.args[0] if e.args else "",
            e.hint,
        )
        return 1
    return 0
