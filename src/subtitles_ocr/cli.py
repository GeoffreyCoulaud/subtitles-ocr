import argparse
import logging
import shlex
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from subtitles_ocr.config import PipelineConfig, PipelineGlobals, section_for
from subtitles_ocr.exceptions import PipelineError

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

    # TODO P3.1: replace with FfmpegRunner probe. These transitory flags exist
    # so PipelineGlobals can be constructed before SubprocessFfmpegRunner lands.
    p.add_argument("--fps-num", type=int, default=24)
    p.add_argument("--fps-den", type=int, default=1)
    p.add_argument("--fansub-width", type=int, default=1920)
    p.add_argument("--fansub-height", type=int, default=1080)
    p.add_argument("--fansub-total-frames", type=int, default=1)

    return p


def parse_args(argv: list[str] | None = None) -> tuple[PipelineGlobals, PipelineConfig]:
    ns = _build_parser().parse_args(argv)

    globals_ = PipelineGlobals(
        workdir=ns.workdir,
        hardsub_path=ns.hardsub,
        raw_path=ns.raw,
        out_path=ns.out,
        fps=Fraction(ns.fps_num, ns.fps_den),
        fansub_width=ns.fansub_width,
        fansub_height=ns.fansub_height,
        fansub_total_frames=ns.fansub_total_frames,
        debug_images=ns.debug_images,
    )

    config = PipelineConfig(
        ar_strategy=ns.ar_strategy,
        synopsis_path=ns.synopsis,
        color_cluster_threshold=ns.color_cluster_threshold,
        hardsub_audio_track=ns.hardsub_audio_track,
        raw_audio_track=ns.raw_audio_track,
        hardsub_skip_ranges=list(ns.hardsub_skip or []),
        raw_skip_ranges=list(ns.raw_skip or []),
    )

    return globals_, config


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


def build_stages() -> list:
    return []


def run_pipeline(
    globals: PipelineGlobals,
    config: PipelineConfig,
    stages: list | None = None,
) -> None:
    for stage in stages if stages is not None else build_stages():
        stage_config = section_for(config, stage)
        stage.run(globals, stage_config)


def main(argv: list[str] | None = None) -> int:
    globals_, config = parse_args(argv)
    globals_.workdir.mkdir(parents=True, exist_ok=True)
    stdout_level = logging.DEBUG if globals_.debug_images else logging.INFO
    setup_logging(stdout_level, globals_.workdir / "pipeline.log")
    try:
        run_pipeline(globals_, config)
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
