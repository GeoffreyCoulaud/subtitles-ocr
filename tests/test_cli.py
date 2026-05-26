import logging
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import pytest

from subtitles_ocr.cli import build_stages, main, parse_args, run_pipeline, setup_logging
from subtitles_ocr.config import PipelineConfig, PipelineGlobals
from subtitles_ocr.exceptions import AlignmentRatioTooLow
from subtitles_ocr.ffmpeg.protocol import VideoMetadata


def _make_fake_meta(width: int = 1920, height: int = 1080, fps_num: int = 24) -> VideoMetadata:
    return VideoMetadata(
        width=width,
        height=height,
        fps_num=fps_num,
        fps_den=1,
        total_frames=120,
        duration_s=5.0,
        pix_fmt="yuv420p",
        colorspace="bt709",
    )


@dataclass
class _FakeProbeFfmpeg:
    """Minimal FfmpegRunner that only services probe(); other methods unused."""

    meta: VideoMetadata = field(default_factory=_make_fake_meta)
    probe_calls: list[Path] = field(default_factory=list)

    def probe(self, path: Path) -> VideoMetadata:
        self.probe_calls.append(path)
        return self.meta

    def transcode(self, args) -> None:  # pragma: no cover — not used here
        raise NotImplementedError

    def extract_audio(self, path: Path, track_index: int, out: Path) -> None:  # pragma: no cover
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _restore_subtitles_logger():
    # setup_logging mutates the "subtitles_ocr" logger (handlers, level, propagate).
    # Restore around every test so we don't pollute neighbours (e.g. test_retry).
    log = logging.getLogger("subtitles_ocr")
    saved_handlers = log.handlers[:]
    saved_level = log.level
    saved_propagate = log.propagate
    try:
        yield
    finally:
        for h in log.handlers:
            try:
                h.close()
            except Exception:
                pass
        log.handlers = saved_handlers
        log.level = saved_level
        log.propagate = saved_propagate


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--hardsub", str(tmp_path / "hardsub.avi"),
        "--raw", str(tmp_path / "raw.mkv"),
        "--out", str(tmp_path / "out.ass"),
        "--workdir", str(tmp_path / "work"),
    ]


def _parse(argv: list[str], meta: VideoMetadata | None = None) -> tuple[PipelineGlobals, PipelineConfig, bool]:
    ffmpeg = _FakeProbeFfmpeg(meta=meta or _make_fake_meta())
    return parse_args(argv, ffmpeg=ffmpeg)


# ---------------------------------------------------------------------------
# Fixtures smoke
# ---------------------------------------------------------------------------


def test_fixtures_smoke(tmp_workdir: Path, mock_globals: PipelineGlobals) -> None:
    assert (tmp_workdir / "06_ocr").is_dir()
    assert mock_globals.fps == Fraction(24, 1)
    assert mock_globals.workdir == tmp_workdir


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_returns_globals_and_config(tmp_path: Path) -> None:
    globals_, config, _debug = _parse(_base_args(tmp_path))
    assert isinstance(globals_, PipelineGlobals)
    assert isinstance(config, PipelineConfig)


def test_parse_args_paths_map_into_globals(tmp_path: Path) -> None:
    globals_, _config, _debug = _parse(_base_args(tmp_path))
    assert globals_.hardsub_path == tmp_path / "hardsub.avi"
    assert globals_.raw_path == tmp_path / "raw.mkv"
    assert globals_.out_path == tmp_path / "out.ass"
    assert globals_.workdir == tmp_path / "work"


def test_parse_args_debug_images_flag(tmp_path: Path) -> None:
    globals_, _config, _debug = _parse([*_base_args(tmp_path), "--debug-images"])
    assert globals_.debug_images is True


def test_parse_args_debug_images_default_false(tmp_path: Path) -> None:
    globals_, _config, _debug = _parse(_base_args(tmp_path))
    assert globals_.debug_images is False


def test_parse_args_debug_flag_true(tmp_path: Path) -> None:
    _globals, _config, debug = _parse([*_base_args(tmp_path), "--debug"])
    assert debug is True


def test_parse_args_debug_flag_default_false(tmp_path: Path) -> None:
    _globals, _config, debug = _parse(_base_args(tmp_path))
    assert debug is False


def test_parse_args_globals_built_from_ffmpeg_probe(tmp_path: Path) -> None:
    """Probe at boot: fps/width/height/total_frames come from FfmpegRunner.probe."""
    meta = _make_fake_meta(width=1280, height=720, fps_num=24000)
    meta = meta.model_copy(update={"fps_den": 1001, "total_frames": 17280})
    globals_, _config, _debug = _parse(_base_args(tmp_path), meta=meta)
    assert globals_.fansub_width == 1280
    assert globals_.fansub_height == 720
    assert globals_.fps == Fraction(24000, 1001)
    assert globals_.fansub_total_frames == 17280


def test_parse_args_probes_hardsub_path(tmp_path: Path) -> None:
    ffmpeg = _FakeProbeFfmpeg()
    parse_args(_base_args(tmp_path), ffmpeg=ffmpeg)
    assert ffmpeg.probe_calls == [tmp_path / "hardsub.avi"]


def test_parse_args_rejects_legacy_fps_flags(tmp_path: Path) -> None:
    """The transitory --fps-num/--fps-den flags are gone — argparse must reject them."""
    with pytest.raises(SystemExit):
        _parse([*_base_args(tmp_path), "--fps-num", "24000"])


def test_parse_args_ar_strategy_routed_to_conform(tmp_path: Path) -> None:
    _globals, config, _debug = _parse([*_base_args(tmp_path), "--ar-strategy", "letterbox"])
    assert config.conform.ar_strategy == "letterbox"


def test_parse_args_ar_strategy_default(tmp_path: Path) -> None:
    _globals, config, _debug = _parse(_base_args(tmp_path))
    assert config.conform.ar_strategy == "error"


def test_parse_args_ar_strategy_only_on_conform(tmp_path: Path) -> None:
    """Issue 3: ar_strategy now lives exclusively in ConformConfig — the root
    field has been removed to eliminate the duplication."""
    _globals, config, _debug = _parse([*_base_args(tmp_path), "--ar-strategy", "crop"])
    assert config.conform.ar_strategy == "crop"
    assert not hasattr(config, "ar_strategy")


def test_parse_args_synopsis_routed_to_doc_cleanup(tmp_path: Path) -> None:
    syn = tmp_path / "syn.txt"
    _globals, config, _debug = _parse([*_base_args(tmp_path), "--synopsis", str(syn)])
    assert config.doc_cleanup.synopsis_path == syn


def test_parse_args_color_cluster_threshold_routed_to_export(tmp_path: Path) -> None:
    _globals, config, _debug = _parse([*_base_args(tmp_path), "--color-cluster-threshold", "12.5"])
    assert config.export.color_cluster_threshold == 12.5


def test_parse_args_language_routed_to_ocr(tmp_path: Path) -> None:
    _globals, config, _debug = _parse([*_base_args(tmp_path), "--language", "japan"])
    assert config.ocr.language == "japan"


def test_parse_args_ocr_device_routed_to_ocr(tmp_path: Path) -> None:
    _globals, config, _debug = _parse([*_base_args(tmp_path), "--ocr-device", "cuda"])
    assert config.ocr.device == "cuda"


def test_parse_args_event_cleanup_model_routed(tmp_path: Path) -> None:
    _globals, config, _debug = _parse(
        [*_base_args(tmp_path), "--event-cleanup-model", "qwen2.5:7b"]
    )
    assert config.event_cleanup.model == "qwen2.5:7b"


def test_parse_args_event_cleanup_parallelism_routed(tmp_path: Path) -> None:
    _globals, config, _debug = _parse(
        [*_base_args(tmp_path), "--event-cleanup-parallelism", "8"]
    )
    assert config.event_cleanup.parallelism == 8


def test_parse_args_doc_cleanup_model_routed(tmp_path: Path) -> None:
    _globals, config, _debug = _parse(
        [*_base_args(tmp_path), "--doc-cleanup-model", "qwen2.5:14b"]
    )
    assert config.doc_cleanup.model == "qwen2.5:14b"


def test_parse_args_doc_cleanup_parallelism_routed(tmp_path: Path) -> None:
    _globals, config, _debug = _parse(
        [*_base_args(tmp_path), "--doc-cleanup-parallelism", "3"]
    )
    assert config.doc_cleanup.parallelism == 3


def test_parse_args_hardsub_audio_track_routed_to_alignment(tmp_path: Path) -> None:
    """Issue 4: audio-track flags participate in AlignmentStage's sidecar."""
    _globals, config, _debug = _parse([*_base_args(tmp_path), "--hardsub-audio-track", "1"])
    assert config.alignment.hardsub_audio_track == 1


def test_parse_args_raw_audio_track_routed_to_alignment(tmp_path: Path) -> None:
    _globals, config, _debug = _parse([*_base_args(tmp_path), "--raw-audio-track", "0"])
    assert config.alignment.raw_audio_track == 0


def test_parse_args_hardsub_skip_repeatable_routed_to_alignment(tmp_path: Path) -> None:
    _globals, config, _debug = _parse(
        [
            *_base_args(tmp_path),
            "--hardsub-skip",
            "00:00:00-00:00:10",
            "--hardsub-skip",
            "00:01:00-00:01:30",
        ]
    )
    assert config.alignment.hardsub_skip_ranges == [
        "00:00:00-00:00:10",
        "00:01:00-00:01:30",
    ]


def test_parse_args_hardsub_skip_empty_default_on_alignment(tmp_path: Path) -> None:
    _globals, config, _debug = _parse(_base_args(tmp_path))
    assert config.alignment.hardsub_skip_ranges == []


def test_parse_args_raw_skip_repeatable_routed_to_alignment(tmp_path: Path) -> None:
    _globals, config, _debug = _parse(
        [
            *_base_args(tmp_path),
            "--raw-skip",
            "00:00:00-00:00:05",
            "--raw-skip",
            "00:02:00-00:02:15",
        ]
    )
    assert config.alignment.raw_skip_ranges == [
        "00:00:00-00:00:05",
        "00:02:00-00:02:15",
    ]


def test_parse_args_argv_none_uses_sys_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["prog", *_base_args(tmp_path)])
    ffmpeg = _FakeProbeFfmpeg()
    globals_, _config, _debug = parse_args(None, ffmpeg=ffmpeg)
    assert globals_.hardsub_path == tmp_path / "hardsub.avi"


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


def test_setup_logging_creates_log_file(tmp_path: Path) -> None:
    log_file = tmp_path / "pipeline.log"
    setup_logging(logging.INFO, log_file)
    assert log_file.exists()


def test_setup_logging_writes_header(tmp_path: Path) -> None:
    log_file = tmp_path / "pipeline.log"
    setup_logging(logging.INFO, log_file)
    content = log_file.read_text()
    assert "==== run started" in content
    assert 'cmd="' in content


def test_setup_logging_log_messages_appear_in_file(tmp_path: Path) -> None:
    log_file = tmp_path / "pipeline.log"
    setup_logging(logging.INFO, log_file)
    logger = logging.getLogger("subtitles_ocr.test_stage")
    logger.warning("hello-from-test")
    for h in logging.getLogger("subtitles_ocr").handlers:
        h.flush()
    content = log_file.read_text()
    assert "hello-from-test" in content
    assert "[test_stage]" in content


def test_setup_logging_appends_on_second_call(tmp_path: Path) -> None:
    log_file = tmp_path / "pipeline.log"
    setup_logging(logging.INFO, log_file)
    setup_logging(logging.INFO, log_file)
    content = log_file.read_text()
    assert content.count("==== run started") == 2


# ---------------------------------------------------------------------------
# build_stages
# ---------------------------------------------------------------------------


def test_build_stages_returns_nine_stages() -> None:
    stages = build_stages(PipelineConfig())
    assert len(stages) == 9


def test_build_stages_in_pipeline_order() -> None:
    stages = build_stages(PipelineConfig())
    names = [type(s).__name__ for s in stages]
    assert names == [
        "ConformStage",
        "AlignmentStage",
        "OcrStage",
        "GroupStage",
        "AnimationStage",
        "ColorStage",
        "EventCleanupStage",
        "DocCleanupStage",
        "ExportStage",
    ]


def test_build_stages_alignment_constructed_without_audio_or_skip_kwargs() -> None:
    """Issue 4: AlignmentStage no longer takes audio/skip via constructor —
    those values are routed through ``config.alignment`` and read inside ``run()``.
    """
    from subtitles_ocr.config import AlignmentConfig

    cfg = PipelineConfig(
        alignment=AlignmentConfig(
            hardsub_audio_track=2,
            raw_audio_track=3,
            hardsub_skip_ranges=["00:00:00-00:00:05"],
            raw_skip_ranges=["00:00:10-00:00:15"],
        )
    )
    stages = build_stages(cfg)
    alignment = next(s for s in stages if type(s).__name__ == "AlignmentStage")
    # The values must reach the stage via its sub-config, not via attributes.
    assert cfg.alignment.hardsub_audio_track == 2
    assert cfg.alignment.raw_audio_track == 3
    assert cfg.alignment.hardsub_skip_ranges == ["00:00:00-00:00:05"]
    assert cfg.alignment.raw_skip_ranges == ["00:00:10-00:00:15"]
    # Constructor must not be carrying these as instance attributes anymore.
    for forbidden in (
        "hardsub_audio_track",
        "raw_audio_track",
        "hardsub_skip_ranges",
        "raw_skip_ranges",
    ):
        assert not hasattr(alignment, forbidden), (
            f"AlignmentStage must not store {forbidden!r} as an instance attribute"
        )


# ---------------------------------------------------------------------------
# Fake stages for orchestrator tests
# ---------------------------------------------------------------------------


class FakeStage:
    CONFIG_FIELD = "ocr"
    GLOBALS_USED: tuple[str, ...] = ()
    STAGE_VERSION = 1

    def __init__(self) -> None:
        self.called = False

    def run(self, g: PipelineGlobals, c: object) -> None:
        self.called = True
        raise AlignmentRatioTooLow(
            "32% orphan ratio",
            stage="02_alignment",
            hint="provide --hardsub-skip ranges",
        )


class BugStage:
    CONFIG_FIELD = "ocr"
    GLOBALS_USED: tuple[str, ...] = ()
    STAGE_VERSION = 1

    def run(self, g: PipelineGlobals, c: object) -> None:
        raise RuntimeError("bug")


class HappyStage:
    CONFIG_FIELD = "ocr"
    GLOBALS_USED: tuple[str, ...] = ()
    STAGE_VERSION = 1

    def __init__(self) -> None:
        self.called = False

    def run(self, g: PipelineGlobals, c: object) -> None:
        self.called = True


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


def test_run_pipeline_invokes_each_stage_with_its_section(
    mock_globals: PipelineGlobals,
) -> None:
    stage = HappyStage()
    config = PipelineConfig()
    run_pipeline(mock_globals, config, stages=[stage])
    assert stage.called is True


def test_run_pipeline_uses_provided_stages(
    mock_globals: PipelineGlobals,
) -> None:
    s1, s2 = HappyStage(), HappyStage()
    run_pipeline(mock_globals, PipelineConfig(), stages=[s1, s2])
    assert s1.called and s2.called


def test_run_pipeline_propagates_pipeline_error(
    mock_globals: PipelineGlobals,
) -> None:
    with pytest.raises(AlignmentRatioTooLow):
        run_pipeline(mock_globals, PipelineConfig(), stages=[FakeStage()])


def test_run_pipeline_invokes_stages_with_only_globals_and_config(
    mock_globals: PipelineGlobals,
) -> None:
    """Issue 1: orchestrator must use the strict ``run(globals, config)``
    signature (ADR-0004 §3.1 / §3.3) — no kwargs, no result chaining.
    """

    class StrictSignatureStage:
        CONFIG_FIELD = "ocr"
        GLOBALS_USED: tuple[str, ...] = ()
        STAGE_VERSION = 1

        def __init__(self) -> None:
            self.received_args: tuple | None = None
            self.received_kwargs: dict | None = None

        def run(self, *args, **kwargs):  # noqa: D401
            self.received_args = args
            self.received_kwargs = dict(kwargs)
            return None

    s = StrictSignatureStage()
    run_pipeline(mock_globals, PipelineConfig(), stages=[s])
    assert s.received_args is not None
    assert len(s.received_args) == 2
    assert s.received_kwargs == {}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _main(argv: list[str], stages: list) -> int:
    ffmpeg = _FakeProbeFfmpeg()
    return main(argv, ffmpeg=ffmpeg, stages_override=stages)


def test_main_returns_one_on_pipeline_error(tmp_path: Path) -> None:
    rc = _main(_base_args(tmp_path), [FakeStage()])
    assert rc == 1


def test_main_logs_pipeline_error_with_format(tmp_path: Path) -> None:
    _main(_base_args(tmp_path), [FakeStage()])
    log_file = tmp_path / "work" / "pipeline.log"
    content = log_file.read_text()
    assert "[stage 02_alignment]" in content
    assert "AlignmentRatioTooLow" in content
    assert "32% orphan ratio" in content
    assert "Hint: provide --hardsub-skip ranges" in content


def test_main_propagates_bug_exceptions(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="bug"):
        _main(_base_args(tmp_path), [BugStage()])


def test_main_creates_workdir_if_missing(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    assert not workdir.exists()
    rc = _main(_base_args(tmp_path), [HappyStage()])
    assert rc == 0
    assert workdir.is_dir()
    assert (workdir / "pipeline.log").exists()


def test_main_uses_debug_flag_for_stdout_level(tmp_path: Path) -> None:
    """--debug must set stdout handler level to DEBUG; without it, INFO."""
    # Without --debug: stdout handler should be INFO
    _main(_base_args(tmp_path), [HappyStage()])
    root = logging.getLogger("subtitles_ocr")
    stdout_handler = next(
        h for h in root.handlers if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    )
    assert stdout_handler.level == logging.INFO

    # With --debug: stdout handler should be DEBUG
    _main([*_base_args(tmp_path), "--debug"], [HappyStage()])
    root = logging.getLogger("subtitles_ocr")
    stdout_handler = next(
        h for h in root.handlers if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    )
    assert stdout_handler.level == logging.DEBUG
