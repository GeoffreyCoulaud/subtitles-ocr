import logging
from fractions import Fraction
from pathlib import Path

import pytest

from subtitles_ocr.cli import build_stages, main, parse_args, run_pipeline, setup_logging
from subtitles_ocr.config import PipelineConfig, PipelineGlobals
from subtitles_ocr.exceptions import AlignmentRatioTooLow


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
    globals_, config = parse_args(_base_args(tmp_path))
    assert isinstance(globals_, PipelineGlobals)
    assert isinstance(config, PipelineConfig)


def test_parse_args_paths_map_into_globals(tmp_path: Path) -> None:
    globals_, _ = parse_args(_base_args(tmp_path))
    assert globals_.hardsub_path == tmp_path / "hardsub.avi"
    assert globals_.raw_path == tmp_path / "raw.mkv"
    assert globals_.out_path == tmp_path / "out.ass"
    assert globals_.workdir == tmp_path / "work"


def test_parse_args_debug_images_flag(tmp_path: Path) -> None:
    globals_, _ = parse_args([*_base_args(tmp_path), "--debug-images"])
    assert globals_.debug_images is True


def test_parse_args_debug_images_default_false(tmp_path: Path) -> None:
    globals_, _ = parse_args(_base_args(tmp_path))
    assert globals_.debug_images is False


def test_parse_args_fps_defaults_to_24(tmp_path: Path) -> None:
    globals_, _ = parse_args(_base_args(tmp_path))
    assert globals_.fps == Fraction(24, 1)


def test_parse_args_fps_num_den_override(tmp_path: Path) -> None:
    globals_, _ = parse_args(
        [*_base_args(tmp_path), "--fps-num", "24000", "--fps-den", "1001"]
    )
    assert globals_.fps == Fraction(24000, 1001)


def test_parse_args_ar_strategy(tmp_path: Path) -> None:
    _, config = parse_args([*_base_args(tmp_path), "--ar-strategy", "letterbox"])
    assert config.ar_strategy == "letterbox"


def test_parse_args_ar_strategy_default(tmp_path: Path) -> None:
    _, config = parse_args(_base_args(tmp_path))
    assert config.ar_strategy == "error"


def test_parse_args_synopsis(tmp_path: Path) -> None:
    syn = tmp_path / "syn.txt"
    _, config = parse_args([*_base_args(tmp_path), "--synopsis", str(syn)])
    assert config.synopsis_path == syn


def test_parse_args_color_cluster_threshold(tmp_path: Path) -> None:
    _, config = parse_args([*_base_args(tmp_path), "--color-cluster-threshold", "12.5"])
    assert config.color_cluster_threshold == 12.5


def test_parse_args_hardsub_audio_track(tmp_path: Path) -> None:
    _, config = parse_args([*_base_args(tmp_path), "--hardsub-audio-track", "1"])
    assert config.hardsub_audio_track == 1


def test_parse_args_raw_audio_track(tmp_path: Path) -> None:
    _, config = parse_args([*_base_args(tmp_path), "--raw-audio-track", "0"])
    assert config.raw_audio_track == 0


def test_parse_args_hardsub_skip_repeatable(tmp_path: Path) -> None:
    _, config = parse_args(
        [
            *_base_args(tmp_path),
            "--hardsub-skip",
            "00:00:00-00:00:10",
            "--hardsub-skip",
            "00:01:00-00:01:30",
        ]
    )
    assert config.hardsub_skip_ranges == ["00:00:00-00:00:10", "00:01:00-00:01:30"]


def test_parse_args_hardsub_skip_empty_default(tmp_path: Path) -> None:
    _, config = parse_args(_base_args(tmp_path))
    assert config.hardsub_skip_ranges == []


def test_parse_args_raw_skip_repeatable(tmp_path: Path) -> None:
    _, config = parse_args(
        [
            *_base_args(tmp_path),
            "--raw-skip",
            "00:00:00-00:00:05",
            "--raw-skip",
            "00:02:00-00:02:15",
        ]
    )
    assert config.raw_skip_ranges == ["00:00:00-00:00:05", "00:02:00-00:02:15"]


# Stage-specific flags (--language, --ocr-device, --event-cleanup-*, --doc-cleanup-*)
# are accepted by argparse today but live on their stage's sub-config, which is empty
# at this phase. They are wired into their stage's sub-model in P4 (Stage 6, 10, 11).
# For now we only verify parse_args accepts them without error.
def test_parse_args_accepts_stage_flags_without_error(tmp_path: Path) -> None:
    globals_, config = parse_args(
        [
            *_base_args(tmp_path),
            "--language", "japan",
            "--ocr-device", "cuda",
            "--event-cleanup-model", "qwen2.5:7b",
            "--event-cleanup-parallelism", "4",
            "--doc-cleanup-model", "qwen2.5:14b",
            "--doc-cleanup-parallelism", "2",
        ]
    )
    assert isinstance(globals_, PipelineGlobals)
    assert isinstance(config, PipelineConfig)


def test_parse_args_argv_none_uses_sys_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["prog", *_base_args(tmp_path)])
    globals_, _ = parse_args(None)
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


def test_build_stages_returns_empty_list() -> None:
    assert build_stages() == []


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


def test_run_pipeline_uses_build_stages_when_none(
    mock_globals: PipelineGlobals,
) -> None:
    # build_stages returns [] so this is a no-op; should not raise.
    run_pipeline(mock_globals, PipelineConfig())


def test_run_pipeline_propagates_pipeline_error(
    mock_globals: PipelineGlobals,
) -> None:
    with pytest.raises(AlignmentRatioTooLow):
        run_pipeline(mock_globals, PipelineConfig(), stages=[FakeStage()])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_returns_one_on_pipeline_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("subtitles_ocr.cli.build_stages", lambda: [FakeStage()])
    rc = main(_base_args(tmp_path))
    assert rc == 1


def test_main_logs_pipeline_error_with_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("subtitles_ocr.cli.build_stages", lambda: [FakeStage()])
    main(_base_args(tmp_path))
    log_file = tmp_path / "work" / "pipeline.log"
    content = log_file.read_text()
    assert "[stage 02_alignment]" in content
    assert "AlignmentRatioTooLow" in content
    assert "32% orphan ratio" in content
    assert "Hint: provide --hardsub-skip ranges" in content


def test_main_propagates_bug_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("subtitles_ocr.cli.build_stages", lambda: [BugStage()])
    with pytest.raises(RuntimeError, match="bug"):
        main(_base_args(tmp_path))


def test_main_creates_workdir_if_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("subtitles_ocr.cli.build_stages", lambda: [HappyStage()])
    workdir = tmp_path / "work"
    assert not workdir.exists()
    rc = main(_base_args(tmp_path))
    assert rc == 0
    assert workdir.is_dir()
    assert (workdir / "pipeline.log").exists()
