"""End-to-end test of ``main()`` with a fully fake pipeline.

Verifies the CLI boot path:
  - argparse → parse_args probes ffmpeg → PipelineGlobals built from probe
  - run_pipeline iterates the injected stages in order
  - each stage receives its own sub-config via ``section_for``
  - main() returns 0 on success and writes the pipeline.log header

The 9 production stages are mocked end-to-end; the real ffmpeg/ocr/llm
dependencies are never invoked. This is purely a wiring test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from subtitles_ocr.cli import build_stages, main
from subtitles_ocr.config import PipelineConfig, PipelineGlobals
from subtitles_ocr.ffmpeg.protocol import VideoMetadata


@pytest.fixture(autouse=True)
def _restore_subtitles_logger():
    # main() calls setup_logging which mutates the "subtitles_ocr" logger
    # (handlers, level, propagate). Restore around every test so we don't
    # pollute neighbours (e.g. test_retry's caplog expectations).
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


def _make_meta() -> VideoMetadata:
    return VideoMetadata(
        width=1920,
        height=1080,
        fps_num=24,
        fps_den=1,
        total_frames=120,
        duration_s=5.0,
        pix_fmt="yuv420p",
        colorspace="bt709",
    )


@dataclass
class _FakeProbeFfmpeg:
    meta: VideoMetadata = field(default_factory=_make_meta)
    probe_calls: list[Path] = field(default_factory=list)

    def probe(self, path: Path) -> VideoMetadata:
        self.probe_calls.append(path)
        return self.meta

    def transcode(self, args) -> None:  # pragma: no cover
        raise NotImplementedError

    def extract_audio(self, path: Path, track_index: int, out: Path) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class _Recorder:
    """Captures (stage_name, globals, config) tuples across all stages."""

    events: list[tuple[str, PipelineGlobals, BaseModel]] = field(default_factory=list)


def _fake_stage_cls(stage_name: str, config_field: str) -> type:
    """Generate a fake stage class with the given CONFIG_FIELD."""

    class _FakeStage:
        CONFIG_FIELD: ClassVar[str] = config_field
        GLOBALS_USED: ClassVar[tuple[str, ...]] = ()
        STAGE_VERSION: ClassVar[int] = 1

        def __init__(self, recorder: _Recorder) -> None:
            self._recorder = recorder
            self._name = stage_name

        def run(self, globals: PipelineGlobals, config, **kwargs) -> None:
            self._recorder.events.append((self._name, globals, config))
            return None

    _FakeStage.__name__ = f"Fake_{stage_name}"
    return _FakeStage


def _build_fake_stages(recorder: _Recorder) -> list:
    spec = [
        ("Conform", "conform"),
        ("Alignment", "alignment"),
        ("Ocr", "ocr"),
        ("Group", "group"),
        ("Animation", "animation"),
        ("Color", "color"),
        ("EventCleanup", "event_cleanup"),
        ("Normalize", "normalize"),
        ("Export", "export"),
    ]
    return [_fake_stage_cls(name, field)(recorder) for name, field in spec]


def _base_argv(tmp_path: Path, out_name: str = "out.ass") -> list[str]:
    return [
        "--hardsub", str(tmp_path / "hardsub.avi"),
        "--raw", str(tmp_path / "raw.mkv"),
        "--out", str(tmp_path / out_name),
        "--workdir", str(tmp_path / "work"),
    ]


def test_main_runs_all_nine_stages_in_order(tmp_path: Path) -> None:
    recorder = _Recorder()
    rc = main(
        _base_argv(tmp_path),
        ffmpeg=_FakeProbeFfmpeg(),
        stages_override=_build_fake_stages(recorder),
    )
    assert rc == 0
    names = [evt[0] for evt in recorder.events]
    assert names == [
        "Conform",
        "Alignment",
        "Ocr",
        "Group",
        "Animation",
        "Color",
        "EventCleanup",
        "Normalize",
        "Export",
    ]


def test_main_each_stage_receives_its_sub_config(tmp_path: Path) -> None:
    recorder = _Recorder()
    main(
        _base_argv(tmp_path),
        ffmpeg=_FakeProbeFfmpeg(),
        stages_override=_build_fake_stages(recorder),
    )
    field_by_name = {
        "Conform": "ConformConfig",
        "Alignment": "AlignmentConfig",
        "Ocr": "OcrConfig",
        "Group": "GroupConfig",
        "Animation": "AnimationConfig",
        "Color": "ColorConfig",
        "EventCleanup": "EventCleanupConfig",
        "Normalize": "NormalizeConfig",
        "Export": "ExportConfig",
    }
    for stage_name, _g, sub_config in recorder.events:
        assert type(sub_config).__name__ == field_by_name[stage_name]


def test_main_globals_built_from_probe(tmp_path: Path) -> None:
    recorder = _Recorder()
    meta = VideoMetadata(
        width=720,
        height=480,
        fps_num=30000,
        fps_den=1001,
        total_frames=42,
        duration_s=1.4,
        pix_fmt="yuv420p",
        colorspace=None,
    )
    ffmpeg = _FakeProbeFfmpeg(meta=meta)
    main(
        _base_argv(tmp_path),
        ffmpeg=ffmpeg,
        stages_override=_build_fake_stages(recorder),
    )
    # Probe was called on the hardsub path only.
    assert ffmpeg.probe_calls == [tmp_path / "hardsub.avi"]
    # Globals propagated from probe meta.
    _name, g, _cfg = recorder.events[0]
    assert g.fansub_width == 720
    assert g.fansub_height == 480
    assert g.fansub_total_frames == 42
    assert g.fps.numerator == 30000
    assert g.fps.denominator == 1001


def test_main_writes_pipeline_log(tmp_path: Path) -> None:
    recorder = _Recorder()
    main(
        _base_argv(tmp_path),
        ffmpeg=_FakeProbeFfmpeg(),
        stages_override=_build_fake_stages(recorder),
    )
    log = tmp_path / "work" / "pipeline.log"
    assert log.exists()
    assert "==== run started" in log.read_text()


def test_main_routes_flags_to_their_sub_configs(tmp_path: Path) -> None:
    """End-to-end: every routed CLI flag reaches the right sub-config."""
    recorder = _Recorder()
    rc = main(
        [
            *_base_argv(tmp_path),
            "--language", "japan",
            "--ocr-device", "cpu",
            "--event-cleanup-model", "m-event",
            "--event-cleanup-parallelism", "7",
            "--color-cluster-threshold", "15.5",
            "--ar-strategy", "letterbox",
        ],
        ffmpeg=_FakeProbeFfmpeg(),
        stages_override=_build_fake_stages(recorder),
    )
    assert rc == 0
    by_name = {evt[0]: evt[2] for evt in recorder.events}
    assert by_name["Ocr"].language == "japan"
    assert by_name["Ocr"].device == "cpu"
    assert by_name["EventCleanup"].model == "m-event"
    assert by_name["EventCleanup"].parallelism == 7
    assert by_name["Export"].color_cluster_threshold == 15.5
    assert by_name["Conform"].ar_strategy == "letterbox"


def test_build_stages_callable_with_default_config() -> None:
    """build_stages must work without any flag — defaults compose correctly."""
    stages = build_stages(PipelineConfig())
    assert len(stages) == 9


def test_main_default_invocation_no_extra_flags(tmp_path: Path) -> None:
    """Sanity: minimal CLI invocation succeeds when all stages are stubbed."""
    recorder = _Recorder()
    rc = main(
        _base_argv(tmp_path),
        ffmpeg=_FakeProbeFfmpeg(),
        stages_override=_build_fake_stages(recorder),
    )
    assert rc == 0
    assert len(recorder.events) == 9
