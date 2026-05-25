"""Tests for ConformStage (Stage 1 — spatial conform).

Per ADR-0004 §10.3, the external ffmpeg dependency is fronted by the
`FfmpegRunner` Protocol; tests inject a `FakeFfmpegRunner` recording calls.
No monkeypatching of subprocess / ffmpeg-python / PyAV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from subtitles_ocr.config import ConformConfig, PipelineGlobals
from subtitles_ocr.exceptions import AspectRatioMismatch
from subtitles_ocr.ffmpeg.protocol import TranscodeArgs, VideoMetadata
from subtitles_ocr.pipeline.conform import STAGE_VERSION, ConformResult, ConformStage


@dataclass
class FakeFfmpegRunner:
    """In-memory FfmpegRunner. Records calls; writes a stub output file on transcode."""

    probe_returns: dict[Path, VideoMetadata] = field(default_factory=dict)
    transcode_calls: list[TranscodeArgs] = field(default_factory=list)
    probe_calls: list[Path] = field(default_factory=list)

    def probe(self, path: Path) -> VideoMetadata:
        self.probe_calls.append(path)
        return self.probe_returns[path]

    def transcode(self, args: TranscodeArgs) -> None:
        self.transcode_calls.append(args)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_bytes(b"FAKE_MKV_CONTENT")

    def extract_audio(self, path: Path, track_index: int, out: Path) -> None:
        raise AssertionError("extract_audio is not used by ConformStage")


def _meta(width: int, height: int, *, colorspace: str | None = "bt709") -> VideoMetadata:
    return VideoMetadata(
        width=width,
        height=height,
        fps_num=24000,
        fps_den=1001,
        total_frames=1000,
        duration_s=41.7,
        pix_fmt="yuv420p",
        colorspace=colorspace,
    )


@pytest.fixture
def real_input_files(tmp_workdir: Path) -> tuple[Path, Path]:
    """Real on-disk files so we can fingerprint them (mtime/size) for the cache."""
    hardsub = tmp_workdir / "fansub.avi"
    raw = tmp_workdir / "raw.mkv"
    hardsub.write_bytes(b"FAKE_FANSUB" * 100)
    raw.write_bytes(b"FAKE_RAW_SOURCE" * 100)
    return hardsub, raw


@pytest.fixture
def globals_with_real_inputs(
    real_input_files: tuple[Path, Path],
    mock_globals: PipelineGlobals,
) -> PipelineGlobals:
    hardsub, raw = real_input_files
    return mock_globals.model_copy(update={"hardsub_path": hardsub, "raw_path": raw})


# -------------------- AR match: transcode runs once, sidecar written --------------------

def test_ar_match_transcodes_once_and_writes_sidecar(
    tmp_workdir: Path,
    globals_with_real_inputs: PipelineGlobals,
) -> None:
    fake = FakeFfmpegRunner(
        probe_returns={
            globals_with_real_inputs.hardsub_path: _meta(1440, 1080),  # 4:3
            globals_with_real_inputs.raw_path: _meta(1920, 1440),       # 4:3
        }
    )
    stage = ConformStage(ffmpeg=fake)

    result = stage.run(globals_with_real_inputs, ConformConfig())

    assert isinstance(result, ConformResult)
    assert len(fake.transcode_calls) == 1
    out_path = tmp_workdir / "01_conform" / "raw.mkv"
    meta_path = tmp_workdir / "01_conform" / "raw.meta.json"
    assert out_path.exists()
    assert meta_path.exists()
    assert result.raw_conformed_path == out_path


# -------------------- AR mismatch + error strategy: raises, no transcode --------------------

def test_ar_mismatch_error_strategy_raises_and_skips_transcode(
    globals_with_real_inputs: PipelineGlobals,
) -> None:
    fake = FakeFfmpegRunner(
        probe_returns={
            globals_with_real_inputs.hardsub_path: _meta(1440, 1080),  # 4:3
            globals_with_real_inputs.raw_path: _meta(1920, 1080),       # 16:9
        }
    )
    stage = ConformStage(ffmpeg=fake)
    cfg = ConformConfig(ar_strategy="error")

    with pytest.raises(AspectRatioMismatch):
        stage.run(globals_with_real_inputs, cfg)

    assert fake.transcode_calls == []


# -------------------- Re-run: sidecar valid → no transcode --------------------

def test_rerun_with_valid_sidecar_skips_transcode(
    globals_with_real_inputs: PipelineGlobals,
) -> None:
    fake = FakeFfmpegRunner(
        probe_returns={
            globals_with_real_inputs.hardsub_path: _meta(1440, 1080),
            globals_with_real_inputs.raw_path: _meta(1920, 1440),
        }
    )
    stage = ConformStage(ffmpeg=fake)

    stage.run(globals_with_real_inputs, ConformConfig())
    assert len(fake.transcode_calls) == 1

    stage.run(globals_with_real_inputs, ConformConfig())
    assert len(fake.transcode_calls) == 1  # NOT re-called


# -------------------- mtime change → invalidate cache --------------------

def test_mtime_change_invalidates_cache(
    globals_with_real_inputs: PipelineGlobals,
) -> None:
    fake = FakeFfmpegRunner(
        probe_returns={
            globals_with_real_inputs.hardsub_path: _meta(1440, 1080),
            globals_with_real_inputs.raw_path: _meta(1920, 1440),
        }
    )
    stage = ConformStage(ffmpeg=fake)

    stage.run(globals_with_real_inputs, ConformConfig())
    assert len(fake.transcode_calls) == 1

    # bump mtime of the raw input by writing it again with new content
    raw = globals_with_real_inputs.raw_path
    raw.write_bytes(b"DIFFERENT_RAW_CONTENT" * 100)

    stage.run(globals_with_real_inputs, ConformConfig())
    assert len(fake.transcode_calls) == 2


# -------------------- STAGE_VERSION bump simulated → invalidate --------------------

def test_stage_version_bump_invalidates_cache(
    tmp_workdir: Path,
    globals_with_real_inputs: PipelineGlobals,
) -> None:
    fake = FakeFfmpegRunner(
        probe_returns={
            globals_with_real_inputs.hardsub_path: _meta(1440, 1080),
            globals_with_real_inputs.raw_path: _meta(1920, 1440),
        }
    )
    stage = ConformStage(ffmpeg=fake)

    stage.run(globals_with_real_inputs, ConformConfig())
    assert len(fake.transcode_calls) == 1

    # Patch the persisted sidecar to claim a different stage_version, simulating
    # an old cache produced by a previous code version.
    meta_path = tmp_workdir / "01_conform" / "raw.meta.json"
    persisted = json.loads(meta_path.read_text())
    persisted["stage_version"] = STAGE_VERSION + 99
    meta_path.write_text(json.dumps(persisted))

    stage.run(globals_with_real_inputs, ConformConfig())
    assert len(fake.transcode_calls) == 2


# -------------------- Filter chain contains scale=...:flags=area and format=yuv420p --------------------

def test_filter_chain_uses_area_flag_and_yuv420p(
    globals_with_real_inputs: PipelineGlobals,
) -> None:
    fake = FakeFfmpegRunner(
        probe_returns={
            globals_with_real_inputs.hardsub_path: _meta(1440, 1080),
            globals_with_real_inputs.raw_path: _meta(1920, 1440),
        }
    )
    stage = ConformStage(ffmpeg=fake)

    stage.run(globals_with_real_inputs, ConformConfig())

    chain = fake.transcode_calls[0].filter_chain
    assert "scale=" in chain
    assert "flags=area" in chain
    assert "format=yuv420p" in chain


# -------------------- Filter chain encodes target W:H matching fansub dims --------------------

def test_filter_chain_targets_fansub_dimensions(
    globals_with_real_inputs: PipelineGlobals,
) -> None:
    fake = FakeFfmpegRunner(
        probe_returns={
            # fansub is 1440x1080 (4:3)
            globals_with_real_inputs.hardsub_path: _meta(1440, 1080),
            # raw is 1920x1440 (4:3, bigger)
            globals_with_real_inputs.raw_path: _meta(1920, 1440),
        }
    )
    globals_4_3 = globals_with_real_inputs.model_copy(
        update={"fansub_width": 1440, "fansub_height": 1080}
    )
    stage = ConformStage(ffmpeg=fake)

    stage.run(globals_4_3, ConformConfig())

    call = fake.transcode_calls[0]
    assert call.target_width == 1440
    assert call.target_height == 1080
    assert "1440:1080" in call.filter_chain
