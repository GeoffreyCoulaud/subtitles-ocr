"""Smoke tests for the ffmpeg sub-package: import and Pydantic round-trip."""

from pathlib import Path

from subtitles_ocr.ffmpeg import FfmpegRunner, TranscodeArgs, VideoMetadata


def test_video_metadata_round_trip() -> None:
    original = VideoMetadata(
        width=1920,
        height=1080,
        fps_num=24000,
        fps_den=1001,
        total_frames=35832,
        duration_s=1493.5,
        pix_fmt="yuv420p",
        colorspace="bt709",
    )
    restored = VideoMetadata.model_validate_json(original.model_dump_json())
    assert restored == original


def test_video_metadata_round_trip_colorspace_none() -> None:
    original = VideoMetadata(
        width=640,
        height=480,
        fps_num=25,
        fps_den=1,
        total_frames=1000,
        duration_s=40.0,
        pix_fmt="yuv420p",
        colorspace=None,
    )
    restored = VideoMetadata.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.colorspace is None


def test_transcode_args_round_trip() -> None:
    original = TranscodeArgs(
        input_path=Path("/tmp/input.mkv"),
        output_path=Path("/tmp/output.mp4"),
        filter_chain="scale=1920:1080",
        target_width=1920,
        target_height=1080,
        target_pix_fmt="yuv420p",
    )
    restored = TranscodeArgs.model_validate_json(original.model_dump_json())
    assert restored == original


def test_ffmpeg_runner_is_importable() -> None:
    # FfmpegRunner must be importable from the public package interface
    assert FfmpegRunner is not None
