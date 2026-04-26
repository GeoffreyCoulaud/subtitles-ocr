import json
from pathlib import Path
from subtitles_ocr.pipeline.extract import parse_video_info, compute_frame_timestamps
from subtitles_ocr.models import VideoInfo, Frame

FFPROBE_OUTPUT = json.dumps({
    "streams": [{
        "width": 1920,
        "height": 1080,
        "r_frame_rate": "24000/1001",
    }]
})

FFPROBE_OUTPUT_INTEGER_FPS = json.dumps({
    "streams": [{
        "width": 1280,
        "height": 720,
        "r_frame_rate": "24/1",
    }]
})


def test_parse_video_info_fractional_fps():
    info = parse_video_info(FFPROBE_OUTPUT)
    assert info.width == 1920
    assert info.height == 1080
    assert abs(info.fps - 23.976) < 0.001


def test_parse_video_info_integer_fps():
    info = parse_video_info(FFPROBE_OUTPUT_INTEGER_FPS)
    assert info.fps == 24.0


def test_compute_frame_timestamps_first_is_zero():
    paths = [Path("000001.jpg"), Path("000002.jpg"), Path("000003.jpg")]
    frames = compute_frame_timestamps(paths, fps=24.0)
    assert frames[0].timestamp == 0.0
    assert frames[0].path == Path("000001.jpg")


def test_compute_frame_timestamps_spacing():
    paths = [Path(f"{i:06d}.jpg") for i in range(1, 4)]
    frames = compute_frame_timestamps(paths, fps=24.0)
    assert abs(frames[1].timestamp - 1 / 24) < 1e-6
    assert abs(frames[2].timestamp - 2 / 24) < 1e-6


def test_compute_frame_timestamps_empty():
    frames = compute_frame_timestamps([], fps=24.0)
    assert frames == []
