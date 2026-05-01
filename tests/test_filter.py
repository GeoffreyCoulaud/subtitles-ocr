from pathlib import Path
from unittest.mock import patch
import imagehash
from PIL import Image
from subtitles_ocr.models import Frame
from subtitles_ocr.pipeline.filter import (
    compute_groups,
    compute_hash,
    SUBTITLE_STRIP_RATIO,
)

HASH_A = imagehash.hex_to_hash("0" * 16)
HASH_B = imagehash.hex_to_hash("f" * 16)


def _frames(*timestamps: float) -> list[Frame]:
    return [Frame(path=Path(f"{i:06d}.jpg"), timestamp=t) for i, t in enumerate(timestamps, 1)]


def test_single_frame_is_one_group():
    frames = _frames(0.0)
    with patch("subtitles_ocr.pipeline.filter.compute_hash", return_value=HASH_A):
        groups = compute_groups(frames)
    assert len(groups) == 1
    assert groups[0].start_time == 0.0
    assert groups[0].end_time == 0.0


def test_identical_frames_form_one_group():
    frames = _frames(0.0, 0.042, 0.083)
    with patch("subtitles_ocr.pipeline.filter.compute_hash", return_value=HASH_A):
        groups = compute_groups(frames)
    assert len(groups) == 1
    assert groups[0].start_time == 0.0
    assert groups[0].end_time == 0.083
    assert groups[0].frame == Path("000001.jpg")


def test_different_frames_form_separate_groups():
    frames = _frames(0.0, 1.0)
    hashes = [HASH_A, HASH_B]
    with patch("subtitles_ocr.pipeline.filter.compute_hash", side_effect=hashes):
        groups = compute_groups(frames)
    assert len(groups) == 2
    assert groups[0].start_time == 0.0
    assert groups[0].end_time == 0.0
    assert groups[1].start_time == 1.0
    assert groups[1].end_time == 1.0


def test_empty_frames_returns_empty():
    groups = compute_groups([])
    assert groups == []


def test_representative_frame_is_first_of_group():
    frames = _frames(0.0, 0.042, 0.083)
    with patch("subtitles_ocr.pipeline.filter.compute_hash", return_value=HASH_A):
        groups = compute_groups(frames)
    assert groups[0].frame == Path("000001.jpg")


def test_compute_hash_sensitive_to_strip_changes(tmp_path):
    frame_a = Image.new("RGB", (100, 100), color=(255, 255, 255))
    frame_b = Image.new("RGB", (100, 100), color=(255, 255, 255))
    black_strip = Image.new("RGB", (100, 20), color=(0, 0, 0))
    frame_b.paste(black_strip, (0, 80))  # replace bottom 20px (bottom 20% of 100px)
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    frame_a.save(path_a)
    frame_b.save(path_b)
    assert compute_hash(path_a) != compute_hash(path_b)


def test_hash_distance_threshold_controls_grouping():
    frames = _frames(0.0, 1.0)
    hashes = [HASH_A, HASH_B]
    with patch("subtitles_ocr.pipeline.filter.compute_hash", side_effect=hashes):
        tight = compute_groups(frames, hash_distance=0)
    with patch("subtitles_ocr.pipeline.filter.compute_hash", side_effect=hashes):
        loose = compute_groups(frames, hash_distance=64)
    assert len(tight) == 2
    assert len(loose) == 1


def test_compute_hash_ignores_middle_changes(tmp_path):
    frame_a = Image.new("RGB", (100, 100), color=(255, 255, 255))
    frame_b = Image.new("RGB", (100, 100), color=(255, 255, 255))
    black_middle = Image.new("RGB", (100, 60), color=(0, 0, 0))
    frame_b.paste(black_middle, (0, 20))  # rows 20–79, between top 20% and bottom 20%
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    frame_a.save(path_a)
    frame_b.save(path_b)
    assert compute_hash(path_a) == compute_hash(path_b)
