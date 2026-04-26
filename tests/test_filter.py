from pathlib import Path
from unittest.mock import patch
import imagehash
from subtitles_ocr.models import Frame
from subtitles_ocr.pipeline.filter import compute_groups, HASH_DISTANCE_THRESHOLD

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
