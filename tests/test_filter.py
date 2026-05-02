from pathlib import Path
from unittest.mock import patch
from PIL import Image
from subtitles_ocr.models import Frame
from subtitles_ocr.pipeline.filter import (
    compute_groups,
    compute_edge_map,
    edge_diff,
    SUBTITLE_STRIP_RATIO,
)

_SIZE = (100, 40)
EDGES_A = Image.new("L", _SIZE, 0)
EDGES_B = Image.new("L", _SIZE, 255)


def _frames(*timestamps: float) -> list[Frame]:
    return [Frame(path=Path(f"{i:06d}.jpg"), timestamp=t) for i, t in enumerate(timestamps, 1)]


def test_single_frame_is_one_group():
    frames = _frames(0.0)
    with patch("subtitles_ocr.pipeline.filter.compute_edge_map", return_value=EDGES_A):
        groups = compute_groups(frames)
    assert len(groups) == 1
    assert groups[0].start_time == 0.0
    assert groups[0].end_time == 0.0


def test_identical_frames_form_one_group():
    frames = _frames(0.0, 0.042, 0.083)
    with patch("subtitles_ocr.pipeline.filter.compute_edge_map", return_value=EDGES_A):
        groups = compute_groups(frames)
    assert len(groups) == 1
    assert groups[0].start_time == 0.0
    assert groups[0].end_time == 0.083
    assert groups[0].frame == Path("000001.jpg")


def test_different_frames_form_separate_groups():
    frames = _frames(0.0, 1.0)
    edge_maps = [EDGES_A, EDGES_B]
    with patch("subtitles_ocr.pipeline.filter.compute_edge_map", side_effect=edge_maps):
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
    with patch("subtitles_ocr.pipeline.filter.compute_edge_map", return_value=EDGES_A):
        groups = compute_groups(frames)
    assert groups[0].frame == Path("000001.jpg")


def test_diff_threshold_controls_grouping():
    frames = _frames(0.0, 1.0)
    # EDGES_A vs EDGES_B: diff = 255 per pixel
    edge_maps_tight = [EDGES_A, EDGES_B]
    edge_maps_loose = [EDGES_A, EDGES_B]
    with patch("subtitles_ocr.pipeline.filter.compute_edge_map", side_effect=edge_maps_tight):
        tight = compute_groups(frames, diff_threshold=100.0)
    with patch("subtitles_ocr.pipeline.filter.compute_edge_map", side_effect=edge_maps_loose):
        loose = compute_groups(frames, diff_threshold=300.0)
    assert len(tight) == 2
    assert len(loose) == 1


def test_compute_edge_map_sensitive_to_strip_changes(tmp_path):
    frame_a = Image.new("RGB", (100, 100), color=(255, 255, 255))
    frame_b = Image.new("RGB", (100, 100), color=(255, 255, 255))
    black_strip = Image.new("RGB", (100, 20), color=(0, 0, 0))
    frame_b.paste(black_strip, (0, 80))  # bottom 20% of 100px
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    frame_a.save(path_a)
    frame_b.save(path_b)
    assert edge_diff(compute_edge_map(path_a), compute_edge_map(path_b)) > 0


def test_compute_edge_map_ignores_middle_changes(tmp_path):
    frame_a = Image.new("RGB", (100, 100), color=(255, 255, 255))
    frame_b = Image.new("RGB", (100, 100), color=(255, 255, 255))
    black_middle = Image.new("RGB", (100, 60), color=(0, 0, 0))
    frame_b.paste(black_middle, (0, 20))  # rows 20–79, between top 20% and bottom 20%
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    frame_a.save(path_a)
    frame_b.save(path_b)
    assert edge_diff(compute_edge_map(path_a), compute_edge_map(path_b)) == 0
