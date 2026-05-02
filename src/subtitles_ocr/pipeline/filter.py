from pathlib import Path
from typing import Iterable
from PIL import Image, ImageChops, ImageFilter
from subtitles_ocr.models import Frame, FrameGroup

SUBTITLE_STRIP_RATIO = 0.20
EDGE_DIFF_THRESHOLD = 8.0


def compute_edge_map(frame_path: Path) -> Image.Image:
    with Image.open(frame_path) as img:
        w, h = img.size
        strip_h = round(h * SUBTITLE_STRIP_RATIO)
        top = img.crop((0, 0, w, strip_h))
        bottom = img.crop((0, h - strip_h, w, h))
        combined = Image.new(img.mode, (w, strip_h * 2))
        combined.paste(top, (0, 0))
        combined.paste(bottom, (0, strip_h))
        return combined.convert("L").filter(ImageFilter.FIND_EDGES)


def edge_diff(map_a: Image.Image, map_b: Image.Image) -> float:
    diff = ImageChops.difference(map_a, map_b)
    pixels = diff.get_flattened_data()
    return sum(pixels) / len(pixels)


def compute_groups(
    frames: Iterable[Frame],
    diff_threshold: float = EDGE_DIFF_THRESHOLD,
) -> list[FrameGroup]:
    frames_iter = iter(frames)
    first = next(frames_iter, None)
    if first is None:
        return []

    groups: list[FrameGroup] = []
    group_start = first
    group_end = first
    group_edges = compute_edge_map(first.path)

    for frame in frames_iter:
        frame_edges = compute_edge_map(frame.path)
        if edge_diff(group_edges, frame_edges) <= diff_threshold:
            group_end = frame
        else:
            groups.append(FrameGroup(
                start_time=group_start.timestamp,
                end_time=group_end.timestamp,
                frame=group_start.path,
            ))
            group_start = frame
            group_end = frame
            group_edges = frame_edges

    groups.append(FrameGroup(
        start_time=group_start.timestamp,
        end_time=group_end.timestamp,
        frame=group_start.path,
    ))
    return groups
