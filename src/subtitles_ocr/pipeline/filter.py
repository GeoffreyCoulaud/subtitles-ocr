import imagehash
from pathlib import Path
from typing import Iterable
from PIL import Image
from subtitles_ocr.models import Frame, FrameGroup

SUBTITLE_STRIP_RATIO = 0.20


def compute_hash(frame_path: Path) -> imagehash.ImageHash:
    with Image.open(frame_path) as img:
        w, h = img.size
        strip_h = round(h * SUBTITLE_STRIP_RATIO)
        top = img.crop((0, 0, w, strip_h))
        bottom = img.crop((0, h - strip_h, w, h))
        combined = Image.new(img.mode, (w, strip_h * 2))
        combined.paste(top, (0, 0))
        combined.paste(bottom, (0, strip_h))
        return imagehash.phash(combined)


def compute_groups(
    frames: Iterable[Frame],
    hash_distance: int = 10,
) -> list[FrameGroup]:
    frames_iter = iter(frames)
    first = next(frames_iter, None)
    if first is None:
        return []

    groups: list[FrameGroup] = []
    group_start = first
    group_end = first
    group_hash = compute_hash(first.path)

    for frame in frames_iter:
        frame_hash = compute_hash(frame.path)
        if frame_hash - group_hash <= hash_distance:
            group_end = frame
        else:
            groups.append(FrameGroup(
                start_time=group_start.timestamp,
                end_time=group_end.timestamp,
                frame=group_start.path,
            ))
            group_start = frame
            group_end = frame
            group_hash = frame_hash

    groups.append(FrameGroup(
        start_time=group_start.timestamp,
        end_time=group_end.timestamp,
        frame=group_start.path,
    ))
    return groups
