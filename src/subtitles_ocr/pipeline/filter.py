import imagehash
from pathlib import Path
from PIL import Image
from subtitles_ocr.models import Frame, FrameGroup

HASH_DISTANCE_THRESHOLD = 10


def compute_hash(frame_path: Path) -> imagehash.ImageHash:
    with Image.open(frame_path) as img:
        return imagehash.phash(img)


def compute_groups(
    frames: list[Frame],
    threshold: int = HASH_DISTANCE_THRESHOLD,
) -> list[FrameGroup]:
    if not frames:
        return []

    groups: list[FrameGroup] = []
    group_start = frames[0]
    group_end = frames[0]
    group_hash = compute_hash(frames[0].path)

    for frame in frames[1:]:
        frame_hash = compute_hash(frame.path)
        if frame_hash - group_hash <= threshold:
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
