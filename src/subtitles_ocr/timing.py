from fractions import Fraction


def frame_to_ms(frame_idx: int, fps: Fraction) -> int:
    return int(round(frame_idx * 1000 / fps))


def ms_to_frame(ms: int, fps: Fraction) -> int:
    return int(round(ms * fps / 1000))
