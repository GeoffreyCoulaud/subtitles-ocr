from fractions import Fraction

import pytest

from subtitles_ocr.timing import frame_to_ms, ms_to_frame

FPS_23976 = Fraction(24000, 1001)
FPS_24 = Fraction(24, 1)
FRAME_RANGE = range(0, 35000, 137)


def test_frame_to_ms_zero():
    assert frame_to_ms(0, FPS_23976) == 0


def test_round_trip_23976_all_frames():
    for i in FRAME_RANGE:
        assert ms_to_frame(frame_to_ms(i, FPS_23976), FPS_23976) == i, (
            f"Round-trip failed at frame {i}"
        )


def test_round_trip_24_all_frames():
    for i in FRAME_RANGE:
        assert ms_to_frame(frame_to_ms(i, FPS_24), FPS_24) == i, (
            f"Round-trip failed at frame {i}"
        )


def test_footgun_float_vs_fraction_at_frame_35000():
    """
    float(23.976) is not equal to Fraction(24000, 1001): 23.976 is a decimal
    approximation with IEEE-754 rounding, while 24000/1001 is the exact NTSC
    ratio.  At frame 35000 the accumulated error causes frame_to_ms to return a
    different millisecond value.  The Fraction path stays within the
    half-integer rounding window (bounded drift ≤ 0.5 ms); the float path
    crosses it and lands 1 ms off.
    """
    frame_idx = 35000
    fps_fraction = Fraction(24000, 1001)
    fps_float = 23.976  # the footgun: not the same rational number

    ms_fraction = frame_to_ms(frame_idx, fps_fraction)

    # Reproduce what naïve float code would compute:
    ms_float_naive = int(round(frame_idx * 1000 / fps_float))

    assert ms_fraction != ms_float_naive, (
        f"Expected Fraction and float paths to produce different ms at frame "
        f"{frame_idx}, but both gave {ms_fraction}. The footgun demonstration "
        "is no longer visible — check the frame index or float constant."
    )
    # The Fraction result is correct: verify it is consistent with exact rational
    # arithmetic (the exact value is frame_idx * 1000 / fps_fraction ms).
    exact_ms = frame_idx * 1000 / fps_fraction  # a Fraction, exact
    assert abs(ms_fraction - exact_ms) < Fraction(1, 2), (
        f"Fraction path ms={ms_fraction} is more than 0.5 ms from exact "
        f"value {exact_ms}."
    )
