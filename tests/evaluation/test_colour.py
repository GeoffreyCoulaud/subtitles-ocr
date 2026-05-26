"""Tests for sRGB→LAB conversion and ΔE76 colour scoring (ADR-0006 §5.7)."""

from __future__ import annotations

import math

import pytest

from subtitles_ocr.evaluation._colour import (
    colour_score,
    delta_e_76,
    srgb_to_lab,
)


def test_white_srgb_maps_to_l_100() -> None:
    L, a, b = srgb_to_lab((255, 255, 255))
    assert L == pytest.approx(100.0, abs=0.5)
    assert a == pytest.approx(0.0, abs=0.5)
    assert b == pytest.approx(0.0, abs=0.5)


def test_black_srgb_maps_to_l_0() -> None:
    L, a, b = srgb_to_lab((0, 0, 0))
    assert L == pytest.approx(0.0, abs=0.5)


def test_red_srgb_maps_to_expected_lab() -> None:
    # Reference values from a standard sRGB→LAB calculator (D65, observer 2°).
    L, a, b = srgb_to_lab((255, 0, 0))
    assert L == pytest.approx(53.24, abs=0.5)
    assert a == pytest.approx(80.09, abs=0.5)
    assert b == pytest.approx(67.20, abs=0.5)


def test_delta_e_76_is_zero_for_identical_colours() -> None:
    lab = (50.0, 0.0, 0.0)
    assert delta_e_76(lab, lab) == pytest.approx(0.0)


def test_delta_e_76_matches_euclidean_distance() -> None:
    a = (50.0, 10.0, 20.0)
    b = (40.0, 14.0, 17.0)
    expected = math.sqrt(10**2 + 4**2 + 3**2)
    assert delta_e_76(a, b) == pytest.approx(expected)


def test_colour_score_is_one_for_imperceptible_difference() -> None:
    assert colour_score((255, 255, 255), (255, 255, 255)) == pytest.approx(1.0)


def test_colour_score_is_zero_for_clearly_different_colours() -> None:
    # Pure red vs pure cyan — ΔE76 well above the d_max threshold.
    assert colour_score((255, 0, 0), (0, 255, 255)) == pytest.approx(0.0)


def test_colour_score_decays_smoothly_between_thresholds() -> None:
    # Two close-but-not-identical greys.
    s = colour_score((100, 100, 100), (120, 120, 120))
    assert 0.0 < s < 1.0
