"""sRGB→CIE-LAB conversion and ΔE76 colour scoring (ADR-0006 §5.7)."""

from __future__ import annotations

import math

# D65 reference white (observer 2°), Y normalised to 100.
_REF_X = 95.047
_REF_Y = 100.000
_REF_Z = 108.883


def _srgb_channel_to_linear(c8: int) -> float:
    """8-bit sRGB → linear [0, 1]."""
    c = c8 / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _linear_rgb_to_xyz(r: float, g: float, b: float) -> tuple[float, float, float]:
    # sRGB D65 matrix (linear RGB → XYZ, Y normalised to 100).
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) * 100.0
    y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) * 100.0
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) * 100.0
    return (x, y, z)


def _f(t: float) -> float:
    delta = 6.0 / 29.0
    if t > delta**3:
        return t ** (1.0 / 3.0)
    return t / (3 * delta**2) + 4.0 / 29.0


def srgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (_srgb_channel_to_linear(c) for c in rgb)
    x, y, z = _linear_rgb_to_xyz(r, g, b)
    fx = _f(x / _REF_X)
    fy = _f(y / _REF_Y)
    fz = _f(z / _REF_Z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)
    return (L, a, b_lab)


def delta_e_76(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def colour_score(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    d_max: float = 25.0,
) -> float:
    lab_a = srgb_to_lab(a)
    lab_b = srgb_to_lab(b)
    dE = delta_e_76(lab_a, lab_b)
    if dE <= 1.0:
        return 1.0
    if dE >= d_max:
        return 0.0
    return 1.0 - (dE - 1.0) / (d_max - 1.0)
