"""Tests for the subtitles-ocr-evaluate CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hello world
"""


@pytest.fixture
def identical_pair(tmp_path: Path) -> tuple[Path, Path]:
    ref = tmp_path / "ref.ass"
    out = tmp_path / "out.ass"
    ref.write_text(_ASS_HEADER, encoding="utf-8")
    out.write_text(_ASS_HEADER, encoding="utf-8")
    return out, ref


def test_cli_prints_human_readable_report(identical_pair: tuple[Path, Path]) -> None:
    out, ref = identical_pair
    result = subprocess.run(
        [sys.executable, "-m", "subtitles_ocr.evaluation.cli",
         "--output", str(out), "--reference", str(ref), "--fps", "24"],
        capture_output=True, text=True, check=True,
    )
    assert "final" in result.stdout.lower()
    assert "1.0" in result.stdout or "1.00" in result.stdout


def test_cli_emits_json_with_flag(identical_pair: tuple[Path, Path]) -> None:
    out, ref = identical_pair
    result = subprocess.run(
        [sys.executable, "-m", "subtitles_ocr.evaluation.cli",
         "--output", str(out), "--reference", str(ref), "--fps", "24", "--json"],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["final"] == 1.0
    assert payload["n_matched"] == 1


def test_cli_accepts_fractional_fps(identical_pair: tuple[Path, Path]) -> None:
    out, ref = identical_pair
    result = subprocess.run(
        [sys.executable, "-m", "subtitles_ocr.evaluation.cli",
         "--output", str(out), "--reference", str(ref), "--fps", "24000/1001", "--json"],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["final"] == 1.0


def test_cli_exits_nonzero_on_missing_file(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "subtitles_ocr.evaluation.cli",
         "--output", str(tmp_path / "missing.ass"),
         "--reference", str(tmp_path / "also-missing.ass"),
         "--fps", "24"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
