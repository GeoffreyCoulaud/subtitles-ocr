"""Integration tests for the top-level score() function (ADR-0006 §6, §7)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from subtitles_ocr.evaluation.report import default_weights
from subtitles_ocr.evaluation.score import score


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
"""


def _write_ass(tmp_path: Path, name: str, events: list[tuple[str, str, str]]) -> Path:
    body = _ASS_HEADER + "".join(
        f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
        for start, end, text in events
    )
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_score_identical_files_returns_one(tmp_path: Path) -> None:
    events = [
        ("0:00:01.00", "0:00:02.00", "Hello world"),
        ("0:00:03.00", "0:00:04.00", "Goodbye"),
    ]
    ref = _write_ass(tmp_path, "ref.ass", events)
    out = _write_ass(tmp_path, "out.ass", events)
    report = score(out, ref, weights=default_weights(), fps=Fraction(24))
    assert report.final == 1.0
    assert report.n_matched == 2


def test_score_raises_on_empty_reference(tmp_path: Path) -> None:
    ref = _write_ass(tmp_path, "ref.ass", [])
    out = _write_ass(tmp_path, "out.ass", [("0:00:01.00", "0:00:02.00", "Hello")])
    with pytest.raises(ValueError):
        score(out, ref, weights=default_weights(), fps=Fraction(24))


def test_score_no_matches_yields_zero_recall(tmp_path: Path) -> None:
    ref = _write_ass(tmp_path, "ref.ass", [("0:00:01.00", "0:00:02.00", "Hello")])
    out = _write_ass(tmp_path, "out.ass", [("0:00:10.00", "0:00:11.00", "Goodbye")])
    report = score(out, ref, weights=default_weights(), fps=Fraction(24))
    assert report.sub_scores["recall"] == 0.0
    assert report.sub_scores["precision"] == 0.0
    assert report.n_matched == 0


def test_score_null_subscores_excluded_from_weighted_sum(tmp_path: Path) -> None:
    # No \fad anywhere -> fade is None and excluded from the weighted sum.
    events = [("0:00:01.00", "0:00:02.00", "Hello")]
    ref = _write_ass(tmp_path, "ref.ass", events)
    out = _write_ass(tmp_path, "out.ass", events)
    report = score(out, ref, weights=default_weights(), fps=Fraction(24))
    assert report.sub_scores["fade"] is None
    assert "fade" not in report.effective_weights or report.effective_weights["fade"] == 0


def test_score_text_difference_reduces_final(tmp_path: Path) -> None:
    ref = _write_ass(tmp_path, "ref.ass", [("0:00:01.00", "0:00:02.00", "Hello world")])
    out = _write_ass(tmp_path, "out.ass", [("0:00:01.00", "0:00:02.00", "Hxllo wxrld")])
    report = score(out, ref, weights=default_weights(), fps=Fraction(24))
    assert 0.5 < report.final < 1.0
    assert report.sub_scores["text_plain"] < 1.0
