"""Tests for Stage 12 — ASS export (ADR-0002 §3 Stage 11, ADR-0003 §4.4)."""

from __future__ import annotations

import os
from pathlib import Path

import pysubs2
import pytest

from subtitles_ocr.config import ExportConfig, PipelineGlobals
from subtitles_ocr.pipeline.animation import AnimatedEvent, AnimationAnalysisResult
from subtitles_ocr.pipeline.color import ColorExtractionResult, EventColors
from subtitles_ocr.pipeline.normalize import NormalizedEvent, NormalizeResult
from subtitles_ocr.pipeline.export import ExportStage, _classify_position


# ---------------------------------------------------------------------------
# _classify_position
# ---------------------------------------------------------------------------


def test_classify_position_bottom_centred() -> None:
    quad = [(860, 960), (1060, 960), (1060, 1020), (860, 1020)]
    assert _classify_position(quad, 1920, 1080) == "Bottom"


def test_classify_position_top_centred() -> None:
    quad = [(860, 60), (1060, 60), (1060, 120), (860, 120)]
    assert _classify_position(quad, 1920, 1080) == "Top"


def test_classify_position_off_centre_is_sign() -> None:
    quad = [(200, 950), (400, 950), (400, 1020), (200, 1020)]
    assert _classify_position(quad, 1920, 1080) == "Sign"


def test_classify_position_rotated_bottom_centred_is_sign() -> None:
    # Even with a centroid in the bottom third and horizontally centred, a
    # rotated quad is treated as Sign — regular dialogue has axis-aligned
    # quads, rotated text is a sign / overlay convention.
    quad = [(860, 960), (1060, 980), (1060, 1040), (860, 1020)]
    assert _classify_position(quad, 1920, 1080) == "Sign"


def test_classify_position_subdegree_noise_stays_bottom() -> None:
    # OCR quads have sub-degree noise on dialogue lines; the rotation
    # threshold for Sign classification must be above that.
    quad = [(860, 960), (1060, 961), (1060, 1021), (860, 1020)]
    assert _classify_position(quad, 1920, 1080) == "Bottom"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bottom_quad() -> list[tuple[int, int]]:
    # 1920x1080 frame; centroid ~ (960, 990) → bottom third, horizontally centered
    return [(860, 960), (1060, 960), (1060, 1020), (860, 1020)]


def _top_quad() -> list[tuple[int, int]]:
    # centroid ~ (960, 90) → top third, horizontally centered
    return [(860, 60), (1060, 60), (1060, 120), (860, 120)]


def _sign_quad() -> list[tuple[int, int]]:
    # centroid ~ (300, 500) → not centered, not in top/bottom third
    return [(200, 450), (400, 450), (400, 550), (200, 550)]


def _rotated_sign_quad() -> list[tuple[int, int]]:
    # TL→TR has clear non-zero angle (30 deg downward to the right)
    # angle from (0,0) to (100, 58) ≈ 30°
    return [(200, 500), (300, 558), (270, 608), (170, 550)]


def _rotated_bottom_quad() -> list[tuple[int, int]]:
    # Centroid lands in bottom-third and horizontally centered, but the top
    # edge slopes ~6° — the geometric tell-tale of a rotated overlay, not
    # plain dialogue text. Classifier should call this a Sign despite the
    # bottom-centre position.
    return [(860, 960), (1060, 970), (1060, 1030), (860, 1020)]


def _write_inputs(
    workdir: Path,
    animation: AnimationAnalysisResult,
    colors: ColorExtractionResult,
    doc: NormalizeResult,
) -> None:
    (workdir / "08_animation").mkdir(parents=True, exist_ok=True)
    (workdir / "09_color").mkdir(parents=True, exist_ok=True)
    (workdir / "11_normalize").mkdir(parents=True, exist_ok=True)
    (workdir / "08_animation" / "animation.json").write_text(
        animation.model_dump_json(), encoding="utf-8"
    )
    (workdir / "09_color" / "colors.json").write_text(
        colors.model_dump_json(), encoding="utf-8"
    )
    (workdir / "11_normalize" / "normalized.json").write_text(
        doc.model_dump_json(), encoding="utf-8"
    )


def _make_event(
    event_id: int,
    quad: list[tuple[int, int]],
    *,
    fansub_frame_start: int = 0,
    fansub_frame_end: int = 24,
    motion: dict | None = None,
    fade_in_ms: int = 0,
    fade_out_ms: int = 0,
) -> AnimatedEvent:
    return AnimatedEvent(
        event_id=event_id,
        fansub_frame_start=fansub_frame_start,
        fansub_frame_end=fansub_frame_end,
        raw_ocr_texts=[],
        raw_ocr_confidences=[],
        quads_per_frame={fansub_frame_start: quad},
        quad_median=quad,
        member_frame_indices=[fansub_frame_start],
        motion=motion,
        fade_in_ms=fade_in_ms,
        fade_out_ms=fade_out_ms,
    )


def _make_colors(
    *triples: tuple[int, tuple[int, int, int] | None, tuple[int, int, int] | None, bool],
) -> ColorExtractionResult:
    events = [
        EventColors(
            event_id=eid,
            fill_color=fill,
            outline_color=outline,
            style_supported=supported,
            stroke_width_px=2.0,
        )
        for (eid, fill, outline, supported) in triples
    ]
    return ColorExtractionResult(events=events, stats={})


def _make_doc(*items: tuple[int, str]) -> NormalizeResult:
    return NormalizeResult(
        events=[NormalizedEvent(event_id=eid, cleaned_text=text) for (eid, text) in items]
    )


# ---------------------------------------------------------------------------
# Pruned-event tolerance (ADR-0005): the normalize stage may drop events as
# OCR noise; their event_id will be absent from text_by_id while still
# present in animation/colors. Export must skip them silently, not crash.
# ---------------------------------------------------------------------------


def test_export_drops_events_shorter_than_min_duration(
    mock_globals: PipelineGlobals,
) -> None:
    # fps=24, so 1 frame ≈ 41.67 ms. With min_event_duration_ms=400 (default),
    # any event spanning fewer than 10 frames must be dropped.
    short = _make_event(0, _bottom_quad(), fansub_frame_start=0, fansub_frame_end=4)  # ~166 ms
    keep = _make_event(1, _bottom_quad(), fansub_frame_start=10, fansub_frame_end=20)  # ~416 ms
    anim = AnimationAnalysisResult(events=[short, keep], stats={})
    colors = _make_colors(
        (0, (255, 255, 255), (0, 0, 0), True),
        (1, (255, 255, 255), (0, 0, 0), True),
    )
    doc = _make_doc((0, "noise"), (1, "real"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())

    assert result.event_count == 1
    subs = pysubs2.load(result.out_path_written)
    assert len(subs) == 1
    assert subs[0].text == "real"


def test_export_min_duration_zero_disables_filter(
    mock_globals: PipelineGlobals,
) -> None:
    short = _make_event(0, _bottom_quad(), fansub_frame_start=0, fansub_frame_end=2)  # ~83 ms
    anim = AnimationAnalysisResult(events=[short], stats={})
    colors = _make_colors((0, (255, 255, 255), (0, 0, 0), True))
    doc = _make_doc((0, "short event"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig(min_event_duration_ms=0))

    assert result.event_count == 1


def test_export_skips_event_when_text_missing(
    mock_globals: PipelineGlobals,
) -> None:
    ev0 = _make_event(0, _bottom_quad())
    ev1 = _make_event(1, _bottom_quad())  # this event will be missing from doc
    ev2 = _make_event(2, _bottom_quad())
    anim = AnimationAnalysisResult(events=[ev0, ev1, ev2], stats={})
    colors = _make_colors(
        (0, (255, 255, 255), (0, 0, 0), True),
        (1, (255, 255, 255), (0, 0, 0), True),
        (2, (255, 255, 255), (0, 0, 0), True),
    )
    doc = _make_doc((0, "first"), (2, "third"))  # event_id=1 missing
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())

    assert result.event_count == 2
    subs = pysubs2.load(result.out_path_written)
    assert len(subs) == 2
    assert subs[0].text == "first"
    assert subs[1].text == "third"


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


def test_export_bottom_event_uses_bottom_style_no_inline_tag(
    mock_globals: PipelineGlobals,
) -> None:
    ev = _make_event(0, _bottom_quad())
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (255, 255, 255), (0, 0, 0), True))
    doc = _make_doc((0, "Hello world"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())

    assert result.event_count == 1
    out_path = Path(result.out_path_written)
    assert out_path.exists()
    subs = pysubs2.load(str(out_path))
    assert len(subs) == 1
    line = subs[0]
    assert line.style == "Bottom-0"
    assert line.text == "Hello world"
    assert "\\pos" not in line.text
    assert "\\frz" not in line.text


def test_export_sign_event_has_pos_and_frz_tags(
    mock_globals: PipelineGlobals,
) -> None:
    ev = _make_event(0, _rotated_sign_quad())
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (200, 100, 50), (10, 10, 10), True))
    doc = _make_doc((0, "Sign here"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    subs = pysubs2.load(str(result.out_path_written))
    assert len(subs) == 1
    line = subs[0]
    assert line.style.startswith("Sign-")
    assert "\\pos(" in line.text
    # ASS spec for \frz uses no parens: `\frzNUMBER`.
    assert "\\frz" in line.text and "\\frz(" not in line.text


def test_export_sign_style_uses_bottom_center_alignment(
    mock_globals: PipelineGlobals,
) -> None:
    # Fansub convention: Sign styles default to alignment 2 (bottom-centre)
    # so that the \pos coordinate marks the bottom-centre of the rendered
    # text. Matches the resolved alignment ref uses for sign overlays.
    ev = _make_event(0, _sign_quad())
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (255, 255, 255), (0, 0, 0), True))
    doc = _make_doc((0, "Sign here"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    subs = pysubs2.load(str(result.out_path_written))
    sign_styles = [name for name in subs.styles if name.startswith("Sign-")]
    assert sign_styles, "expected at least one Sign-* style"
    for name in sign_styles:
        assert int(subs.styles[name].alignment) == 2


def test_export_top_class_emits_pos_at_quad_top_centre(
    mock_globals: PipelineGlobals,
) -> None:
    # Top-of-screen overlays (forced translations, episode titles) render
    # off the default-dialogue baseline. Emit \pos at the top-centre of the
    # quad so the rendered text lands where it was detected; lifts intent
    # and position against ref overlays that consistently use inline \pos.
    quad = [(860, 60), (1060, 60), (1060, 120), (860, 120)]
    ev = _make_event(0, quad)
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (255, 255, 255), (0, 0, 0), True))
    doc = _make_doc((0, "Title"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    line = pysubs2.load(str(result.out_path_written))[0]
    assert line.style.startswith("Top-")
    assert "\\pos(960,60)" in line.text


def test_export_sign_pos_at_bottom_center_of_quad(
    mock_globals: PipelineGlobals,
) -> None:
    # With alignment=2, the \pos coordinate represents the bottom-centre of
    # the rendered text. The pipeline emits it at the bottom-centre of the
    # OCR quad so the rendered text lands where the OCR detected it.
    ev = _make_event(0, _sign_quad())  # x range 200..400, y range 450..550
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (255, 255, 255), (0, 0, 0), True))
    doc = _make_doc((0, "Sign here"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    line = pysubs2.load(str(result.out_path_written))[0]
    # cx = 300 (midpoint of 200..400), bottom y = 550.
    assert "\\pos(300,550)" in line.text


def test_export_sign_event_omits_frz_when_angle_below_threshold(
    mock_globals: PipelineGlobals,
) -> None:
    ev = _make_event(0, _sign_quad())  # axis-aligned: angle = 0
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (200, 100, 50), (10, 10, 10), True))
    doc = _make_doc((0, "Flat sign"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    line = pysubs2.load(str(result.out_path_written))[0]
    assert "\\pos(" in line.text
    assert "\\frz" not in line.text


# ---------------------------------------------------------------------------
# Color clustering
# ---------------------------------------------------------------------------


def test_export_two_bottoms_same_color_share_style(
    mock_globals: PipelineGlobals,
) -> None:
    ev0 = _make_event(0, _bottom_quad(), fansub_frame_start=0, fansub_frame_end=24)
    ev1 = _make_event(1, _bottom_quad(), fansub_frame_start=48, fansub_frame_end=72)
    anim = AnimationAnalysisResult(events=[ev0, ev1], stats={})
    colors = _make_colors(
        (0, (255, 255, 255), (0, 0, 0), True),
        (1, (255, 255, 255), (0, 0, 0), True),
    )
    doc = _make_doc((0, "A"), (1, "B"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    subs = pysubs2.load(str(result.out_path_written))
    styles = {line.style for line in subs}
    assert styles == {"Bottom-0"}


def test_export_two_bottoms_distant_colors_get_distinct_styles(
    mock_globals: PipelineGlobals,
) -> None:
    ev0 = _make_event(0, _bottom_quad(), fansub_frame_start=0, fansub_frame_end=24)
    ev1 = _make_event(1, _bottom_quad(), fansub_frame_start=48, fansub_frame_end=72)
    anim = AnimationAnalysisResult(events=[ev0, ev1], stats={})
    # white vs vivid red: large ΔE76, well above threshold 10
    colors = _make_colors(
        (0, (255, 255, 255), (0, 0, 0), True),
        (1, (255, 0, 0), (0, 0, 0), True),
    )
    doc = _make_doc((0, "A"), (1, "B"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    subs = pysubs2.load(str(result.out_path_written))
    styles = sorted({line.style for line in subs})
    assert styles == ["Bottom-0", "Bottom-1"]


# ---------------------------------------------------------------------------
# Animation tags
# ---------------------------------------------------------------------------


def test_export_linear_motion_emits_move_tag(mock_globals: PipelineGlobals) -> None:
    ev = _make_event(
        0,
        _sign_quad(),
        motion={"type": "linear", "start": (100, 200), "end": (500, 200)},
    )
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (255, 255, 255), (0, 0, 0), True))
    doc = _make_doc((0, "Moving"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    line = pysubs2.load(str(result.out_path_written))[0]
    assert "\\move(100,200,500,200)" in line.text


def test_export_nonlinear_flagged_motion_prepends_comment(
    mock_globals: PipelineGlobals,
) -> None:
    ev = _make_event(0, _sign_quad(), motion={"type": "nonlinear_flagged"})
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (255, 255, 255), (0, 0, 0), True))
    doc = _make_doc((0, "Wild move"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    line = pysubs2.load(str(result.out_path_written))[0]
    # The comment is in the text; pysubs2 keeps it verbatim
    assert "{!sign: animation non reconstruite!}" in line.text
    # And the dialogue text still present
    assert "Wild move" in line.text


def test_export_fade_emits_fad_tag(mock_globals: PipelineGlobals) -> None:
    ev = _make_event(0, _bottom_quad(), fade_in_ms=200, fade_out_ms=300)
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (255, 255, 255), (0, 0, 0), True))
    doc = _make_doc((0, "Faded"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    # emit_animation_fades=True opts into the animation-driven \fad tag
    # (off by default; the current detector misses real overlays and only
    # emits dialogue false positives, so it does not lift the score).
    result = ExportStage().run(
        mock_globals, ExportConfig(emit_animation_fades=True)
    )
    line = pysubs2.load(str(result.out_path_written))[0]
    assert "\\fad(200,300)" in line.text


def test_export_no_fade_when_both_zero(mock_globals: PipelineGlobals) -> None:
    ev = _make_event(0, _bottom_quad(), fade_in_ms=0, fade_out_ms=0)
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (255, 255, 255), (0, 0, 0), True))
    doc = _make_doc((0, "Plain"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    line = pysubs2.load(str(result.out_path_written))[0]
    assert "\\fad" not in line.text


# ---------------------------------------------------------------------------
# Style fallback
# ---------------------------------------------------------------------------


def test_export_style_unsupported_event_uses_default_style(
    mock_globals: PipelineGlobals,
) -> None:
    ev = _make_event(0, _bottom_quad())
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, None, None, False))
    doc = _make_doc((0, "Fallback"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    subs = pysubs2.load(str(result.out_path_written))
    line = subs[0]
    assert line.style == "Bottom-Default"
    # The default style must exist in the file's styles
    assert "Bottom-Default" in subs.styles
    default_style = subs.styles["Bottom-Default"]
    # White fill, black outline per ADR
    assert default_style.primarycolor.r == 255
    assert default_style.primarycolor.g == 255
    assert default_style.primarycolor.b == 255
    assert default_style.outlinecolor.r == 0
    assert default_style.outlinecolor.g == 0
    assert default_style.outlinecolor.b == 0


# ---------------------------------------------------------------------------
# Multi-line text
# ---------------------------------------------------------------------------


def test_export_multiline_text_becomes_hard_break(
    mock_globals: PipelineGlobals,
) -> None:
    ev = _make_event(0, _bottom_quad())
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (255, 255, 255), (0, 0, 0), True))
    doc = _make_doc((0, "Line one\nLine two"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    out_path = Path(result.out_path_written)
    # Read raw to check \N literal in the file
    raw = out_path.read_text(encoding="utf-8")
    assert "Line one\\NLine two" in raw
    # Also: pysubs2 round-trip exposes it as \N in `.text`
    line = pysubs2.load(str(out_path))[0]
    assert "\\N" in line.text


def test_export_wrap_merge_joins_with_space_not_hard_break(
    mock_globals: PipelineGlobals,
) -> None:
    # Two stacked dialogue OCR events that look like a wrapped subtitle.
    # The wrap-merge joins them into a single event whose text is a plain
    # space-separated string — the ASS renderer auto-wraps to match ref's
    # convention (ref dialogue rarely emits explicit \N for natural wraps).
    top = _make_event(
        0,
        [(860, 940), (1060, 940), (1060, 990), (860, 990)],
        fansub_frame_start=0,
        fansub_frame_end=24,
    )
    bottom = _make_event(
        1,
        [(860, 1000), (1060, 1000), (1060, 1050), (860, 1050)],
        fansub_frame_start=0,
        fansub_frame_end=24,
    )
    anim = AnimationAnalysisResult(events=[top, bottom], stats={})
    colors = _make_colors(
        (0, (255, 255, 255), (0, 0, 0), True),
        (1, (255, 255, 255), (0, 0, 0), True),
    )
    doc = _make_doc((0, "Top half"), (1, "bottom half"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    result = ExportStage().run(mock_globals, ExportConfig())
    assert result.event_count == 1
    line = pysubs2.load(str(result.out_path_written))[0]
    assert line.text == "Top half bottom half"
    assert "\\N" not in line.text


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


def test_export_atomicity_rename_failure_leaves_no_out_path(
    mock_globals: PipelineGlobals,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev = _make_event(0, _bottom_quad())
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (255, 255, 255), (0, 0, 0), True))
    doc = _make_doc((0, "Hello"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    # Sanity: out_path is initially absent.
    assert not mock_globals.out_path.exists()

    def boom(*_args, **_kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr("subtitles_ocr.pipeline.export.os.rename", boom)

    with pytest.raises(OSError, match="simulated rename failure"):
        ExportStage().run(mock_globals, ExportConfig())

    assert not mock_globals.out_path.exists()


# ---------------------------------------------------------------------------
# Pysubs2 round-trip / Script Info
# ---------------------------------------------------------------------------


def test_export_script_info_includes_playres_and_wrapstyle(
    mock_globals: PipelineGlobals,
) -> None:
    ev = _make_event(0, _bottom_quad())
    anim = AnimationAnalysisResult(events=[ev], stats={})
    colors = _make_colors((0, (255, 255, 255), (0, 0, 0), True))
    doc = _make_doc((0, "x"))
    _write_inputs(mock_globals.workdir, anim, colors, doc)

    ExportStage().run(mock_globals, ExportConfig())
    subs = pysubs2.load(str(mock_globals.out_path))
    info_lower = {k.lower(): v for k, v in subs.info.items()}
    assert info_lower.get("playresx") == str(mock_globals.fansub_width)
    assert info_lower.get("playresy") == str(mock_globals.fansub_height)
    assert info_lower.get("wrapstyle") == "0"
    assert info_lower.get("scaledborderandshadow", "").lower() == "yes"
