"""Tests for ASS override tag parsing (ADR-0006 §5.7-5.9)."""

from __future__ import annotations

from subtitles_ocr.evaluation._tags import ParsedEvent, parse_event_text


def test_plain_text_extracted_without_tags() -> None:
    parsed = parse_event_text(r"{\i1}Hello{\i0} world")
    assert parsed.plain_text == "Hello world"


def test_italic_detected() -> None:
    assert parse_event_text(r"{\i1}foo").italic is True
    assert parse_event_text(r"foo").italic is False


def test_bold_underline_strikeout() -> None:
    parsed = parse_event_text(r"{\b1\u1\s1}foo")
    assert parsed.bold is True
    assert parsed.underline is True
    assert parsed.strikeout is True


def test_primary_colour_parsed_as_rgb_tuple() -> None:
    # &H00FFAA22& -- alpha 00, BGR is 22 AA FF -> RGB is FF AA 22
    parsed = parse_event_text(r"{\c&H0022AAFF&}foo")
    assert parsed.primary_colour == (0xFF, 0xAA, 0x22)


def test_outline_colour_parsed() -> None:
    parsed = parse_event_text(r"{\3c&H00000000&}foo")
    assert parsed.outline_colour == (0x00, 0x00, 0x00)


def test_font_size_parsed() -> None:
    assert parse_event_text(r"{\fs42}foo").font_size == 42.0


def test_rotation_z_parsed() -> None:
    assert parse_event_text(r"{\frz15.5}foo").rotation_z == 15.5


def test_alignment_parsed() -> None:
    assert parse_event_text(r"{\an8}foo").alignment == 8


def test_pos_parsed() -> None:
    assert parse_event_text(r"{\pos(960,1040)}foo").pos == (960.0, 1040.0)


def test_move_parsed_without_timing() -> None:
    parsed = parse_event_text(r"{\move(100,200,300,400)}foo")
    assert parsed.move == (100.0, 200.0, 300.0, 400.0, None, None)


def test_move_parsed_with_timing() -> None:
    parsed = parse_event_text(r"{\move(100,200,300,400,500,1500)}foo")
    assert parsed.move == (100.0, 200.0, 300.0, 400.0, 500, 1500)


def test_fade_parsed() -> None:
    assert parse_event_text(r"{\fad(200,300)}foo").fade == (200, 300)


def test_multiple_override_blocks_combined() -> None:
    parsed = parse_event_text(r"{\an8}{\pos(960,100)}{\c&H0000FF00&}Sign")
    assert parsed.alignment == 8
    assert parsed.pos == (960.0, 100.0)
    assert parsed.primary_colour == (0x00, 0xFF, 0x00)
    assert parsed.plain_text == "Sign"


def test_default_state_for_event_without_tags() -> None:
    parsed = parse_event_text("Just words")
    assert parsed.italic is False
    assert parsed.bold is False
    assert parsed.primary_colour is None
    assert parsed.font_size is None
    assert parsed.alignment is None
    assert parsed.pos is None
    assert parsed.move is None
    assert parsed.fade is None
    assert parsed.plain_text == "Just words"


def test_line_break_preserved_in_plain_text() -> None:
    parsed = parse_event_text(r"line one\Nline two")
    assert parsed.plain_text == r"line one\Nline two"
