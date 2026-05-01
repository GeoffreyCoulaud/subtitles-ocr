import json
from pathlib import Path
from unittest.mock import MagicMock
from subtitles_ocr.models import FrameGroup, SubtitleElement
from subtitles_ocr.pipeline.analyze import analyze_group, parse_elements

VALID_ELEMENT = {
    "text": "Bonjour",
    "style": "regular",
    "color": "white",
    "border_color": "black",
    "position": "bottom",
    "alignment": "center",
}


def _group() -> FrameGroup:
    return FrameGroup(start_time=1.0, end_time=2.5, frame=Path("frames/000024.jpg"))


def test_parse_elements_valid_response():
    raw = json.dumps([VALID_ELEMENT])
    elements = parse_elements(raw)
    assert len(elements) == 1
    assert elements[0].text == "Bonjour"
    assert elements[0].color == "#FFFFFF"


def test_parse_elements_empty_array():
    elements = parse_elements("[]")
    assert elements == []


def test_parse_elements_invalid_json_returns_empty():
    elements = parse_elements("not json at all")
    assert elements == []


def test_parse_elements_multiple():
    raw = json.dumps([VALID_ELEMENT, {**VALID_ELEMENT, "text": "Au revoir"}])
    elements = parse_elements(raw)
    assert len(elements) == 2
    assert elements[1].text == "Au revoir"


def test_parse_elements_partial_invalid_keeps_valid():
    invalid_element = {"text": "Bad"}  # missing required fields: style, color, etc.
    raw = json.dumps([VALID_ELEMENT, invalid_element])
    elements = parse_elements(raw)
    assert len(elements) == 1
    assert elements[0].text == "Bonjour"


def test_analyze_group_returns_correct_timing():
    client = MagicMock()
    client.analyze.return_value = "[]"
    analysis = analyze_group(_group(), client, prompt="p")
    assert analysis.start_time == 1.0
    assert analysis.end_time == 2.5


def test_analyze_group_parses_elements():
    client = MagicMock()
    client.analyze.return_value = json.dumps([VALID_ELEMENT])
    analysis = analyze_group(_group(), client, prompt="p")
    assert len(analysis.elements) == 1
    assert analysis.elements[0].text == "Bonjour"
