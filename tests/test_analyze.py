import json
import logging
from pathlib import Path
from unittest.mock import MagicMock
from subtitles_ocr.models import FrameGroup, SubtitleElement
from subtitles_ocr.pipeline.analyze import analyze_group, analyze_groups, parse_elements

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


def test_analyze_group_logs_raw_at_debug(caplog):
    client = MagicMock()
    client.analyze.return_value = "[]"
    with caplog.at_level(logging.DEBUG, logger="subtitles_ocr.pipeline.analyze"):
        analyze_group(_group(), client, prompt="p")
    assert any("raw →" in r.message and r.levelno == logging.DEBUG for r in caplog.records)


def test_analyze_group_logs_no_elements(caplog):
    client = MagicMock()
    client.analyze.return_value = "[]"
    with caplog.at_level(logging.INFO, logger="subtitles_ocr.pipeline.analyze"):
        analyze_group(_group(), client, prompt="p")
    info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_msgs) == 1
    assert "(no elements)" in info_msgs[0]


def test_analyze_group_logs_one_info_line_per_element(caplog):
    client = MagicMock()
    client.analyze.return_value = json.dumps([VALID_ELEMENT, {**VALID_ELEMENT, "text": "Au revoir"}])
    with caplog.at_level(logging.INFO, logger="subtitles_ocr.pipeline.analyze"):
        analyze_group(_group(), client, prompt="p")
    info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_msgs) == 2
    assert "Bonjour" in info_msgs[0]
    assert "Au revoir" in info_msgs[1]


def test_analyze_groups_skips_vlm_when_no_text():
    client = MagicMock()
    group = _group()
    result = list(analyze_groups([group], [False], client, "p", workers=1))
    assert len(result) == 1
    assert result[0].elements == []
    assert result[0].start_time == group.start_time
    assert result[0].end_time == group.end_time
    client.analyze.assert_not_called()


def test_analyze_groups_calls_vlm_when_has_text():
    client = MagicMock()
    client.analyze.return_value = json.dumps([VALID_ELEMENT])
    group = _group()
    result = list(analyze_groups([group], [True], client, "p", workers=1))
    assert len(result) == 1
    assert len(result[0].elements) == 1
    assert result[0].elements[0].text == "Bonjour"
    client.analyze.assert_called_once()


def test_analyze_groups_returns_empty_on_runtime_error():
    client = MagicMock()
    client.analyze.side_effect = RuntimeError("model failed")
    group = _group()
    result = list(analyze_groups([group], [True], client, "p", workers=1))
    assert len(result) == 1
    assert result[0].elements == []


def test_analyze_groups_logs_warning_on_runtime_error(caplog):
    client = MagicMock()
    client.analyze.side_effect = RuntimeError("model failed")
    group = _group()
    with caplog.at_level(logging.WARNING, logger="subtitles_ocr.pipeline.analyze"):
        list(analyze_groups([group], [True], client, "p", workers=1))
    assert any("failed" in r.message and r.levelno == logging.WARNING for r in caplog.records)


def test_analyze_groups_preserves_order():
    client = MagicMock()
    client.analyze.return_value = "[]"
    groups = [
        FrameGroup(start_time=1.0, end_time=2.0, frame=Path("frames/a.jpg")),
        FrameGroup(start_time=3.0, end_time=4.0, frame=Path("frames/b.jpg")),
        FrameGroup(start_time=5.0, end_time=6.0, frame=Path("frames/c.jpg")),
    ]
    result = list(analyze_groups(groups, [True, True, True], client, "p", workers=3))
    assert [r.start_time for r in result] == [1.0, 3.0, 5.0]


def test_analyze_groups_mixed_filter():
    client = MagicMock()
    client.analyze.return_value = json.dumps([VALID_ELEMENT])
    groups = [_group(), _group(), _group()]
    result = list(analyze_groups(groups, [False, True, False], client, "p", workers=1))
    assert len(result) == 3
    assert result[0].elements == []
    assert len(result[1].elements) == 1
    assert result[2].elements == []
    client.analyze.assert_called_once()


def test_analyze_groups_empty_returns_empty():
    client = MagicMock()
    result = list(analyze_groups([], [], client, "p", workers=1))
    assert result == []
