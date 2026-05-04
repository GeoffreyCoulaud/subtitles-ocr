# tests/test_analyze.py
import json
import logging
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from subtitles_ocr.models import FrameGroup, SubtitleElement
from subtitles_ocr.pipeline.analyze import analyze_group, analyze_groups, parse_elements
from subtitles_ocr.pipeline.retry import RetryConfig

VALID_ELEMENT = {
    "text": "Bonjour",
    "style": "regular",
    "color": "white",
    "position": "bottom",
}

WRAPPED_VALID = json.dumps({"subtitles": [VALID_ELEMENT]})
WRAPPED_EMPTY = '{"subtitles": []}'


def _no_retry() -> RetryConfig:
    return RetryConfig(max_attempts=1)


def _group() -> FrameGroup:
    return FrameGroup(start_time=1.0, end_time=2.5, frame=Path("frames/000024.jpg"))


# --- parse_elements ---

def test_parse_elements_valid_response():
    elements = parse_elements(WRAPPED_VALID)
    assert len(elements) == 1
    assert elements[0].text == "Bonjour"
    assert elements[0].color == "#FFFFFF"


def test_parse_elements_empty_subtitles_list():
    assert parse_elements(WRAPPED_EMPTY) == []


def test_parse_elements_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_elements("not json at all")


def test_parse_elements_multiple():
    raw = json.dumps({"subtitles": [VALID_ELEMENT, {**VALID_ELEMENT, "text": "Au revoir"}]})
    elements = parse_elements(raw)
    assert len(elements) == 2
    assert elements[1].text == "Au revoir"


def test_parse_elements_array_input_raises():
    """A bare JSON array is not the expected object format — raise so retry can trigger."""
    with pytest.raises(ValueError, match="expected JSON object"):
        parse_elements(json.dumps([VALID_ELEMENT]))


def test_parse_elements_unexpected_type_raises():
    with pytest.raises(ValueError, match="expected JSON"):
        parse_elements('"just a string"')


def test_parse_elements_missing_subtitles_key_raises():
    """A non-empty dict without 'subtitles' key is a format error — raise for retry."""
    with pytest.raises(ValueError, match="missing.*subtitles"):
        parse_elements(json.dumps({"text": "oops"}))


def test_parse_elements_subtitles_not_list_raises():
    with pytest.raises(ValueError, match="subtitles.*list"):
        parse_elements(json.dumps({"subtitles": "not a list"}))


def test_parse_elements_any_invalid_item_raises():
    """Any invalid item in 'subtitles' is a format error — raise so retry can trigger."""
    raw = json.dumps({"subtitles": [VALID_ELEMENT, {"text": "Bad", "style": "superboldbig"}]})
    with pytest.raises(ValueError, match="invalid item"):
        parse_elements(raw)


def test_parse_elements_all_subtitles_items_invalid_raises():
    """When every item in 'subtitles' fails schema validation, raise so retry can trigger."""
    garbage_item = {"]0E@#@$&": "#)8#-.:"}
    raw = json.dumps({"subtitles": [garbage_item]})
    with pytest.raises(ValueError, match="invalid item"):
        parse_elements(raw)


def test_parse_elements_garbage_dict_raises():
    """Garbage dict without 'subtitles' key raises for retry (the real-world gibberish pattern)."""
    garbage = {"]0E@#@$&,FB8$.-B2=A3F766=E+*9*)?2AC@3#1F9B.0": "#)8#-.:"}
    with pytest.raises(ValueError, match="missing.*subtitles"):
        parse_elements(json.dumps(garbage))


def test_parse_elements_empty_object_returns_empty():
    """Empty dict {} is the model's fallback empty signal — treat as no subtitles, no retry."""
    assert parse_elements("{}") == []


def test_parse_elements_accepts_four_field_item():
    """After schema simplification, border_color and alignment are not required."""
    raw = json.dumps({"subtitles": [{"text": "Bonjour", "style": "regular", "color": "white", "position": "bottom"}]})
    elements = parse_elements(raw)
    assert len(elements) == 1
    assert elements[0].text == "Bonjour"


def test_parse_elements_accepts_text_only_item():
    """text is the only required field — missing style/color/position use defaults."""
    raw = json.dumps({"subtitles": [{"text": "Je ne devrais plus le lire, non ?"}]})
    elements = parse_elements(raw)
    assert len(elements) == 1
    assert elements[0].text == "Je ne devrais plus le lire, non ?"
    assert elements[0].style == "regular"
    assert elements[0].color == "#FFFFFF"
    assert elements[0].position == "bottom"


def test_parse_elements_strips_json_code_fence():
    """Model sometimes wraps output in ```json...``` — strip silently before parsing."""
    raw = '```json\n{"subtitles": []}\n```'
    assert parse_elements(raw) == []


def test_parse_elements_strips_bare_code_fence():
    raw = '```\n{"subtitles": []}\n```'
    assert parse_elements(raw) == []


def test_parse_elements_does_not_strip_unclosed_code_fence():
    """Prefix without matching suffix must not be partially stripped — raise JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        parse_elements('```json\n{"subtitles": []}')  # no closing ```


def test_analyze_groups_does_not_retry_empty_object():
    """When model returns {}, accept it as no-subtitle frame on the first attempt; do not retry."""
    client = MagicMock()
    client.analyze.return_value = "{}"
    result = list(analyze_groups([_group()], [True], client, "p", workers=1, retry_config=_no_retry()))
    assert len(result) == 1
    assert result[0] is not None
    assert result[0].elements == []
    assert client.analyze.call_count == 1


# --- analyze_group ---

def test_analyze_group_returns_correct_timing():
    client = MagicMock()
    client.analyze.return_value = WRAPPED_EMPTY
    analysis = analyze_group(_group(), client, prompt="p")
    assert analysis.start_time == 1.0
    assert analysis.end_time == 2.5


def test_analyze_group_parses_elements():
    client = MagicMock()
    client.analyze.return_value = WRAPPED_VALID
    analysis = analyze_group(_group(), client, prompt="p")
    assert len(analysis.elements) == 1
    assert analysis.elements[0].text == "Bonjour"


def test_analyze_group_logs_raw_at_debug(caplog):
    client = MagicMock()
    client.analyze.return_value = WRAPPED_EMPTY
    with caplog.at_level(logging.DEBUG, logger="subtitles_ocr.pipeline.analyze"):
        analyze_group(_group(), client, prompt="p")
    assert any("raw →" in r.message and r.levelno == logging.DEBUG for r in caplog.records)


def test_analyze_group_logs_no_elements(caplog):
    client = MagicMock()
    client.analyze.return_value = WRAPPED_EMPTY
    with caplog.at_level(logging.INFO, logger="subtitles_ocr.pipeline.analyze"):
        analyze_group(_group(), client, prompt="p")
    info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_msgs) == 1
    assert "(no elements)" in info_msgs[0]


def test_analyze_group_passes_prompt_as_system():
    """Instructions belong in the system role — Qwen2.5-VL follows them more reliably there."""
    client = MagicMock()
    client.analyze.return_value = WRAPPED_EMPTY
    analyze_group(_group(), client, prompt="my system prompt")
    _, kwargs = client.analyze.call_args
    assert kwargs.get("system") == "my system prompt"


def test_analyze_group_does_not_use_json_mode():
    """json_mode=True triggers Ollama's grammar-constraint bug with qwen2.5vl — must not be used."""
    client = MagicMock()
    client.analyze.return_value = WRAPPED_EMPTY
    analyze_group(_group(), client, prompt="p")
    _, kwargs = client.analyze.call_args
    assert kwargs.get("json_mode", False) is False


def test_analyze_group_propagates_client_error():
    client = MagicMock()
    client.analyze.side_effect = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        analyze_group(_group(), client, prompt="p")


def test_analyze_group_propagates_parse_error():
    client = MagicMock()
    client.analyze.return_value = "not json"
    with pytest.raises(json.JSONDecodeError):
        analyze_group(_group(), client, prompt="p")


# --- analyze_groups ---

def test_analyze_groups_skips_vlm_when_no_text():
    client = MagicMock()
    group = _group()
    result = list(analyze_groups([group], [False], client, "p", workers=1, retry_config=_no_retry()))
    assert len(result) == 1
    assert result[0].elements == []
    assert result[0].start_time == group.start_time
    client.analyze.assert_not_called()


def test_analyze_groups_calls_vlm_when_has_text():
    client = MagicMock()
    client.analyze.return_value = WRAPPED_VALID
    result = list(analyze_groups([_group()], [True], client, "p", workers=1, retry_config=_no_retry()))
    assert len(result) == 1
    assert len(result[0].elements) == 1
    client.analyze.assert_called_once()


def test_analyze_groups_yields_none_on_exhausted_retries():
    client = MagicMock()
    client.analyze.side_effect = RuntimeError("model failed")
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        result = list(analyze_groups([_group()], [True], client, "p", workers=1, retry_config=_no_retry()))
    assert result == [None]


def test_analyze_groups_logs_warning_on_failure(caplog):
    client = MagicMock()
    client.analyze.side_effect = RuntimeError("model failed")
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        with caplog.at_level(logging.WARNING, logger="subtitles_ocr.pipeline.analyze"):
            list(analyze_groups([_group()], [True], client, "p", workers=1, retry_config=_no_retry()))
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_analyze_groups_error_on_one_does_not_block_others():
    client = MagicMock()
    client.analyze.side_effect = [RuntimeError("fail"), WRAPPED_VALID]
    groups = [
        FrameGroup(start_time=1.0, end_time=2.0, frame=Path("frames/a.jpg")),
        FrameGroup(start_time=3.0, end_time=4.0, frame=Path("frames/b.jpg")),
    ]
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        result = list(analyze_groups(groups, [True, True], client, "p", workers=1, retry_config=_no_retry()))
    assert result[0] is None
    assert result[1] is not None
    assert len(result[1].elements) == 1


def test_analyze_groups_preserves_order():
    client = MagicMock()
    client.analyze.return_value = WRAPPED_EMPTY
    groups = [
        FrameGroup(start_time=1.0, end_time=2.0, frame=Path("frames/a.jpg")),
        FrameGroup(start_time=3.0, end_time=4.0, frame=Path("frames/b.jpg")),
        FrameGroup(start_time=5.0, end_time=6.0, frame=Path("frames/c.jpg")),
    ]
    result = list(analyze_groups(groups, [True, True, True], client, "p", workers=3, retry_config=_no_retry()))
    assert [r.start_time for r in result] == [1.0, 3.0, 5.0]


def test_analyze_groups_mixed_filter():
    client = MagicMock()
    client.analyze.return_value = WRAPPED_VALID
    groups = [_group(), _group(), _group()]
    result = list(analyze_groups(groups, [False, True, False], client, "p", workers=1, retry_config=_no_retry()))
    assert len(result) == 3
    assert result[0].elements == []
    assert len(result[1].elements) == 1
    assert result[2].elements == []
    client.analyze.assert_called_once()


def test_analyze_groups_empty_returns_empty():
    client = MagicMock()
    result = list(analyze_groups([], [], client, "p", workers=1, retry_config=_no_retry()))
    assert result == []


def test_analyze_groups_retries_on_garbage_output():
    """When model returns garbage (no 'subtitles' key), retry and succeed on second attempt."""
    client = MagicMock()
    garbage = json.dumps({"]0E@#@$&,FB8$.-B2=A3F766=E+*9*)?2AC@3#1F9B.0": "#)8#-.:"})
    client.analyze.side_effect = [garbage, WRAPPED_VALID]
    retry_config = RetryConfig(max_attempts=2)
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        result = list(analyze_groups([_group()], [True], client, "p", workers=1, retry_config=retry_config))
    assert result[0] is not None
    assert len(result[0].elements) == 1
    assert client.analyze.call_count == 2


def test_analyze_groups_yields_none_on_non_retryable():
    client = MagicMock()
    client.analyze.side_effect = OSError("disk error")
    result = list(analyze_groups([_group()], [True], client, "p", workers=1, retry_config=_no_retry()))
    assert result == [None]
    assert client.analyze.call_count == 1
