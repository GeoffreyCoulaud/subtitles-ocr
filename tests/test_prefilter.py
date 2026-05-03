from pathlib import Path
from unittest.mock import MagicMock
from subtitles_ocr.models import FrameGroup
from subtitles_ocr.pipeline.prefilter import prefilter_groups
from subtitles_ocr.vlm.prompt import PREFILTER_PROMPT


def test_prefilter_prompt_is_defined():
    assert isinstance(PREFILTER_PROMPT, str)
    assert "has_text" in PREFILTER_PROMPT


def _group(name: str = "a") -> FrameGroup:
    return FrameGroup(start_time=0.0, end_time=1.0, frame=Path(f"frames/{name}.jpg"))


def test_has_text_true_returns_true():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": true}'
    assert list(prefilter_groups([_group()], client, "p", workers=1)) == [True]


def test_has_text_false_returns_false():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": false}'
    assert list(prefilter_groups([_group()], client, "p", workers=1)) == [False]


def test_invalid_json_returns_true_conservative():
    client = MagicMock()
    client.analyze.return_value = "I cannot determine"
    assert list(prefilter_groups([_group()], client, "p", workers=1)) == [True]


def test_missing_field_returns_true_conservative():
    client = MagicMock()
    client.analyze.return_value = '{"result": "yes"}'
    assert list(prefilter_groups([_group()], client, "p", workers=1)) == [True]


def test_has_text_string_true_returns_true():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": "true"}'
    assert list(prefilter_groups([_group()], client, "p", workers=1)) == [True]


def test_has_text_string_false_returns_false():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": "false"}'
    assert list(prefilter_groups([_group()], client, "p", workers=1)) == [False]


def test_has_text_string_mixed_case_returns_false():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": "False"}'
    assert list(prefilter_groups([_group()], client, "p", workers=1)) == [False]


def test_unrecognised_string_returns_true_conservative():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": "yes"}'
    assert list(prefilter_groups([_group()], client, "p", workers=1)) == [True]


def test_partial_errors_return_true_conservative():
    client = MagicMock()
    client.model = "smolvlm2:256m"
    client.analyze.side_effect = ['{"has_text": true}', RuntimeError("network error")]
    result = list(prefilter_groups([_group("a"), _group("b")], client, "p", workers=1))
    assert result == [True, True]


def test_all_errors_raises():
    import pytest
    client = MagicMock()
    client.model = "smolvlm2:256m"
    client.analyze.side_effect = RuntimeError("model not found")
    with pytest.raises(RuntimeError, match="smolvlm2:256m"):
        list(prefilter_groups([_group("a"), _group("b")], client, "p", workers=1))


def test_order_preserved_with_multiple_workers():
    client = MagicMock()
    client.analyze.side_effect = [
        '{"has_text": false}',
        '{"has_text": true}',
        '{"has_text": false}',
        '{"has_text": true}',
    ]
    groups = [_group("a"), _group("b"), _group("c"), _group("d")]
    result = list(prefilter_groups(groups, client, "p", workers=4))
    assert result == [False, True, False, True]


def test_empty_groups_returns_empty():
    client = MagicMock()
    assert list(prefilter_groups([], client, "p", workers=4)) == []


def test_analyze_called_with_json_mode():
    client = MagicMock()
    client.analyze.return_value = '{"has_text": true}'
    group = _group()
    list(prefilter_groups([group], client, "prompt text", workers=1))
    client.analyze.assert_called_once_with(group.frame, "prompt text", json_mode=True)
