from pathlib import Path
from unittest.mock import MagicMock
from subtitles_ocr.models import FrameGroup
from subtitles_ocr.pipeline.prefilter import prefilter_groups
from subtitles_ocr.vlm.prompt import PREFILTER_PROMPT


def test_prefilter_prompt_is_defined():
    assert isinstance(PREFILTER_PROMPT, str)
    assert "yes or no" in PREFILTER_PROMPT.lower()


def _group(name: str = "a") -> FrameGroup:
    return FrameGroup(start_time=0.0, end_time=1.0, frame=Path(f"frames/{name}.jpg"))


def test_yes_response_returns_true():
    client = MagicMock()
    client.analyze.return_value = "yes"
    assert prefilter_groups([_group()], client, "p", workers=1) == [True]


def test_yes_case_insensitive():
    client = MagicMock()
    client.analyze.return_value = "Yes"
    assert prefilter_groups([_group()], client, "p", workers=1) == [True]


def test_no_response_returns_false():
    client = MagicMock()
    client.analyze.return_value = "no"
    assert prefilter_groups([_group()], client, "p", workers=1) == [False]


def test_ambiguous_response_returns_true_conservative():
    client = MagicMock()
    client.analyze.return_value = "I cannot determine"
    assert prefilter_groups([_group()], client, "p", workers=1) == [True]


def test_error_returns_true_conservative():
    client = MagicMock()
    client.analyze.side_effect = RuntimeError("network error")
    assert prefilter_groups([_group()], client, "p", workers=1) == [True]


def test_order_preserved_with_multiple_workers():
    client = MagicMock()
    client.analyze.side_effect = ["no", "yes", "no", "yes"]
    groups = [_group("a"), _group("b"), _group("c"), _group("d")]
    result = prefilter_groups(groups, client, "p", workers=4)
    assert result == [False, True, False, True]


def test_empty_groups_returns_empty():
    client = MagicMock()
    assert prefilter_groups([], client, "p", workers=4) == []
