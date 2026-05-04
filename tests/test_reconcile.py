# tests/test_reconcile.py
import pytest
from unittest.mock import MagicMock, patch
from subtitles_ocr.models import SubtitleElement, SubtitleEvent
from subtitles_ocr.pipeline.reconcile import _reconcile_cluster, reconcile_groups
from subtitles_ocr.pipeline.retry import RetryConfig


def _no_retry() -> RetryConfig:
    return RetryConfig(max_attempts=1)


def _el(text: str, color: str = "white", alignment: str = "center", position: str = "bottom") -> SubtitleElement:
    return SubtitleElement(
        text=text, style="regular", color=color,
        border_color="black", position=position, alignment=alignment,
    )


def _event(start: float, end: float, elements: list[SubtitleElement]) -> SubtitleEvent:
    return SubtitleEvent(start_time=start, end_time=end, elements=elements)


def test_single_event_cluster_passes_through():
    event = _event(0.0, 1.0, [_el("Bonjour")])
    client = MagicMock()
    result = _reconcile_cluster([event], client)
    assert result.start_time == 0.0
    assert result.end_time == 1.0
    assert result.elements[0].text == "Bonjour"
    client.chat.assert_not_called()


def test_identical_texts_skip_llm_call():
    events = [_event(float(i), float(i + 1), [_el("Bonjour tout le monde")]) for i in range(3)]
    client = MagicMock()
    result = _reconcile_cluster(events, client)
    client.chat.assert_not_called()
    assert result.elements[0].text == "Bonjour tout le monde"


def test_start_and_end_time_from_first_and_last_event():
    events = [_event(5.0, 6.0, [_el("A")]), _event(6.0, 7.0, [_el("A")]), _event(7.0, 9.5, [_el("A")])]
    client = MagicMock()
    result = _reconcile_cluster(events, client)
    assert result.start_time == 5.0
    assert result.end_time == 9.5


def test_majority_vote_selects_most_frequent_color():
    events = [
        _event(0.0, 1.0, [_el("A", color="white")]),
        _event(1.0, 2.0, [_el("A", color="white")]),
        _event(2.0, 3.0, [_el("A", color="yellow")]),
    ]
    client = MagicMock()
    result = _reconcile_cluster(events, client)
    client.chat.assert_not_called()
    assert result.elements[0].color == "#FFFFFF"


def test_majority_vote_selects_most_frequent_alignment():
    events = [
        _event(0.0, 1.0, [_el("A", alignment="left")]),
        _event(1.0, 2.0, [_el("A", alignment="center")]),
        _event(2.0, 3.0, [_el("A", alignment="center")]),
    ]
    client = MagicMock()
    assert _reconcile_cluster(events, client).elements[0].alignment == "center"


def test_majority_vote_tie_uses_first_encountered():
    events = [
        _event(0.0, 1.0, [_el("A", alignment="left")]),
        _event(1.0, 2.0, [_el("A", alignment="center")]),
    ]
    client = MagicMock()
    assert _reconcile_cluster(events, client).elements[0].alignment == "left"


def test_llm_called_when_texts_differ():
    events = [_event(0.0, 1.0, [_el("Bonjour monde")]), _event(1.0, 2.0, [_el("Bonsoir monde")])]
    client = MagicMock()
    client.chat.return_value = "Bonjour monde"
    result = _reconcile_cluster(events, client)
    client.chat.assert_called_once()
    assert result.elements[0].text == "Bonjour monde"


def test_llm_failure_propagates_from_reconcile_cluster():
    events = [_event(0.0, 1.0, [_el("Bonjour monde")]), _event(1.0, 2.0, [_el("Bonsoir monde")])]
    client = MagicMock()
    client.chat.side_effect = RuntimeError("model unavailable")
    with pytest.raises(RuntimeError, match="model unavailable"):
        _reconcile_cluster(events, client)


def test_reconcile_groups_yields_one_event_per_cluster():
    clusters = [[_event(0.0, 1.0, [_el("Alpha")])], [_event(2.0, 3.0, [_el("Beta")])]]
    client = MagicMock()
    results = list(reconcile_groups(clusters, client, workers=1, retry_config=_no_retry()))
    assert len(results) == 2
    assert results[0].elements[0].text == "Alpha"
    assert results[1].elements[0].text == "Beta"


def test_reconcile_groups_yields_none_on_exhausted_retries():
    events = [_event(0.0, 1.0, [_el("Bonjour")]), _event(1.0, 2.0, [_el("Bonsoir")])]
    client = MagicMock()
    client.chat.side_effect = RuntimeError("always fails")
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        results = list(reconcile_groups([[*events]], client, workers=1, retry_config=_no_retry()))
    assert results == [None]


def test_reconcile_groups_error_on_one_does_not_block_others():
    cluster_ok = [_event(0.0, 1.0, [_el("OK")])]
    cluster_fail = [_event(2.0, 3.0, [_el("A")]), _event(3.0, 4.0, [_el("B")])]
    client = MagicMock()
    client.chat.side_effect = RuntimeError("fail")
    with patch("subtitles_ocr.pipeline.retry.time.sleep"):
        results = list(reconcile_groups([cluster_ok, cluster_fail], client, workers=1, retry_config=_no_retry()))
    assert results[0] is not None
    assert results[0].elements[0].text == "OK"
    assert results[1] is None


def test_reconcile_groups_preserves_order_with_multiple_workers():
    clusters = [[_event(float(i), float(i + 1), [_el(f"text{i}")])] for i in range(6)]
    client = MagicMock()
    results = list(reconcile_groups(clusters, client, workers=4, retry_config=_no_retry()))
    assert [r.elements[0].text for r in results] == [f"text{i}" for i in range(6)]


def test_reconcile_groups_yields_none_on_non_retryable():
    cluster_fail = [_event(0.0, 1.0, [_el("A")]), _event(1.0, 2.0, [_el("B")])]
    client = MagicMock()
    client.chat.side_effect = OSError("disk error")
    results = list(reconcile_groups([cluster_fail], client, workers=1, retry_config=_no_retry()))
    assert results == [None]
    assert client.chat.call_count == 1
