import pytest
from unittest.mock import MagicMock
from subtitles_ocr.models import SubtitleElement, SubtitleEvent
from subtitles_ocr.pipeline.reconcile import _reconcile_cluster, reconcile_groups


def _el(text: str, color: str = "white", alignment: str = "center", position: str = "bottom") -> SubtitleElement:
    return SubtitleElement(
        text=text,
        style="regular",
        color=color,
        border_color="black",
        position=position,
        alignment=alignment,
    )


def _event(start: float, end: float, elements: list[SubtitleElement]) -> SubtitleEvent:
    return SubtitleEvent(start_time=start, end_time=end, elements=elements)


# --- single-event cluster ---

def test_single_event_cluster_passes_through():
    el = _el("Bonjour")
    event = _event(0.0, 1.0, [el])
    client = MagicMock()
    result = _reconcile_cluster([event], client)
    assert result.start_time == 0.0
    assert result.end_time == 1.0
    assert result.elements[0].text == "Bonjour"
    client.chat.assert_not_called()


# --- identical texts skip LLM ---

def test_identical_texts_skip_llm_call():
    events = [
        _event(0.0, 1.0, [_el("Bonjour tout le monde")]),
        _event(1.0, 2.0, [_el("Bonjour tout le monde")]),
        _event(2.0, 3.0, [_el("Bonjour tout le monde")]),
    ]
    client = MagicMock()
    result = _reconcile_cluster(events, client)
    client.chat.assert_not_called()
    assert result.elements[0].text == "Bonjour tout le monde"


# --- timing ---

def test_start_and_end_time_from_first_and_last_event():
    events = [
        _event(5.0, 6.0, [_el("Bonjour")]),
        _event(6.0, 7.0, [_el("Bonjour")]),
        _event(7.0, 9.5, [_el("Bonjour")]),
    ]
    client = MagicMock()
    result = _reconcile_cluster(events, client)
    assert result.start_time == 5.0
    assert result.end_time == 9.5


# --- majority vote ---

def test_majority_vote_selects_most_frequent_color():
    # 2 white (#FFFFFF after resolution), 1 yellow (#FFFF00 after resolution)
    events = [
        _event(0.0, 1.0, [_el("Bonjour", color="white")]),
        _event(1.0, 2.0, [_el("Bonjour", color="white")]),
        _event(2.0, 3.0, [_el("Bonjour", color="yellow")]),
    ]
    client = MagicMock()
    result = _reconcile_cluster(events, client)
    client.chat.assert_not_called()
    assert result.elements[0].color == "#FFFFFF"


def test_majority_vote_selects_most_frequent_alignment():
    events = [
        _event(0.0, 1.0, [_el("Bonjour", alignment="left")]),
        _event(1.0, 2.0, [_el("Bonjour", alignment="center")]),
        _event(2.0, 3.0, [_el("Bonjour", alignment="center")]),
    ]
    client = MagicMock()
    result = _reconcile_cluster(events, client)
    assert result.elements[0].alignment == "center"


def test_majority_vote_tie_uses_first_encountered():
    events = [
        _event(0.0, 1.0, [_el("Bonjour", alignment="left")]),
        _event(1.0, 2.0, [_el("Bonjour", alignment="center")]),
    ]
    client = MagicMock()
    result = _reconcile_cluster(events, client)
    assert result.elements[0].alignment == "left"  # first encountered wins


# --- LLM text reconciliation ---

def test_llm_called_when_texts_differ():
    events = [
        _event(0.0, 1.0, [_el("Bonjour monde")]),
        _event(1.0, 2.0, [_el("Bonsoir monde")]),
    ]
    client = MagicMock()
    client.chat.return_value = "Bonjour monde"
    result = _reconcile_cluster(events, client)
    client.chat.assert_called_once()
    assert result.elements[0].text == "Bonjour monde"


def test_llm_failure_falls_back_to_mbr():
    # With 2 texts, MBR picks the first (symmetric similarity → tie broken by iteration order)
    events = [
        _event(0.0, 1.0, [_el("Bonjour monde")]),
        _event(1.0, 2.0, [_el("Bonsoir monde")]),
    ]
    client = MagicMock()
    client.chat.side_effect = RuntimeError("model unavailable")
    result = _reconcile_cluster(events, client)
    assert result.elements[0].text in {"Bonjour monde", "Bonsoir monde"}


# --- reconcile_groups (public API) ---

def test_reconcile_groups_yields_one_event_per_cluster():
    clusters = [
        [_event(0.0, 1.0, [_el("Alpha")])],
        [_event(2.0, 3.0, [_el("Beta")])],
    ]
    client = MagicMock()
    results = list(reconcile_groups(clusters, client, workers=1))
    assert len(results) == 2
    assert results[0].elements[0].text == "Alpha"
    assert results[1].elements[0].text == "Beta"


def test_reconcile_groups_preserves_order_with_multiple_workers():
    clusters = [
        [_event(float(i), float(i + 1), [_el(f"text{i}")])]
        for i in range(6)
    ]
    client = MagicMock()
    results = list(reconcile_groups(clusters, client, workers=4))
    assert [r.elements[0].text for r in results] == [f"text{i}" for i in range(6)]
