import pytest
from subtitles_ocr.models import SubtitleElement, SubtitleEvent
from subtitles_ocr.pipeline.fuzzy_group import fuzzy_group_events


def _el(text: str, position: str = "bottom") -> SubtitleElement:
    return SubtitleElement(
        text=text,
        style="regular",
        color="white",
        position=position,
    )


def _event(start: float, end: float, texts: list[str] = (), positions: list[str] = ()) -> SubtitleEvent:
    positions = list(positions) or ["bottom"] * len(texts)
    return SubtitleEvent(
        start_time=start,
        end_time=end,
        elements=[_el(t, p) for t, p in zip(texts, positions)],
    )


def _empty(start: float, end: float) -> SubtitleEvent:
    return SubtitleEvent(start_time=start, end_time=end, elements=[])


# --- basic grouping ---

def test_empty_input_returns_empty():
    assert fuzzy_group_events([], similarity_threshold=0.75, gap_tolerance=0.5) == []


def test_single_event_returns_one_cluster():
    e = _event(0.0, 1.0, ["Bonjour"])
    clusters = fuzzy_group_events([e], similarity_threshold=0.75, gap_tolerance=0.5)
    assert clusters == [[e]]


def test_identical_texts_form_one_cluster():
    a = _event(0.0, 1.0, ["Bonjour tout le monde"])
    b = _event(1.0, 2.0, ["Bonjour tout le monde"])
    clusters = fuzzy_group_events([a, b], similarity_threshold=0.75, gap_tolerance=0.5)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_gap_within_tolerance_bridges_similar_events():
    a = _event(0.0, 1.0, ["N'est pas altéré par les conneries et ne se disperse pas dans les interrogations"])
    b = _event(1.4, 2.0, ["N'est pas altéré par les conneries et ne se disperse pas dans les interrogations"])
    # gap = 1.4 - 1.0 = 0.4s < 0.5s tolerance
    clusters = fuzzy_group_events([a, b], similarity_threshold=0.75, gap_tolerance=0.5)
    assert len(clusters) == 1


def test_gap_above_tolerance_starts_new_cluster():
    a = _event(0.0, 1.0, ["N'est pas altéré par les conneries et ne se disperse pas dans les interrogations"])
    b = _event(1.6, 2.0, ["N'est pas altéré par les conneries et ne se disperse pas dans les interrogations"])
    # gap = 1.6 - 1.0 = 0.6s > 0.5s tolerance
    clusters = fuzzy_group_events([a, b], similarity_threshold=0.75, gap_tolerance=0.5)
    assert len(clusters) == 2


def test_dissimilar_texts_start_new_cluster():
    a = _event(0.0, 1.0, ["Bonjour tout le monde"])
    b = _event(1.0, 2.0, ["Le rapide renard brun saute par-dessus le chien paresseux"])
    clusters = fuzzy_group_events([a, b], similarity_threshold=0.75, gap_tolerance=0.5)
    assert len(clusters) == 2


def test_position_mismatch_starts_new_cluster():
    a = _event(0.0, 1.0, ["Bonjour"], ["bottom"])
    b = _event(1.0, 2.0, ["Bonjour"], ["top"])
    clusters = fuzzy_group_events([a, b], similarity_threshold=0.75, gap_tolerance=0.5)
    assert len(clusters) == 2


# --- empty-element event handling ---

def test_empty_event_is_transparent_within_gap_tolerance():
    a = _event(0.0, 1.0, ["N'est pas altéré par les conneries et ne se disperse pas dans les interrogations"])
    gap = _empty(1.1, 1.2)
    b = _event(1.3, 2.0, ["N'est pas altéré par les conneries et ne se disperse pas dans les interrogations"])
    # gap from a.end_time=1.0 to b.start_time=1.3 = 0.3s < 0.5 tolerance
    clusters = fuzzy_group_events([a, gap, b], similarity_threshold=0.75, gap_tolerance=0.5)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2  # gap event not included in cluster


def test_empty_event_with_large_gap_breaks_cluster():
    a = _event(0.0, 1.0, ["N'est pas altéré par les conneries et ne se disperse pas dans les interrogations"])
    gap = _empty(1.1, 2.5)
    b = _event(2.6, 3.0, ["N'est pas altéré par les conneries et ne se disperse pas dans les interrogations"])
    # gap from a.end_time=1.0 to b.start_time=2.6 = 1.6s > 0.5 tolerance
    clusters = fuzzy_group_events([a, gap, b], similarity_threshold=0.75, gap_tolerance=0.5)
    assert len(clusters) == 2


def test_only_empty_events_returns_empty():
    events = [_empty(0.0, 1.0), _empty(1.0, 2.0)]
    assert fuzzy_group_events(events, similarity_threshold=0.75, gap_tolerance=0.5) == []


# --- similarity uses last event in cluster ---

def test_similarity_compared_against_last_event_in_cluster():
    # Three nearly-identical events: each adjacent pair is similar.
    # Verifies the algorithm compares against cluster[-1], not cluster[0].
    text = "N'est pas altéré par les conneries et ne se disperse pas dans les interrogations"
    a = _event(0.0, 1.0, [text])
    b = _event(1.0, 2.0, [text])
    c = _event(2.0, 3.0, [text])
    clusters = fuzzy_group_events([a, b, c], similarity_threshold=0.75, gap_tolerance=0.5)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3
