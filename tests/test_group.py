from subtitles_ocr.models import FrameAnalysis, SubtitleElement, SubtitleEvent
from subtitles_ocr.pipeline.group import group_events


def _element(text: str = "Bonjour", **kwargs) -> SubtitleElement:
    defaults = dict(
        style="regular",
        color="white",
        position="bottom",
    )
    return SubtitleElement(text=text, **{**defaults, **kwargs})


def _analysis(start: float, end: float, *texts: str) -> FrameAnalysis:
    return FrameAnalysis(
        start_time=start,
        end_time=end,
        elements=[_element(t) for t in texts],
    )


def test_empty_analyses_returns_no_events():
    assert group_events([]) == []


def test_single_analysis_with_no_elements_is_skipped():
    analyses = [_analysis(0.0, 1.0)]
    assert group_events(analyses) == []


def test_single_analysis_with_elements_becomes_event():
    analyses = [_analysis(0.0, 1.0, "Bonjour")]
    events = group_events(analyses)
    assert len(events) == 1
    assert events[0].start_time == 0.0
    assert events[0].end_time == 1.0
    assert events[0].elements[0].text == "Bonjour"


def test_consecutive_identical_analyses_are_merged():
    analyses = [
        _analysis(0.0, 1.0, "Bonjour"),
        _analysis(1.0, 2.0, "Bonjour"),
        _analysis(2.0, 3.0, "Bonjour"),
    ]
    events = group_events(analyses)
    assert len(events) == 1
    assert events[0].start_time == 0.0
    assert events[0].end_time == 3.0


def test_different_texts_stay_separate():
    analyses = [
        _analysis(0.0, 1.0, "Bonjour"),
        _analysis(1.0, 2.0, "Au revoir"),
    ]
    events = group_events(analyses)
    assert len(events) == 2


def test_empty_analysis_breaks_merge():
    analyses = [
        _analysis(0.0, 1.0, "Bonjour"),
        _analysis(1.0, 2.0),           # vide — pas de sous-titre
        _analysis(2.0, 3.0, "Bonjour"),
    ]
    events = group_events(analyses)
    assert len(events) == 2
    assert events[0].end_time == 1.0
    assert events[1].start_time == 2.0


def test_multiple_elements_per_analysis_are_preserved():
    analyses = [_analysis(0.0, 1.0, "Dialogue", "Note du traducteur")]
    events = group_events(analyses)
    assert len(events[0].elements) == 2
