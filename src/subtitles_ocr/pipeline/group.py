from subtitles_ocr.models import FrameAnalysis, SubtitleEvent


def _elements_key(analysis: FrameAnalysis) -> list[dict]:
    return [e.model_dump() for e in analysis.elements]


def group_events(analyses: list[FrameAnalysis]) -> list[SubtitleEvent]:
    events: list[SubtitleEvent] = []
    current: SubtitleEvent | None = None
    current_key: list[dict] | None = None

    for analysis in analyses:
        if not analysis.elements:
            current = None
            current_key = None
            continue

        key = _elements_key(analysis)
        if current is not None and key == current_key:
            # current aliases events[-1] — extending the event in-place
            current.end_time = analysis.end_time
        else:
            current = SubtitleEvent(
                start_time=analysis.start_time,
                end_time=analysis.end_time,
                elements=analysis.elements,
            )
            current_key = key
            events.append(current)

    return events
