from subtitles_ocr.vlm.prompt import PREFILTER_PROMPT


def test_prefilter_prompt_is_defined():
    assert isinstance(PREFILTER_PROMPT, str)
    assert "yes or no" in PREFILTER_PROMPT.lower()
