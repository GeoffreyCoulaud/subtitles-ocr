from subtitles_ocr.llm import LlmCallFailed, LlmClient


def test_llm_client_and_llm_call_failed_are_importable():
    assert LlmClient is not None
    assert LlmCallFailed is not None


def test_llm_call_failed_is_exception_subclass():
    assert issubclass(LlmCallFailed, Exception)
