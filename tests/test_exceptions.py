import importlib

import pytest

from subtitles_ocr.exceptions import (
    AlignmentRatioTooLow,
    AspectRatioMismatch,
    CacheCorruptionError,
    InputProbeError,
    LlmPromptTooLarge,
    LlmResponseSchemaError,
    LlmRetryExhausted,
    OcrDeviceInitError,
    PipelineError,
)

SUBCLASSES = [
    InputProbeError,
    AspectRatioMismatch,
    AlignmentRatioTooLow,
    OcrDeviceInitError,
    LlmRetryExhausted,
    LlmResponseSchemaError,
    LlmPromptTooLarge,
    CacheCorruptionError,
]


@pytest.mark.parametrize("cls", SUBCLASSES)
def test_subclass_inherits_pipeline_error(cls):
    assert issubclass(cls, PipelineError)


def test_pipeline_error_stage_and_hint():
    err = PipelineError("x", stage="07", hint="y")
    assert err.stage == "07"
    assert err.hint == "y"


def test_pipeline_error_hint_defaults_to_none():
    err = PipelineError("x", stage="07")
    assert err.hint is None


def test_pipeline_error_is_exception():
    err = PipelineError("x", stage="07")
    assert isinstance(err, Exception)


def test_pipeline_error_message_preserved():
    err = PipelineError("some message", stage="07")
    assert str(err) == "some message"


def test_import_has_no_side_effects():
    # Reimporting the module must not raise or print anything.
    mod = importlib.import_module("subtitles_ocr.exceptions")
    importlib.reload(mod)


@pytest.mark.parametrize("cls", SUBCLASSES)
def test_subclass_instantiation_with_stage_and_hint(cls):
    err = cls("msg", stage="01", hint="fix it")
    assert err.stage == "01"
    assert err.hint == "fix it"
    assert str(err) == "msg"


@pytest.mark.parametrize("cls", SUBCLASSES)
def test_subclass_hint_defaults_to_none(cls):
    err = cls("msg", stage="01")
    assert err.hint is None
