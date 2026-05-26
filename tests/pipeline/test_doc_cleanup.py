"""Tests for DocCleanupStage (Stage 11, ADR-0002 §3 Stage 10).

Uses a Protocol-based FakeLlm via constructor DI per ADR-0004 §13.4. No
monkeypatch. Real JSONL/JSON I/O against tmp_workdir to exercise the on-disk
contract end-to-end.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from subtitles_ocr.config import DocCleanupConfig, PipelineGlobals
from subtitles_ocr.exceptions import (
    LlmPromptTooLarge,
    LlmResponseSchemaError,
    LlmRetryExhausted,
)
from subtitles_ocr.io import JsonlWriter
from subtitles_ocr.llm.protocol import LlmCallFailed
from subtitles_ocr.meta import BaseMeta, FileFingerprint
from subtitles_ocr.pipeline.doc_cleanup import (
    DocCleanupResult,
    DocCleanupStage,
    FinalEvent,
    STAGE_VERSION,
)
from subtitles_ocr.pipeline.event_cleanup import EventCleanupItem


# -------------------- helpers --------------------


class FakeLlm:
    """Protocol-conforming LlmClient that captures invocations."""

    def __init__(
        self,
        *,
        responses: list[BaseModel] | None = None,
        errors: list[Exception] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.errors = list(errors or [])
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, response_schema: type, *, model: str):
        self.calls.append({"prompt": prompt, "schema": response_schema, "model": model})
        if self.errors:
            raise self.errors.pop(0)
        return self.responses.pop(0)


def _write_event_cleanup_input(
    workdir: Path, items: list[EventCleanupItem]
) -> None:
    path = workdir / "10_event_cleanup" / "cleaned.jsonl"
    with JsonlWriter(path, EventCleanupItem) as w:
        for it in items:
            w.append(it)


def _make_items(n: int) -> list[EventCleanupItem]:
    return [
        EventCleanupItem(event_id=i, cleaned_text=f"text {i}", skipped_llm=False)
        for i in range(n)
    ]


# -------------------- happy path --------------------


def test_happy_path_writes_cleaned_final_and_returns_result(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_event_cleanup_input(tmp_workdir, _make_items(5))
    llm = FakeLlm(
        responses=[
            DocCleanupResult(
                events=[FinalEvent(event_id=i, cleaned_text=f"clean {i}") for i in range(5)]
            )
        ]
    )

    stage = DocCleanupStage(llm=llm)
    result = stage.run(mock_globals, DocCleanupConfig(model="m"))

    assert len(result.events) == 5
    assert [e.event_id for e in result.events] == [0, 1, 2, 3, 4]
    assert result.events[0].cleaned_text == "clean 0"
    out = tmp_workdir / "11_doc_cleanup" / "cleaned_final.json"
    assert out.exists()
    persisted = DocCleanupResult.model_validate_json(out.read_text())
    assert len(persisted.events) == 5
    sidecar = tmp_workdir / "11_doc_cleanup" / "cleaned_final.meta.json"
    assert sidecar.exists()
    meta = BaseMeta.model_validate_json(sidecar.read_text())
    assert meta.stage_name == "11_doc_cleanup"
    assert meta.stage_version == STAGE_VERSION
    assert "model" in meta.config
    assert llm.calls[0]["model"] == "m"
    assert llm.calls[0]["schema"] is DocCleanupResult


def test_happy_path_calls_llm_exactly_once(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_event_cleanup_input(tmp_workdir, _make_items(3))
    llm = FakeLlm(
        responses=[
            DocCleanupResult(
                events=[FinalEvent(event_id=i, cleaned_text=f"x{i}") for i in range(3)]
            )
        ]
    )

    DocCleanupStage(llm=llm).run(mock_globals, DocCleanupConfig(model="m"))

    assert len(llm.calls) == 1


# -------------------- prompt structure --------------------


def test_prompt_user_section_contains_synopsis_null_when_no_path(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_event_cleanup_input(tmp_workdir, _make_items(2))
    llm = FakeLlm(
        responses=[
            DocCleanupResult(
                events=[FinalEvent(event_id=i, cleaned_text=f"x{i}") for i in range(2)]
            )
        ]
    )

    DocCleanupStage(llm=llm).run(mock_globals, DocCleanupConfig(model="m"))

    prompt = llm.calls[0]["prompt"]
    # The prompt must encode synopsis as JSON null when no synopsis file is set
    assert '"synopsis": null' in prompt
    # And carry every event id + text
    assert '"id": 0' in prompt
    assert '"id": 1' in prompt
    assert '"text": "text 0"' in prompt


def test_prompt_includes_synopsis_file_content(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_event_cleanup_input(tmp_workdir, _make_items(2))
    synopsis_path = tmp_workdir / "synopsis.md"
    synopsis_text = "# Episode 1\n\nThe hero meets the mentor."
    synopsis_path.write_text(synopsis_text, encoding="utf-8")

    llm = FakeLlm(
        responses=[
            DocCleanupResult(
                events=[FinalEvent(event_id=i, cleaned_text=f"x{i}") for i in range(2)]
            )
        ]
    )

    DocCleanupStage(llm=llm).run(
        mock_globals,
        DocCleanupConfig(model="m", synopsis_path=synopsis_path),
    )

    prompt = llm.calls[0]["prompt"]
    # The full synopsis text must be embedded inside the JSON payload (string-encoded)
    assert "hero meets the mentor" in prompt
    assert '"synopsis": null' not in prompt


# -------------------- validation failures --------------------


def test_id_mismatch_raises_llm_response_schema_error(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_event_cleanup_input(tmp_workdir, _make_items(3))
    # Same count, wrong id sequence
    llm = FakeLlm(
        responses=[
            DocCleanupResult(
                events=[
                    FinalEvent(event_id=0, cleaned_text="x"),
                    FinalEvent(event_id=2, cleaned_text="y"),  # should be 1
                    FinalEvent(event_id=1, cleaned_text="z"),  # should be 2
                ]
            )
        ]
    )

    with pytest.raises(LlmResponseSchemaError) as exc_info:
        DocCleanupStage(llm=llm).run(mock_globals, DocCleanupConfig(model="m"))
    assert exc_info.value.stage == "11_doc_cleanup"


def test_count_mismatch_raises_llm_response_schema_error(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_event_cleanup_input(tmp_workdir, _make_items(3))
    llm = FakeLlm(
        responses=[
            DocCleanupResult(
                events=[FinalEvent(event_id=0, cleaned_text="x")]  # too few
            )
        ]
    )

    with pytest.raises(LlmResponseSchemaError) as exc_info:
        DocCleanupStage(llm=llm).run(mock_globals, DocCleanupConfig(model="m"))
    assert exc_info.value.stage == "11_doc_cleanup"


def test_schema_error_does_not_write_output(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_event_cleanup_input(tmp_workdir, _make_items(2))
    llm = FakeLlm(
        responses=[DocCleanupResult(events=[FinalEvent(event_id=0, cleaned_text="x")])]
    )

    with pytest.raises(LlmResponseSchemaError):
        DocCleanupStage(llm=llm).run(mock_globals, DocCleanupConfig(model="m"))

    assert not (tmp_workdir / "11_doc_cleanup" / "cleaned_final.json").exists()
    assert not (tmp_workdir / "11_doc_cleanup" / "cleaned_final.meta.json").exists()


# -------------------- retry exhaustion --------------------


def test_llm_call_failed_is_converted_to_llm_retry_exhausted(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_event_cleanup_input(tmp_workdir, _make_items(2))
    llm = FakeLlm(errors=[LlmCallFailed("3/3 attempts failed")])

    with pytest.raises(LlmRetryExhausted) as exc_info:
        DocCleanupStage(llm=llm).run(mock_globals, DocCleanupConfig(model="m"))
    assert exc_info.value.stage == "11_doc_cleanup"
    assert isinstance(exc_info.value.__cause__, LlmCallFailed)


# -------------------- context overflow --------------------


def test_context_overflow_error_is_converted_to_llm_prompt_too_large(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_event_cleanup_input(tmp_workdir, _make_items(2))
    # Ollama-style error message indicating context overflow
    llm = FakeLlm(
        errors=[LlmCallFailed("model context length exceeded: prompt has too many tokens")]
    )

    with pytest.raises(LlmPromptTooLarge) as exc_info:
        DocCleanupStage(llm=llm).run(mock_globals, DocCleanupConfig(model="m"))
    assert exc_info.value.stage == "11_doc_cleanup"


# -------------------- resume --------------------


def test_resume_skips_llm_when_sidecar_matches(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_event_cleanup_input(tmp_workdir, _make_items(2))

    # Pre-populate output + matching sidecar
    out_dir = tmp_workdir / "11_doc_cleanup"
    out_dir.mkdir(exist_ok=True)
    cached_result = DocCleanupResult(
        events=[FinalEvent(event_id=i, cleaned_text=f"cached {i}") for i in range(2)]
    )
    (out_dir / "cleaned_final.json").write_text(cached_result.model_dump_json())

    # Build the sidecar exactly as the stage would (so resume sees a match)
    cfg = DocCleanupConfig(model="m")
    # Trigger first run with a fresh stage that will write the legit sidecar
    llm_first = FakeLlm(responses=[cached_result])
    # Remove the precomputed output, let the first run create both files atomically
    (out_dir / "cleaned_final.json").unlink()
    DocCleanupStage(llm=llm_first).run(mock_globals, cfg)
    assert (out_dir / "cleaned_final.meta.json").exists()
    assert len(llm_first.calls) == 1

    # Second run: LLM has no responses queued -> must NOT be called
    llm_second = FakeLlm(responses=[])
    result = DocCleanupStage(llm=llm_second).run(mock_globals, cfg)
    assert len(llm_second.calls) == 0
    assert [e.cleaned_text for e in result.events] == ["cached 0", "cached 1"]


def test_resume_recomputes_when_config_changes(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_event_cleanup_input(tmp_workdir, _make_items(2))

    initial_result = DocCleanupResult(
        events=[FinalEvent(event_id=i, cleaned_text=f"a{i}") for i in range(2)]
    )
    new_result = DocCleanupResult(
        events=[FinalEvent(event_id=i, cleaned_text=f"b{i}") for i in range(2)]
    )

    # First run with model "m1"
    DocCleanupStage(llm=FakeLlm(responses=[initial_result])).run(
        mock_globals, DocCleanupConfig(model="m1")
    )
    # Second run with different model -> cache invalidates -> LLM called again
    llm_second = FakeLlm(responses=[new_result])
    result = DocCleanupStage(llm=llm_second).run(
        mock_globals, DocCleanupConfig(model="m2")
    )
    assert len(llm_second.calls) == 1
    assert [e.cleaned_text for e in result.events] == ["b0", "b1"]
