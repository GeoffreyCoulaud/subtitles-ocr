"""Tests for EventCleanupStage (ADR-0002 §3 Stage 9 / ADR-0003 Stage 10)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from pydantic import BaseModel

from subtitles_ocr.config import EventCleanupConfig, PipelineGlobals
from subtitles_ocr.exceptions import LlmRetryExhausted
from subtitles_ocr.llm.protocol import LlmCallFailed
from subtitles_ocr.pipeline.animation import AnimatedEvent, AnimationAnalysisResult
from subtitles_ocr.pipeline.event_cleanup import (
    CleanedEvent,
    EventCleanupItem,
    EventCleanupStage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(event_id: int, raw_ocr_texts: list[str]) -> AnimatedEvent:
    quad = [(0, 0), (10, 0), (10, 5), (0, 5)]
    return AnimatedEvent(
        event_id=event_id,
        fansub_frame_start=event_id * 10,
        fansub_frame_end=event_id * 10 + 5,
        raw_ocr_texts=raw_ocr_texts,
        raw_ocr_confidences=[0.9] * len(raw_ocr_texts),
        quads_per_frame={event_id * 10: quad},
        quad_median=quad,
        member_frame_indices=[event_id * 10],
        motion=None,
        fade_in_ms=0,
        fade_out_ms=0,
    )


def _write_animation_json(workdir: Path, events: list[AnimatedEvent]) -> None:
    result = AnimationAnalysisResult(events=events, stats={})
    path = workdir / "08_animation" / "animation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(), encoding="utf-8")


class FakeLlm:
    """Records calls and returns scripted CleanedEvent responses."""

    def __init__(self, response_text: str = "canonical") -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, str]] = []  # (prompt, model)
        self._lock = threading.Lock()

    def complete(self, prompt: str, response_schema: type[BaseModel], *, model: str):
        with self._lock:
            self.calls.append((prompt, model))
        assert response_schema is CleanedEvent
        return CleanedEvent(text=self.response_text)


class FailingLlm:
    def __init__(self) -> None:
        self.call_count = 0
        self._lock = threading.Lock()

    def complete(self, prompt: str, response_schema: type[BaseModel], *, model: str):
        with self._lock:
            self.call_count += 1
        raise LlmCallFailed("retries exhausted")


class BarrierLlm:
    """Blocks until N callers have arrived, proving real parallel execution."""

    def __init__(self, expected_parties: int, response_text: str = "p") -> None:
        self.barrier = threading.Barrier(expected_parties, timeout=5)
        self.response_text = response_text
        self.calls: list[int] = []
        self._lock = threading.Lock()

    def complete(self, prompt: str, response_schema: type[BaseModel], *, model: str):
        # If parallelism is sequential, this barrier will time out (BrokenBarrierError).
        self.barrier.wait()
        with self._lock:
            self.calls.append(threading.get_ident())
        return CleanedEvent(text=self.response_text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_consensus_skips_llm_call(tmp_workdir: Path, mock_globals: PipelineGlobals) -> None:
    _write_animation_json(
        tmp_workdir,
        [_make_event(0, ["hello", "hello", "hello"])],
    )
    fake = FakeLlm()
    stage = EventCleanupStage(llm=fake)
    result = stage.run(mock_globals, EventCleanupConfig(model="m"))

    assert len(fake.calls) == 0
    assert len(result.items) == 1
    assert result.items[0].event_id == 0
    assert result.items[0].cleaned_text == "hello"
    assert result.items[0].skipped_llm is True


def test_divergent_variants_call_llm_once_per_event(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_animation_json(
        tmp_workdir,
        [
            _make_event(0, ["foo", "f00", "foo"]),
            _make_event(1, ["bar", "bar"]),  # consensus → no call
            _make_event(2, ["baz", "haz"]),
        ],
    )
    fake = FakeLlm(response_text="cleaned")
    stage = EventCleanupStage(llm=fake)
    result = stage.run(mock_globals, EventCleanupConfig(model="m"))

    assert len(fake.calls) == 2
    assert len(result.items) == 3
    by_id = {it.event_id: it for it in result.items}
    assert by_id[0].cleaned_text == "cleaned"
    assert by_id[0].skipped_llm is False
    assert by_id[1].cleaned_text == "bar"
    assert by_id[1].skipped_llm is True
    assert by_id[2].cleaned_text == "cleaned"
    assert by_id[2].skipped_llm is False


def test_llm_call_failed_is_wrapped_in_llm_retry_exhausted(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_animation_json(
        tmp_workdir,
        [_make_event(0, ["a", "b"])],
    )
    stage = EventCleanupStage(llm=FailingLlm())
    with pytest.raises(LlmRetryExhausted) as exc_info:
        stage.run(mock_globals, EventCleanupConfig(model="m"))

    assert exc_info.value.stage == "10_event_cleanup"
    assert exc_info.value.hint is not None
    assert "--event-cleanup-model" in exc_info.value.hint
    assert isinstance(exc_info.value.__cause__, LlmCallFailed)


def test_resume_skips_already_persisted_events(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    events = [_make_event(i, [f"v{i}", f"w{i}"]) for i in range(5)]
    _write_animation_json(tmp_workdir, events)

    # Pre-populate jsonl with 3 already-processed events.
    out_dir = tmp_workdir / "10_event_cleanup"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "cleaned.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i in range(3):
            f.write(
                EventCleanupItem(
                    event_id=i, cleaned_text=f"pre{i}", skipped_llm=False
                ).model_dump_json()
                + "\n"
            )

    fake = FakeLlm(response_text="new")
    stage = EventCleanupStage(llm=fake)
    result = stage.run(mock_globals, EventCleanupConfig(model="m"))

    # Only events 3 and 4 should hit the LLM.
    assert len(fake.calls) == 2
    assert len(result.items) == 5
    by_id = {it.event_id: it for it in result.items}
    assert by_id[0].cleaned_text == "pre0"
    assert by_id[2].cleaned_text == "pre2"
    assert by_id[3].cleaned_text == "new"
    assert by_id[4].cleaned_text == "new"


def test_parallel_execution_with_parallelism_4(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    events = [_make_event(i, [f"a{i}", f"b{i}"]) for i in range(4)]
    _write_animation_json(tmp_workdir, events)

    barrier_llm = BarrierLlm(expected_parties=4)
    stage = EventCleanupStage(llm=barrier_llm)
    result = stage.run(
        mock_globals,
        EventCleanupConfig(model="m", parallelism=4),
    )

    assert len(barrier_llm.calls) == 4
    assert len(result.items) == 4
    # Order preserved via map().
    assert [it.event_id for it in result.items] == [0, 1, 2, 3]


def test_sidecar_contains_model_but_not_parallelism(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_animation_json(
        tmp_workdir,
        [_make_event(0, ["x", "x"])],
    )
    stage = EventCleanupStage(llm=FakeLlm())
    stage.run(
        mock_globals,
        EventCleanupConfig(model="my-model", parallelism=7),
    )

    meta_path = tmp_workdir / "10_event_cleanup" / "cleaned.meta.json"
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["config"].get("model") == "my-model"
    assert "parallelism" not in payload["config"]


def test_jsonl_written_with_one_line_per_event(
    tmp_workdir: Path, mock_globals: PipelineGlobals
) -> None:
    _write_animation_json(
        tmp_workdir,
        [_make_event(0, ["x", "x"]), _make_event(1, ["y", "z"])],
    )
    stage = EventCleanupStage(llm=FakeLlm(response_text="ok"))
    stage.run(mock_globals, EventCleanupConfig(model="m"))

    jsonl_path = tmp_workdir / "10_event_cleanup" / "cleaned.jsonl"
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    items = [EventCleanupItem.model_validate_json(line) for line in lines]
    assert items[0].event_id == 0
    assert items[0].skipped_llm is True
    assert items[1].event_id == 1
    assert items[1].cleaned_text == "ok"
