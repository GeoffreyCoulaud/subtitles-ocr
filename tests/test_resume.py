# tests/test_resume.py
import json
import pytest
from dataclasses import dataclass
from pathlib import Path
from subtitles_ocr.pipeline.resume import resume_from_jsonl


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


def test_resume_returns_all_remaining_when_file_missing(tmp_path):
    elements = ["a", "b", "c"]
    processed, remaining = resume_from_jsonl(elements, tmp_path / "missing.jsonl", lambda x: x)
    assert processed == []
    assert remaining == ["a", "b", "c"]


def test_resume_returns_empty_remaining_when_all_processed(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"id": "a"}, {"id": "b"}])
    processed, remaining = resume_from_jsonl(["a", "b"], path, lambda x: x)
    ids = [json.loads(line)["id"] for line in processed]
    assert ids == ["a", "b"]
    assert remaining == []


def test_resume_filters_processed_by_id(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"id": "a", "value": 1}, {"id": "c", "value": 3}])
    processed, remaining = resume_from_jsonl(["a", "b", "c"], path, lambda x: x)
    assert remaining == ["b"]
    assert len(processed) == 2


def test_resume_returns_processed_in_element_order(tmp_path):
    path = tmp_path / "data.jsonl"
    # Written out of order: c before a
    _write_jsonl(path, [{"id": "c", "v": 3}, {"id": "a", "v": 1}])
    processed, remaining = resume_from_jsonl(["a", "b", "c"], path, lambda x: x)
    # processed must be sorted to match element order: a then c
    ids = [json.loads(line)["id"] for line in processed]
    assert ids == ["a", "c"]
    assert remaining == ["b"]


def test_resume_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")
    processed, remaining = resume_from_jsonl(["a", "b", "c"], path, lambda x: x)
    assert remaining == ["c"]
    assert len(processed) == 2


def test_resume_with_custom_id_extractor(tmp_path):
    @dataclass
    class Item:
        name: str
        value: int

    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"id": "foo", "result": 42}])
    items = [Item("foo", 1), Item("bar", 2)]
    processed, remaining = resume_from_jsonl(items, path, lambda item: item.name)
    assert len(processed) == 1
    assert remaining == [Item("bar", 2)]
