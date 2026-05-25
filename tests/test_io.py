import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from subtitles_ocr.exceptions import CacheCorruptionError
from subtitles_ocr.io import JsonlWriter


class FakeItem(BaseModel):
    n: int


def test_happy_path_write_then_read(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    with JsonlWriter(path, FakeItem) as w:
        for i in range(5):
            w.append(FakeItem(n=i))

    with JsonlWriter(path, FakeItem) as r:
        persisted = list(r.iter_persisted())

    assert persisted == [FakeItem(n=i) for i in range(5)]


def test_resume_after_partial_write(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    with JsonlWriter(path, FakeItem) as w:
        for i in range(3):
            w.append(FakeItem(n=i))

    with JsonlWriter(path, FakeItem) as w:
        assert w.resume_index() == 3
        w.append(FakeItem(n=3))
        w.append(FakeItem(n=4))

    with JsonlWriter(path, FakeItem) as r:
        persisted = list(r.iter_persisted())

    assert persisted == [FakeItem(n=i) for i in range(5)]


def test_truncated_last_line_is_silently_rejected(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    with JsonlWriter(path, FakeItem) as w:
        for i in range(3):
            w.append(FakeItem(n=i))

    with path.open("a", encoding="utf-8") as f:
        f.write('{"n": 4')

    with JsonlWriter(path, FakeItem) as w:
        assert w.resume_index() == 3
        w.append(FakeItem(n=3))

    with JsonlWriter(path, FakeItem) as r:
        persisted = list(r.iter_persisted())

    assert persisted == [FakeItem(n=0), FakeItem(n=1), FakeItem(n=2), FakeItem(n=3)]


def test_mid_file_corruption_raises(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    with JsonlWriter(path, FakeItem) as w:
        for i in range(3):
            w.append(FakeItem(n=i))

    with path.open("a", encoding="utf-8") as f:
        f.write("!!!garbage!!!\n")

    with JsonlWriter(path, FakeItem) as w:
        w.append(FakeItem(n=99))

    reader = JsonlWriter(path, FakeItem)
    with pytest.raises(CacheCorruptionError):
        reader.resume_index()


def test_fsync_every_cadence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "items.jsonl"
    calls: list[int] = []
    original_fsync = os.fsync

    def spy(fd: int) -> None:
        calls.append(fd)
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)

    with JsonlWriter(path, FakeItem, fsync_every=2) as w:
        for i in range(5):
            w.append(FakeItem(n=i))
        assert len(calls) == 2
    assert len(calls) == 3
