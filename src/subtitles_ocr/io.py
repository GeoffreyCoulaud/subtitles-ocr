import os
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Self

from pydantic import BaseModel

from subtitles_ocr.exceptions import CacheCorruptionError


class JsonlWriter[T: BaseModel]:
    def __init__(self, path: Path, model_cls: type[T], fsync_every: int = 1) -> None:
        self._path = path
        self._model_cls = model_cls
        self._fsync_every = fsync_every
        self._fp: IO[str] | None = None
        self._appends_since_fsync = 0
        self._pending_since_fsync = False

    def resume_index(self) -> int:
        if not self._path.exists():
            return 0
        # Read bytes to distinguish a trailing newline (complete last line) from
        # a truncated last line written before a crash.
        raw = self._path.read_bytes()
        if not raw:
            return 0
        text = raw.decode("utf-8")
        lines = text.split("\n")
        last_is_complete = text.endswith("\n")
        if last_is_complete:
            candidate_lines = lines[:-1]
            tolerate_last_invalid = False
        else:
            candidate_lines = lines
            tolerate_last_invalid = True

        valid_count = 0
        last_index = len(candidate_lines) - 1
        for i, line in enumerate(candidate_lines):
            try:
                self._model_cls.model_validate_json(line)
            except ValueError as exc:
                if tolerate_last_invalid and i == last_index:
                    break
                raise CacheCorruptionError(
                    f"Corrupted JSONL line {i + 1} in {self._path}",
                    stage="io",
                    hint="Remove the corrupted file or the entire workdir to recover.",
                ) from exc
            valid_count += 1
        return valid_count

    def iter_persisted(self) -> Iterator[T]:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.rstrip("\n")
                if not stripped:
                    continue
                try:
                    yield self._model_cls.model_validate_json(stripped)
                except ValueError:
                    # Tolerate a truncated trailing line during iteration; mid-file
                    # corruption is the caller's concern (use resume_index() first).
                    return

    def append(self, item: T) -> None:
        if self._fp is None:
            raise RuntimeError("JsonlWriter must be used as a context manager")
        self._fp.write(item.model_dump_json() + "\n")
        self._fp.flush()
        self._appends_since_fsync += 1
        self._pending_since_fsync = True
        if self._appends_since_fsync >= self._fsync_every:
            os.fsync(self._fp.fileno())
            self._appends_since_fsync = 0
            self._pending_since_fsync = False

    def close(self) -> None:
        if self._fp is None:
            return
        self._fp.flush()
        if self._pending_since_fsync:
            os.fsync(self._fp.fileno())
            self._pending_since_fsync = False
        self._fp.close()
        self._fp = None

    def __enter__(self) -> Self:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Drop bytes after the last newline so a partial trailing line from a
        # previous crash does not get concatenated with the next append. This
        # does not validate content — mid-file corruption is surfaced by an
        # explicit resume_index() call by the caller.
        if self._path.exists():
            raw = self._path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                last_newline = raw.rfind(b"\n")
                truncate_to = last_newline + 1 if last_newline != -1 else 0
                with self._path.open("r+b") as f:
                    f.truncate(truncate_to)
        self._fp = self._path.open("a", encoding="utf-8")
        self._appends_since_fsync = 0
        self._pending_since_fsync = False
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
