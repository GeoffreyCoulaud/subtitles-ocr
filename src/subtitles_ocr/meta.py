import hashlib
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class NoCacheKey:
    """Marker for Annotated[T, NoCacheKey] config fields excluded from cache keys."""


class FileFingerprint(BaseModel):
    path: Path
    size: int
    mtime: float
    head_tail_hash: str | None = None
    full_hash: str | None = None


def fingerprint(
    path: Path,
    *,
    treat_as_intermediate: bool = False,
    full_hash_max_bytes: int = 1_000_000,
) -> FileFingerprint:
    stat = path.stat()
    size = stat.st_size
    mtime = stat.st_mtime

    if treat_as_intermediate:
        return FileFingerprint(path=path, size=size, mtime=mtime)

    if size <= full_hash_max_bytes:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return FileFingerprint(path=path, size=size, mtime=mtime, full_hash=h.hexdigest())

    # Large file: hash head + tail only to bound I/O regardless of file size
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(full_hash_max_bytes))
        f.seek(-full_hash_max_bytes, 2)
        h.update(f.read(full_hash_max_bytes))
    return FileFingerprint(path=path, size=size, mtime=mtime, head_tail_hash=h.hexdigest())


class BaseMeta(BaseModel):
    stage_name: str
    stage_version: int
    config: dict
    globals_subset: dict
    input_fingerprints: dict[str, FileFingerprint]
    written_at: datetime

    def matches(self, other: "BaseMeta") -> bool:
        return self.model_dump(exclude={"written_at"}) == other.model_dump(exclude={"written_at"})


def cache_invalidating_dict(config: BaseModel) -> dict:
    # Accept both `Annotated[T, NoCacheKey]` (class as marker, per ADR-0004 §5.2)
    # and `Annotated[T, NoCacheKey()]` (instance) for ergonomics.
    excluded = {
        name
        for name, field in type(config).model_fields.items()
        if any(m is NoCacheKey or isinstance(m, NoCacheKey) for m in field.metadata)
    }
    return config.model_dump(exclude=excluded)
