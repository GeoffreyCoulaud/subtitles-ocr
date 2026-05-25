from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel

from subtitles_ocr.meta import (
    BaseMeta,
    FileFingerprint,
    NoCacheKey,
    cache_invalidating_dict,
    fingerprint,
)


def test_fingerprint_small_file_has_full_hash(tmp_path: Path) -> None:
    p = tmp_path / "small.bin"
    p.write_bytes(b"hello world")

    fp = fingerprint(p)

    assert fp.size == len(b"hello world")
    assert fp.full_hash is not None
    assert fp.head_tail_hash is None
    assert fp.path == p
    assert isinstance(fp.mtime, float)


def test_fingerprint_large_file_has_head_tail_hash(tmp_path: Path) -> None:
    p = tmp_path / "large.bin"
    # Sparse-ish file: 2MB total, exceeds default 1MB threshold
    with p.open("wb") as f:
        f.seek(2 * 1024 * 1024)
        f.write(b"x")

    fp = fingerprint(p)

    assert fp.size == 2 * 1024 * 1024 + 1
    assert fp.head_tail_hash is not None
    assert fp.full_hash is None


def test_fingerprint_intermediate_skips_hashing(tmp_path: Path) -> None:
    p = tmp_path / "intermediate.bin"
    p.write_bytes(b"some content")

    fp = fingerprint(p, treat_as_intermediate=True)

    assert fp.full_hash is None
    assert fp.head_tail_hash is None
    assert fp.size == len(b"some content")


def test_fingerprint_full_hash_threshold_boundary(tmp_path: Path) -> None:
    # File exactly at the threshold uses full_hash (<=)
    threshold = 1024
    p = tmp_path / "boundary.bin"
    p.write_bytes(b"a" * threshold)

    fp = fingerprint(p, full_hash_max_bytes=threshold)

    assert fp.full_hash is not None
    assert fp.head_tail_hash is None


def test_fingerprint_full_hash_changes_with_content(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"content A")
    b.write_bytes(b"content B")

    fa = fingerprint(a)
    fb = fingerprint(b)

    assert fa.full_hash != fb.full_hash


def test_fingerprint_head_tail_hash_changes_with_content(tmp_path: Path) -> None:
    big_size = 3 * 1024 * 1024
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"A" * big_size)
    b.write_bytes(b"B" * big_size)

    fa = fingerprint(a, full_hash_max_bytes=1_000_000)
    fb = fingerprint(b, full_hash_max_bytes=1_000_000)

    assert fa.head_tail_hash != fb.head_tail_hash


def test_cache_invalidating_dict_excludes_nocachekey_fields() -> None:
    class MyConfig(BaseModel):
        important: int
        secret: str
        verbose: Annotated[bool, NoCacheKey] = False
        progress_bar: Annotated[bool, NoCacheKey] = True

    cfg = MyConfig(important=42, secret="hello", verbose=True, progress_bar=False)

    result = cache_invalidating_dict(cfg)

    assert "important" in result
    assert "secret" in result
    assert "verbose" not in result
    assert "progress_bar" not in result
    assert result == {"important": 42, "secret": "hello"}


def _make_meta(**overrides) -> BaseMeta:
    base = dict(
        stage_name="probe",
        stage_version=1,
        config={"k": "v"},
        globals_subset={"g": 1},
        input_fingerprints={
            "raw": FileFingerprint(path=Path("/raw.mkv"), size=10, mtime=1.0, full_hash="abc")
        },
        written_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return BaseMeta(**base)


def test_basemeta_matches_ignores_written_at() -> None:
    a = _make_meta()
    b = _make_meta(written_at=datetime(2099, 12, 31, tzinfo=timezone.utc))

    assert a.matches(b)


def test_basemeta_matches_detects_stage_version_change() -> None:
    a = _make_meta()
    b = _make_meta(stage_version=2)

    assert not a.matches(b)


def test_basemeta_matches_detects_config_change() -> None:
    a = _make_meta()
    b = _make_meta(config={"k": "different"})

    assert not a.matches(b)


def test_basemeta_matches_detects_input_fingerprint_change() -> None:
    a = _make_meta()
    b = _make_meta(
        input_fingerprints={
            "raw": FileFingerprint(path=Path("/raw.mkv"), size=10, mtime=2.0, full_hash="abc")
        }
    )

    assert not a.matches(b)


def test_basemeta_matches_detects_globals_subset_change() -> None:
    a = _make_meta()
    b = _make_meta(globals_subset={"g": 2})

    assert not a.matches(b)


def test_basemeta_matches_detects_stage_name_change() -> None:
    a = _make_meta()
    b = _make_meta(stage_name="other")

    assert not a.matches(b)
