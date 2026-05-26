"""Tests for Stage 2 — AlignmentStage (audio + phash).

All inputs are synthetic numpy. Real ffmpeg / silero-vad / opencv are replaced
by Protocol-based fakes injected through the stage constructor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from subtitles_ocr.config import AlignmentConfig, PipelineGlobals
from subtitles_ocr.exceptions import AlignmentRatioTooLow
from subtitles_ocr.pipeline.alignment import (
    AlignmentResult,
    AlignmentSegment,
    AlignmentStage,
)
from subtitles_ocr.pipeline.alignment.audio import (
    hierarchical_cross_correlate,
    post_filter_isolated_matches,
    normalize_zscore,
    WindowVerdict,
)
from subtitles_ocr.pipeline.alignment.phash import (
    hamming_distance64,
    phash_fallback,
    refine_phash,
)


# ---------------------------------------------------------------------------
# Fakes (Protocol DI — no monkeypatching)
# ---------------------------------------------------------------------------


@dataclass
class FakeFfmpegRunner:
    """Records extract_audio calls and writes a sentinel file at the out path."""

    fail_on_track: int | None = None
    audio_payloads: dict[Path, bytes] = field(default_factory=dict)
    extract_calls: list[tuple[Path, int, Path]] = field(default_factory=list)

    def probe(self, path):
        raise NotImplementedError

    def transcode(self, args):
        raise NotImplementedError

    def extract_audio(self, path: Path, track_index: int, out: Path) -> None:
        self.extract_calls.append((path, track_index, out))
        if self.fail_on_track is not None and track_index == self.fail_on_track:
            raise RuntimeError("synthetic extract failure")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.audio_payloads.get(out, b"fake-wav"))


@dataclass
class FakeVadModel:
    """Returns user-supplied probability arrays keyed by sample-hash so multiple
    calls with different audio inputs return different probs.
    """

    probs_by_key: dict[str, np.ndarray] = field(default_factory=dict)
    default_probs: np.ndarray | None = None

    def compute_probs(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        key = f"{samples.shape}-{float(samples.sum()):.6f}"
        if key in self.probs_by_key:
            return self.probs_by_key[key]
        if self.default_probs is not None:
            return self.default_probs
        raise KeyError(f"FakeVadModel: no probs registered for key {key}")


@dataclass
class FakeAudioLoader:
    """Returns per-WAV-path synthetic mono float32 samples."""

    samples_by_path: dict[Path, tuple[np.ndarray, int]] = field(default_factory=dict)

    def load(self, path: Path) -> tuple[np.ndarray, int]:
        if path not in self.samples_by_path:
            raise FileNotFoundError(path)
        return self.samples_by_path[path]


@dataclass
class FakePhasher:
    """Returns predetermined 64-bit hashes per (source, frame_idx)."""

    hashes: dict[tuple[str, int], int] = field(default_factory=dict)

    def hash_frame(self, source: str, frame_idx: int) -> int:
        return self.hashes.get((source, frame_idx), 0)


@dataclass
class FakeFrameSource:
    """Provides phash for a given source (fansub/raw) and frame_idx via FakePhasher."""

    phasher: FakePhasher

    def get_phash(self, source: str, frame_idx: int) -> int:
        return self.phasher.hash_frame(source, frame_idx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_globals(tmp_workdir: Path, fansub_total: int = 1000) -> PipelineGlobals:
    return PipelineGlobals(
        workdir=tmp_workdir,
        hardsub_path=Path("/dev/null/fansub.avi"),
        raw_path=Path("/dev/null/raw.mkv"),
        out_path=tmp_workdir / "out.ass",
        fps=Fraction(24, 1),
        fansub_width=640,
        fansub_height=480,
        fansub_total_frames=fansub_total,
        debug_images=False,
    )


# =============================================================================
# Pure function tests — audio.py
# =============================================================================


def test_normalize_zscore_centers_and_scales() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = normalize_zscore(arr)
    assert abs(float(out.mean())) < 1e-6
    assert abs(float(out.std()) - 1.0) < 1e-6


def test_normalize_zscore_zero_std_returns_zeros() -> None:
    arr = np.array([3.0, 3.0, 3.0])
    out = normalize_zscore(arr)
    assert np.all(out == 0.0)


def test_hierarchical_xcorr_finds_known_offset() -> None:
    # Two probability streams with a deterministic offset of +5 samples (b leads).
    rng = np.random.default_rng(0)
    base = rng.normal(size=2000).astype(np.float32)
    a = base.copy()
    b = np.zeros_like(a)
    # Shift the burst so that b is "a" delayed by 5 indices (the offset of a→b is +5).
    b[5:] = a[:-5]
    a = normalize_zscore(a)
    b = normalize_zscore(b)

    windows = hierarchical_cross_correlate(
        a,
        b,
        sample_rate_hz=31.25,  # silero-style VAD rate
        window_init_samples=500,
        window_floor_samples=125,
        thresh_low=0.10,
        thresh_high=0.40,
        thresh_snr=2.0,
    )

    confident = [w for w in windows if w.verdict == WindowVerdict.CONFIDENT_MATCH]
    assert confident, "expected at least one confident match"
    # Offsets must cluster around the known +5
    offsets = [w.offset for w in confident]
    assert all(abs(o - 5) <= 1 for o in offsets), offsets


def test_post_filter_reclassifies_isolated_match_as_orphan() -> None:
    # 5 windows: orphan, orphan, tiny-match (isolated, no neighbour), orphan, orphan.
    from subtitles_ocr.pipeline.alignment.audio import WindowResult

    windows = [
        WindowResult(start=0, end=100, offset=None, peak=0.0, snr=0.0, verdict=WindowVerdict.NO_MATCH),
        WindowResult(start=100, end=200, offset=None, peak=0.0, snr=0.0, verdict=WindowVerdict.NO_MATCH),
        WindowResult(start=200, end=210, offset=42, peak=0.6, snr=4.0, verdict=WindowVerdict.CONFIDENT_MATCH),
        WindowResult(start=210, end=310, offset=None, peak=0.0, snr=0.0, verdict=WindowVerdict.NO_MATCH),
        WindowResult(start=310, end=410, offset=None, peak=0.0, snr=0.0, verdict=WindowVerdict.NO_MATCH),
    ]
    filtered = post_filter_isolated_matches(
        windows,
        sample_rate_hz=10.0,  # 10 samples per second → 10-sample window = 1.0 s
        min_match_s=2.0,  # 10 samples too short
        offset_tolerance_frames=2,
    )
    assert filtered[2].verdict == WindowVerdict.NO_MATCH
    assert filtered[2].offset is None


def test_post_filter_keeps_match_with_coherent_neighbour() -> None:
    from subtitles_ocr.pipeline.alignment.audio import WindowResult

    windows = [
        WindowResult(start=0, end=100, offset=10, peak=0.6, snr=4.0, verdict=WindowVerdict.CONFIDENT_MATCH),
        WindowResult(start=100, end=110, offset=11, peak=0.6, snr=4.0, verdict=WindowVerdict.CONFIDENT_MATCH),
    ]
    filtered = post_filter_isolated_matches(
        windows,
        sample_rate_hz=10.0,
        min_match_s=2.0,
        offset_tolerance_frames=2,
    )
    assert filtered[1].verdict == WindowVerdict.CONFIDENT_MATCH


# =============================================================================
# Pure function tests — phash.py
# =============================================================================


def test_hamming_distance64_known() -> None:
    assert hamming_distance64(0b1010, 0b0011) == 2
    assert hamming_distance64(0, (1 << 64) - 1) == 64
    assert hamming_distance64(0xFF, 0xFF) == 0


def test_refine_phash_perfect_agreement() -> None:
    # Synthetic: fansub frame N and raw frame N+offset have identical hash.
    phasher = FakePhasher(hashes={})
    offset = 7
    for n in range(20):
        h = 0xDEADBEEF + n  # unique
        phasher.hashes[("fansub", n)] = h
        phasher.hashes[("raw", n + offset)] = h
    fs = FakeFrameSource(phasher)

    result = refine_phash(
        fansub_frames=range(20),
        predicted_offset=offset,
        frame_source=fs,
        window=2,
        thresh_agree=10,
    )
    assert result.disagreement_ratio == 0.0
    assert all(off == offset for off in result.per_frame_offsets)


def test_refine_phash_high_disagreement() -> None:
    # Hashes diverge wildly so distance > thresh_agree for every frame.
    phasher = FakePhasher()
    for n in range(20):
        phasher.hashes[("fansub", n)] = 0
        # Force all candidate raws to have all 64 bits flipped vs the fansub
        for k in range(-2, 3):
            phasher.hashes[("raw", n + k)] = (1 << 64) - 1
    fs = FakeFrameSource(phasher)

    result = refine_phash(
        fansub_frames=range(20),
        predicted_offset=0,
        frame_source=fs,
        window=2,
        thresh_agree=10,
    )
    assert result.disagreement_ratio == 1.0


def test_phash_fallback_aligns_via_search() -> None:
    # Build a known monotone offset of +3.
    phasher = FakePhasher()
    offset = 3
    for n in range(30):
        h = 1 << (n % 64)
        phasher.hashes[("fansub", n)] = h
        phasher.hashes[("raw", n + offset)] = h
    fs = FakeFrameSource(phasher)

    matches = phash_fallback(
        fansub_frames=range(30),
        raw_total_frames=40,
        frame_source=fs,
        w_initial=4,
        w_min=2,
        w_max=16,
        grow_step=2,
        shrink_step=1,
        thresh_match=4,
    )
    # Every fansub frame should match its raw counterpart at +offset.
    matched = [m for m in matches if m.raw_frame_idx is not None]
    assert len(matched) == 30
    assert all(m.raw_frame_idx == m.fansub_frame_idx + offset for m in matched)


def test_phash_fallback_records_misses_as_orphans() -> None:
    # No raw matches → all orphans.
    phasher = FakePhasher()
    for n in range(10):
        phasher.hashes[("fansub", n)] = 0xAA
        # raw is the opposite (max Hamming distance) at every candidate index
        for k in range(40):
            phasher.hashes[("raw", k)] = ~0xAA & ((1 << 64) - 1)
    fs = FakeFrameSource(phasher)

    matches = phash_fallback(
        fansub_frames=range(10),
        raw_total_frames=40,
        frame_source=fs,
        w_initial=2,
        w_min=2,
        w_max=8,
        grow_step=2,
        shrink_step=1,
        thresh_match=4,
    )
    assert all(m.raw_frame_idx is None for m in matches)


# =============================================================================
# AlignmentStage.run — orchestration
# =============================================================================


def _identity_phasher_for_offset(offset: int, n: int) -> FakePhasher:
    p = FakePhasher()
    for i in range(n):
        h = 1 + (i << 1)
        p.hashes[("fansub", i)] = h
        p.hashes[("raw", i + offset)] = h
    return p


def test_run_phash_only_branch_when_no_audio_tracks(tmp_workdir: Path) -> None:
    globals_ = make_globals(tmp_workdir, fansub_total=30)
    phasher = _identity_phasher_for_offset(2, 35)
    fs = FakeFrameSource(phasher)

    stage = AlignmentStage(
        ffmpeg=FakeFfmpegRunner(),
        vad=FakeVadModel(),
        audio_loader=FakeAudioLoader(),
        frame_source=fs,
        raw_total_frames=35,
    )
    config = AlignmentConfig()
    result = stage.run(globals_, config)

    assert result.method_used == "phash_only"
    # No audio extraction was attempted
    out = tmp_workdir / "02_alignment" / "alignment.json"
    assert out.exists()


def test_run_writes_alignment_json_and_sidecar(tmp_workdir: Path) -> None:
    globals_ = make_globals(tmp_workdir, fansub_total=10)
    fs = FakeFrameSource(_identity_phasher_for_offset(0, 15))

    stage = AlignmentStage(
        ffmpeg=FakeFfmpegRunner(),
        vad=FakeVadModel(),
        audio_loader=FakeAudioLoader(),
        frame_source=fs,
        raw_total_frames=15,
    )
    stage.run(globals_, AlignmentConfig())

    out = tmp_workdir / "02_alignment" / "alignment.json"
    sidecar = tmp_workdir / "02_alignment" / "alignment.meta.json"
    assert out.exists()
    assert sidecar.exists()

    data = json.loads(out.read_text())
    AlignmentResult.model_validate(data)


def test_run_orphan_ratio_too_high_raises(tmp_workdir: Path) -> None:
    # All frames mis-match → 100% orphan
    globals_ = make_globals(tmp_workdir, fansub_total=10)
    phasher = FakePhasher()
    for n in range(10):
        phasher.hashes[("fansub", n)] = 0
    for k in range(20):
        phasher.hashes[("raw", k)] = (1 << 64) - 1
    fs = FakeFrameSource(phasher)

    cfg = AlignmentConfig(orphan_ratio_max=0.30, thresh_match=4)
    stage = AlignmentStage(
        ffmpeg=FakeFfmpegRunner(),
        vad=FakeVadModel(),
        audio_loader=FakeAudioLoader(),
        frame_source=fs,
        raw_total_frames=20,
    )
    with pytest.raises(AlignmentRatioTooLow):
        stage.run(globals_, cfg)


def test_run_user_skip_ranges_excluded_from_denominator(tmp_workdir: Path) -> None:
    # 10 frames at 24 fps, but skip frames [0..5) ("00:00:00-00:00:00.208" ≈ 5 frames).
    # If all remaining 5 are orphan → orphan_ratio over denominator-of-5 = 1.0.
    # If skip frames were not excluded → 5/10 = 0.5 orphan also fails. But denominator
    # behaviour is what we want to verify: pass when, with skip, the ratio computes
    # over the un-skipped slice only.
    # We pick a setup where 4/5 remaining match and 1/5 is orphan = 20% → below 30%.
    globals_ = make_globals(tmp_workdir, fansub_total=10)
    phasher = FakePhasher()
    # Frames 0..4 → in user skip; we set their hashes to 0 (won't match anything).
    # Frames 5..8 match offset 0; frame 9 is orphan.
    for n in range(5):
        phasher.hashes[("fansub", n)] = 0
    for n in range(5, 9):
        h = 1 << (n + 1)
        phasher.hashes[("fansub", n)] = h
        phasher.hashes[("raw", n)] = h
    phasher.hashes[("fansub", 9)] = 0xABCDEF
    # No matching raw frame for 9 within window
    for k in range(15):
        if ("raw", k) not in phasher.hashes:
            phasher.hashes[("raw", k)] = (1 << 64) - 1
    fs = FakeFrameSource(phasher)

    cfg = AlignmentConfig(
        orphan_ratio_max=0.30,
        thresh_match=4,
        w_initial=2,
        w_min=2,
        w_max=4,
    )
    stage = AlignmentStage(
        ffmpeg=FakeFfmpegRunner(),
        vad=FakeVadModel(),
        audio_loader=FakeAudioLoader(),
        frame_source=fs,
        raw_total_frames=15,
        hardsub_skip_ranges=["00:00:00-00:00:00.208"],
    )
    result = stage.run(globals_, cfg)
    # 5 frames skipped → user_skipped_ratio = 0.5
    assert result.user_skipped_ratio == pytest.approx(0.5, abs=0.01)
    # orphan_ratio computed over non-skipped denominator: 1 orphan out of 5 → 0.2
    assert result.orphan_ratio == pytest.approx(0.2, abs=0.05)


def test_run_resume_skips_recomputation(tmp_workdir: Path) -> None:
    globals_ = make_globals(tmp_workdir, fansub_total=10)

    # Pre-populate a valid alignment.json + matching sidecar.
    existing = AlignmentResult(
        fansub_total_frames=10,
        raw_total_frames=15,
        method_used="phash_only",
        aligned_ratio=1.0,
        orphan_ratio=0.0,
        user_skipped_ratio=0.0,
        segments=[
            AlignmentSegment(
                fansub_frame_start=0,
                fansub_frame_end=10,
                raw_frame_start=0,
                raw_frame_end=10,
                offset_frames=0,
                status="ALIGNED",
                confidence_avg=1.0,
            )
        ],
        warnings=[],
    )
    out_path = tmp_workdir / "02_alignment" / "alignment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(existing.model_dump_json())

    # Phasher that would raise if invoked.
    class ExplodingPhasher:
        def hash_frame(self, source, frame_idx):
            raise AssertionError("frame_source must not be called on resume")

    fs = FakeFrameSource(ExplodingPhasher())  # type: ignore[arg-type]

    stage = AlignmentStage(
        ffmpeg=FakeFfmpegRunner(),
        vad=FakeVadModel(),
        audio_loader=FakeAudioLoader(),
        frame_source=fs,
        raw_total_frames=15,
    )
    # Write a sidecar that matches the current cache key.
    cfg = AlignmentConfig()
    sidecar_path = tmp_workdir / "02_alignment" / "alignment.meta.json"
    # First call: should compute, write the sidecar.
    # Then we wipe the phasher behaviour and confirm second call resumes.
    sidecar_path.unlink(missing_ok=True)
    # Build a fresh stage that writes meta deterministically.
    bootstrap_phasher = _identity_phasher_for_offset(0, 15)
    bootstrap_stage = AlignmentStage(
        ffmpeg=FakeFfmpegRunner(),
        vad=FakeVadModel(),
        audio_loader=FakeAudioLoader(),
        frame_source=FakeFrameSource(bootstrap_phasher),
        raw_total_frames=15,
    )
    first = bootstrap_stage.run(globals_, cfg)
    assert sidecar_path.exists()

    # Second call with exploding phasher should not invoke it (resume hit).
    result = stage.run(globals_, cfg)
    assert result.model_dump() == first.model_dump()


def test_run_audio_branch_aligns_via_xcorr(tmp_workdir: Path) -> None:
    """End-to-end audio path: real cross-corr on synthetic VAD probs."""
    globals_ = make_globals(tmp_workdir, fansub_total=20)

    # Synthetic VAD probabilities: 2000 samples with a known offset of 0.
    rng = np.random.default_rng(42)
    base = rng.normal(size=2000).astype(np.float32)
    fansub_probs = base.copy()
    raw_probs = base.copy()

    hardsub_wav = tmp_workdir / "02_alignment" / "hardsub_audio.wav"
    raw_wav = tmp_workdir / "02_alignment" / "raw_audio.wav"

    samples_a = np.zeros(16000, dtype=np.float32)
    samples_b = np.ones(16000, dtype=np.float32) * 0.5  # different sum → different key
    loader = FakeAudioLoader(
        samples_by_path={hardsub_wav: (samples_a, 16000), raw_wav: (samples_b, 16000)}
    )
    vad = FakeVadModel(
        probs_by_key={
            f"{samples_a.shape}-{float(samples_a.sum()):.6f}": fansub_probs,
            f"{samples_b.shape}-{float(samples_b.sum()):.6f}": raw_probs,
        }
    )
    ffmpeg = FakeFfmpegRunner()
    # phash refinement: agreement perfect (all frames match)
    phasher = _identity_phasher_for_offset(0, 30)
    fs = FakeFrameSource(phasher)

    cfg = AlignmentConfig()
    stage = AlignmentStage(
        ffmpeg=ffmpeg,
        vad=vad,
        audio_loader=loader,
        frame_source=fs,
        raw_total_frames=30,
        hardsub_audio_track=0,
        raw_audio_track=1,
    )
    result = stage.run(globals_, cfg)
    assert result.method_used == "audio+phash_refinement"
    assert len(ffmpeg.extract_calls) == 2
    assert result.aligned_ratio > 0.9


def test_run_audio_extract_failure_falls_back_to_phash(tmp_workdir: Path) -> None:
    globals_ = make_globals(tmp_workdir, fansub_total=10)
    ffmpeg = FakeFfmpegRunner(fail_on_track=1)
    fs = FakeFrameSource(_identity_phasher_for_offset(0, 15))

    stage = AlignmentStage(
        ffmpeg=ffmpeg,
        vad=FakeVadModel(),
        audio_loader=FakeAudioLoader(),
        frame_source=fs,
        raw_total_frames=15,
        hardsub_audio_track=0,
        raw_audio_track=1,
    )
    result = stage.run(globals_, AlignmentConfig())
    assert result.method_used == "phash_only"
    # Warning about audio fallback
    assert any("audio" in w.lower() for w in result.warnings)


def test_run_phash_refinement_disagreement_triggers_fallback(tmp_workdir: Path) -> None:
    """2b disagreement > threshold → fallback to 2c with a warning."""
    globals_ = make_globals(tmp_workdir, fansub_total=20)

    rng = np.random.default_rng(7)
    base = rng.normal(size=2000).astype(np.float32)
    samples_a = np.zeros(16000, dtype=np.float32)
    samples_b = np.ones(16000, dtype=np.float32)
    hardsub_wav = tmp_workdir / "02_alignment" / "hardsub_audio.wav"
    raw_wav = tmp_workdir / "02_alignment" / "raw_audio.wav"
    loader = FakeAudioLoader(samples_by_path={
        hardsub_wav: (samples_a, 16000),
        raw_wav: (samples_b, 16000),
    })
    vad = FakeVadModel(probs_by_key={
        f"{samples_a.shape}-{float(samples_a.sum()):.6f}": base.copy(),
        f"{samples_b.shape}-{float(samples_b.sum()):.6f}": base.copy(),
    })

    # Phasher: 2b is queried with predicted_offset=0; every fansub frame
    # disagrees because raws within ±2 are all inverse-hash. The test only
    # asserts that the fallback path is taken (and a warning logged); we keep
    # orphan_ratio_max=1.0 so the resulting full-orphan 2c does not abort.
    phasher = FakePhasher()
    for n in range(30):
        phasher.hashes[("fansub", n)] = 1 << (n % 64)
        phasher.hashes[("raw", n)] = ~(1 << (n % 64)) & ((1 << 64) - 1)
    fs = FakeFrameSource(phasher)

    stage = AlignmentStage(
        ffmpeg=FakeFfmpegRunner(),
        vad=vad,
        audio_loader=loader,
        frame_source=fs,
        raw_total_frames=30,
        hardsub_audio_track=0,
        raw_audio_track=1,
    )
    result = stage.run(
        globals_,
        AlignmentConfig(threshold_disagree=0.30, orphan_ratio_max=1.0),
    )
    # Falls back to phash_only when refinement disagrees badly
    assert result.method_used == "phash_only"
    assert any("disagree" in w.lower() or "fallback" in w.lower() for w in result.warnings)
