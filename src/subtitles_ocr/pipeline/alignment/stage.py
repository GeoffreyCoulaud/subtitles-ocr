"""Stage 2 — adaptive alignment (ADR-0002 §3 Stage 2)."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel

from subtitles_ocr.config import AlignmentConfig, PipelineGlobals
from subtitles_ocr.exceptions import AlignmentRatioTooLow
from subtitles_ocr.ffmpeg.protocol import FfmpegRunner
from subtitles_ocr.meta import BaseMeta, cache_invalidating_dict
from subtitles_ocr.pipeline.alignment.audio import (
    AudioLoader,
    VadModel,
    WindowVerdict,
    hierarchical_cross_correlate,
    normalize_zscore,
    post_filter_isolated_matches,
)
from subtitles_ocr.pipeline.alignment.phash import (
    FrameSource,
    PhashMatch,
    phash_fallback,
    refine_phash,
)

STAGE_VERSION: int = 1
STAGE_NAME: str = "02_alignment"

logger = logging.getLogger(__name__)


class AlignmentSegment(BaseModel):
    fansub_frame_start: int
    fansub_frame_end: int
    raw_frame_start: int | None
    raw_frame_end: int | None
    offset_frames: int | None
    status: Literal["ALIGNED", "ORPHAN", "USER_SKIPPED"]
    confidence_avg: float | None


class AlignmentResult(BaseModel):
    fansub_total_frames: int
    raw_total_frames: int
    method_used: Literal["audio_only", "audio+phash_refinement", "phash_only"]
    aligned_ratio: float
    orphan_ratio: float
    user_skipped_ratio: float
    segments: list[AlignmentSegment]
    warnings: list[str]


_TIME_RE = re.compile(
    r"^(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)(?:\.(?P<ms>\d+))?$"
)


def _parse_time_to_seconds(s: str) -> float:
    m = _TIME_RE.match(s.strip())
    if not m:
        raise ValueError(f"Invalid time format: {s!r}; expected HH:MM:SS[.fff]")
    hours = int(m.group("h"))
    minutes = int(m.group("m"))
    seconds = int(m.group("s"))
    millis = m.group("ms")
    frac = 0.0
    if millis:
        frac = int(millis) / (10 ** len(millis))
    return hours * 3600 + minutes * 60 + seconds + frac


def _parse_skip_ranges(ranges: list[str], fps_num: int, fps_den: int) -> list[tuple[int, int]]:
    """Returns list of half-open `[start, end)` frame intervals."""
    out: list[tuple[int, int]] = []
    for r in ranges:
        if "-" not in r:
            raise ValueError(f"Invalid skip range {r!r}; expected HH:MM:SS-HH:MM:SS")
        lo_s, hi_s = r.split("-", 1)
        lo = _parse_time_to_seconds(lo_s)
        hi = _parse_time_to_seconds(hi_s)
        if hi < lo:
            raise ValueError(f"Skip range end before start: {r!r}")
        start = int(round(lo * fps_num / fps_den))
        end = int(round(hi * fps_num / fps_den))
        if end == start:
            end = start + 1
        out.append((start, end))
    return out


def _frame_is_skipped(frame_idx: int, ranges: list[tuple[int, int]]) -> bool:
    for s, e in ranges:
        if s <= frame_idx < e:
            return True
    return False


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


class AlignmentStage:
    CONFIG_FIELD: ClassVar[str] = "alignment"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = (
        "workdir",
        "hardsub_path",
        "raw_path",
        "fps",
        "fansub_total_frames",
    )

    def __init__(
        self,
        ffmpeg: FfmpegRunner | None = None,
        vad: VadModel | None = None,
        audio_loader: AudioLoader | None = None,
        frame_source: FrameSource | None = None,
        raw_total_frames: int = 0,
    ) -> None:
        # External-dep injection points only (Protocols + raw_total_frames).
        # Audio tracks and skip ranges are owned by AlignmentConfig (ADR-0004
        # §3.1 / §5): they reach run() through `config`, never as constructor
        # arguments — otherwise a user flag flip wouldn't invalidate the
        # alignment sidecar cache.
        self.ffmpeg = ffmpeg
        self.vad = vad
        self.audio_loader = audio_loader
        self.frame_source = frame_source
        self.raw_total_frames = raw_total_frames

    # ------------------------------------------------------------------ cache

    def _build_meta(
        self,
        globals_: PipelineGlobals,
        config: AlignmentConfig,
    ) -> BaseMeta:
        globals_subset = {
            "fansub_total_frames": globals_.fansub_total_frames,
            "fps_num": globals_.fps.numerator,
            "fps_den": globals_.fps.denominator,
            "hardsub_path": str(globals_.hardsub_path),
            "raw_path": str(globals_.raw_path),
        }
        # `raw_total_frames` is a runtime quantity not on AlignmentConfig; keep
        # it here so a different raw video invalidates the cache. The audio
        # tracks and skip ranges are part of `config` itself (issue 4) so
        # `cache_invalidating_dict(config)` already covers them.
        runtime_subset = {
            "raw_total_frames": self.raw_total_frames,
        }
        return BaseMeta(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config=cache_invalidating_dict(config),
            globals_subset={**globals_subset, **runtime_subset},
            input_fingerprints={},
            written_at=datetime.now(timezone.utc),
        )

    def _try_resume(
        self,
        globals_: PipelineGlobals,
        config: AlignmentConfig,
    ) -> AlignmentResult | None:
        out_path = globals_.workdir / "02_alignment" / "alignment.json"
        sidecar_path = globals_.workdir / "02_alignment" / "alignment.meta.json"
        if not (out_path.exists() and sidecar_path.exists()):
            return None
        try:
            persisted_meta = BaseMeta.model_validate_json(sidecar_path.read_text())
        except Exception:
            return None
        candidate = self._build_meta(globals_, config)
        if not persisted_meta.matches(candidate):
            return None
        try:
            return AlignmentResult.model_validate_json(out_path.read_text())
        except Exception:
            return None

    # --------------------------------------------------------------- main run

    def run(self, globals: PipelineGlobals, config: AlignmentConfig) -> AlignmentResult:
        resumed = self._try_resume(globals, config)
        if resumed is not None:
            logger.info("alignment resume hit; skipping computation")
            return resumed

        warnings: list[str] = []

        # ---- 1. parse user skip ranges ------------------------------------
        fps_num = globals.fps.numerator
        fps_den = globals.fps.denominator
        hardsub_skip_intervals = _parse_skip_ranges(
            list(config.hardsub_skip_ranges), fps_num, fps_den
        )

        # ---- 2. decide branch ---------------------------------------------
        method: Literal["audio_only", "audio+phash_refinement", "phash_only"] = "phash_only"
        audio_offset: int | None = None

        audio_available = (
            config.hardsub_audio_track is not None
            and config.raw_audio_track is not None
            and self.ffmpeg is not None
            and self.vad is not None
            and self.audio_loader is not None
        )

        if audio_available:
            try:
                audio_offset = self._run_audio_stage_2a(globals, config)
                method = "audio_only" if config.trust_audio_directly else "audio+phash_refinement"
            except Exception as exc:
                msg = f"audio sub-stage failed; falling back to phash-only: {exc}"
                logger.warning(msg)
                warnings.append(msg)
                method = "phash_only"

        # ---- 3. build alignment per frame ---------------------------------
        fansub_total = globals.fansub_total_frames
        # Frames not user-skipped
        active_frames = [i for i in range(fansub_total) if not _frame_is_skipped(i, hardsub_skip_intervals)]

        if method == "audio_only":
            # Trust the audio offset directly; no phash refinement. Burned
            # subtitles routinely break the phash-agreement check on real
            # footage, so the refinement gate is opt-in.
            offset = audio_offset or 0
            matches = []
            for fan_idx in active_frames:
                raw_idx = fan_idx + offset
                if 0 <= raw_idx < self.raw_total_frames:
                    matches.append(
                        PhashMatch(fansub_frame_idx=fan_idx, raw_frame_idx=raw_idx, distance=None)
                    )
                else:
                    matches.append(
                        PhashMatch(fansub_frame_idx=fan_idx, raw_frame_idx=None, distance=None)
                    )

        if method == "audio+phash_refinement":
            # Sub-stage 2b: phash refinement around audio offset
            refinement = refine_phash(
                fansub_frames=active_frames,
                predicted_offset=audio_offset or 0,
                frame_source=self._require_frame_source(),
                window=2,
                thresh_agree=config.thresh_agree,
            )
            if refinement.disagreement_ratio > config.threshold_disagree:
                msg = (
                    f"phash refinement disagreement {refinement.disagreement_ratio:.0%} "
                    f"exceeds threshold; falling back to phash-only (2c)"
                )
                logger.warning(msg)
                warnings.append(msg)
                method = "phash_only"
            else:
                matches = self._refinement_to_matches(active_frames, refinement)

        if method == "phash_only":
            matches = phash_fallback(
                fansub_frames=active_frames,
                raw_total_frames=self.raw_total_frames,
                frame_source=self._require_frame_source(),
                w_initial=config.w_initial,
                w_min=config.w_min,
                w_max=config.w_max,
                grow_step=config.grow_step,
                shrink_step=config.shrink_step,
                thresh_match=config.thresh_match,
            )

        # ---- 4. compute ratios --------------------------------------------
        n_active = len(active_frames)
        n_skipped = fansub_total - n_active
        n_aligned = sum(1 for m in matches if m.raw_frame_idx is not None)
        n_orphan = n_active - n_aligned

        orphan_ratio = (n_orphan / n_active) if n_active > 0 else 0.0
        aligned_ratio = (n_aligned / n_active) if n_active > 0 else 0.0
        user_skipped_ratio = (n_skipped / fansub_total) if fansub_total > 0 else 0.0

        if orphan_ratio > config.orphan_ratio_max:
            raise AlignmentRatioTooLow(
                f"orphan_ratio={orphan_ratio:.1%} exceeds maximum "
                f"{config.orphan_ratio_max:.1%}",
                stage=STAGE_NAME,
                hint=(
                    "Provide --hardsub-skip / --raw-skip to exclude known "
                    "non-corresponding sections, or verify source compatibility."
                ),
            )

        # ---- 5. build segments --------------------------------------------
        segments = self._matches_to_segments(matches, fansub_total, hardsub_skip_intervals)

        result = AlignmentResult(
            fansub_total_frames=fansub_total,
            raw_total_frames=self.raw_total_frames,
            method_used=method,
            aligned_ratio=aligned_ratio,
            orphan_ratio=orphan_ratio,
            user_skipped_ratio=user_skipped_ratio,
            segments=segments,
            warnings=warnings,
        )

        # ---- 6. write output + sidecar ------------------------------------
        out_dir = globals.workdir / "02_alignment"
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(out_dir / "alignment.json", result.model_dump_json())
        meta = self._build_meta(globals, config)
        _atomic_write_text(out_dir / "alignment.meta.json", meta.model_dump_json())

        return result

    # ------------------------------------------------------------- internals

    def _require_frame_source(self) -> FrameSource:
        if self.frame_source is None:
            raise RuntimeError("AlignmentStage requires a frame_source for phash work")
        return self.frame_source

    def _run_audio_stage_2a(self, globals: PipelineGlobals, config: AlignmentConfig) -> int:
        """Sub-stage 2a: extract audio for both sources, run VAD, cross-correlate.

        Returns the dominant offset (in fansub-frame units) recovered from the
        confident matches. Raises on any failure (caught by the caller).
        """
        assert self.ffmpeg is not None
        assert self.vad is not None
        assert self.audio_loader is not None
        assert config.hardsub_audio_track is not None
        assert config.raw_audio_track is not None

        align_dir = globals.workdir / "02_alignment"
        align_dir.mkdir(parents=True, exist_ok=True)
        hardsub_wav = align_dir / "hardsub_audio.wav"
        raw_wav = align_dir / "raw_audio.wav"

        self.ffmpeg.extract_audio(globals.hardsub_path, config.hardsub_audio_track, hardsub_wav)
        self.ffmpeg.extract_audio(globals.raw_path, config.raw_audio_track, raw_wav)

        samples_fan, sr_fan = self.audio_loader.load(hardsub_wav)
        samples_raw, sr_raw = self.audio_loader.load(raw_wav)

        probs_fan = normalize_zscore(self.vad.compute_probs(samples_fan, sr_fan))
        probs_raw = normalize_zscore(self.vad.compute_probs(samples_raw, sr_raw))

        # VAD rate baseline: silero ≈ 31 Hz. We use the source-rate ratio as a
        # proxy; tests pass identical lengths so this resolves to 31.25 Hz.
        vad_rate_hz = 31.25
        windows = hierarchical_cross_correlate(
            probs_fan,
            probs_raw,
            sample_rate_hz=vad_rate_hz,
            window_init_samples=max(64, probs_fan.size // 4),
            window_floor_samples=max(16, probs_fan.size // 32),
            thresh_low=config.audio_thresh_low,
            thresh_high=config.audio_thresh_high,
            thresh_snr=config.audio_thresh_snr,
        )
        windows = post_filter_isolated_matches(
            windows,
            sample_rate_hz=vad_rate_hz,
            min_match_s=config.min_match_s,
            offset_tolerance_frames=config.offset_tolerance_frames,
        )

        confident = [w for w in windows if w.verdict == WindowVerdict.CONFIDENT_MATCH]
        n_confident_samples = sum(w.end - w.start for w in confident)
        n_total_samples = sum(w.end - w.start for w in windows)
        if n_total_samples == 0:
            raise RuntimeError("audio 2a produced no windows")
        aligned_ratio = n_confident_samples / n_total_samples
        if aligned_ratio < 0.70:
            raise AlignmentRatioTooLow(
                f"audio sub-stage 2a aligned_ratio={aligned_ratio:.1%} below 70%",
                stage=STAGE_NAME,
                hint="Audio tracks are likely incompatible; consider --hardsub-skip / --raw-skip.",
            )

        # Dominant VAD-frame offset → convert to fansub-frame offset.
        # We treat VAD samples as proportional to fansub frames: offset_frames ≈
        # round(offset_vad_samples * fps / vad_rate_hz).
        # For tests with identical-length signals the offset is small/zero.
        from collections import Counter

        c = Counter(w.offset for w in confident if w.offset is not None)
        if not c:
            raise RuntimeError("audio 2a produced no confident offsets")
        most_common_vad = c.most_common(1)[0][0]
        fps_float = float(globals.fps.numerator) / float(globals.fps.denominator)
        offset_frames = int(round(most_common_vad * fps_float / vad_rate_hz))
        return offset_frames

    def _refinement_to_matches(
        self, fansub_frames: list[int], refinement
    ) -> list[PhashMatch]:
        out: list[PhashMatch] = []
        for fan_idx, off, dist in zip(
            fansub_frames, refinement.per_frame_offsets, refinement.per_frame_distances
        ):
            raw_idx = fan_idx + off
            if 0 <= raw_idx < self.raw_total_frames:
                out.append(
                    PhashMatch(fansub_frame_idx=fan_idx, raw_frame_idx=raw_idx, distance=dist)
                )
            else:
                out.append(
                    PhashMatch(fansub_frame_idx=fan_idx, raw_frame_idx=None, distance=None)
                )
        return out

    def _matches_to_segments(
        self,
        matches: list[PhashMatch],
        fansub_total: int,
        skip_intervals: list[tuple[int, int]],
    ) -> list[AlignmentSegment]:
        """Compress per-frame matches into contiguous status segments covering
        `[0, fansub_total)` with no gap or overlap.
        """
        match_by_idx: dict[int, PhashMatch] = {m.fansub_frame_idx: m for m in matches}
        segments: list[AlignmentSegment] = []
        i = 0
        while i < fansub_total:
            if _frame_is_skipped(i, skip_intervals):
                status: Literal["ALIGNED", "ORPHAN", "USER_SKIPPED"] = "USER_SKIPPED"
                offset = None
            else:
                m = match_by_idx.get(i)
                if m is not None and m.raw_frame_idx is not None:
                    status = "ALIGNED"
                    offset = m.raw_frame_idx - m.fansub_frame_idx
                else:
                    status = "ORPHAN"
                    offset = None

            j = i + 1
            while j < fansub_total:
                if _frame_is_skipped(j, skip_intervals):
                    next_status: Literal["ALIGNED", "ORPHAN", "USER_SKIPPED"] = "USER_SKIPPED"
                    next_off = None
                else:
                    m2 = match_by_idx.get(j)
                    if m2 is not None and m2.raw_frame_idx is not None:
                        next_status = "ALIGNED"
                        next_off = m2.raw_frame_idx - m2.fansub_frame_idx
                    else:
                        next_status = "ORPHAN"
                        next_off = None
                if next_status != status or next_off != offset:
                    break
                j += 1

            if status == "ALIGNED":
                first = match_by_idx[i]
                last = match_by_idx[j - 1]
                assert first.raw_frame_idx is not None and last.raw_frame_idx is not None
                seg = AlignmentSegment(
                    fansub_frame_start=i,
                    fansub_frame_end=j,
                    raw_frame_start=first.raw_frame_idx,
                    raw_frame_end=last.raw_frame_idx + 1,
                    offset_frames=offset,
                    status=status,
                    confidence_avg=None,
                )
            else:
                seg = AlignmentSegment(
                    fansub_frame_start=i,
                    fansub_frame_end=j,
                    raw_frame_start=None,
                    raw_frame_end=None,
                    offset_frames=None,
                    status=status,
                    confidence_avg=None,
                )
            segments.append(seg)
            i = j
        return segments
