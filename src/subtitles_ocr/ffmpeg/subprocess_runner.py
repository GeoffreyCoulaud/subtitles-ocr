"""Subprocess-based default FfmpegRunner.

Why subprocess (vs ffmpeg-python / PyAV):
    ADR-0002 §3 Stage 1 lists Python bindings as "preferred for progress
    reporting", with subprocess fallback acceptable. Progress reporting is a
    post-MVP concern (no tqdm wired on Stage 1 yet). Subprocess keeps the
    dependency surface minimal — zero new wheels, no version pinning, no
    binding-shape divergence between dev machines and CI.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from subtitles_ocr.exceptions import InputProbeError
from subtitles_ocr.ffmpeg.protocol import TranscodeArgs, VideoMetadata

logger = logging.getLogger(__name__)


def _parse_fps(rate: str) -> tuple[int, int]:
    if "/" in rate:
        num, den = rate.split("/", 1)
        return int(num), int(den)
    return int(rate), 1


class SubprocessFfmpegRunner:
    def __init__(self, ffprobe_bin: str = "ffprobe", ffmpeg_bin: str = "ffmpeg") -> None:
        self.ffprobe_bin = ffprobe_bin
        self.ffmpeg_bin = ffmpeg_bin

    def probe(self, path: Path) -> VideoMetadata:
        cmd = [
            self.ffprobe_bin,
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(path),
        ]
        try:
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise InputProbeError(
                f"ffprobe binary not found: {self.ffprobe_bin}",
                stage="01_conform",
                hint="Install ffmpeg or set PATH so ffprobe is reachable.",
            ) from e
        except subprocess.CalledProcessError as e:
            raise InputProbeError(
                f"ffprobe failed on {path}: {e.stderr.strip()}",
                stage="01_conform",
            ) from e

        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError as e:
            raise InputProbeError(
                f"ffprobe returned invalid JSON for {path}",
                stage="01_conform",
            ) from e

        streams = data.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if not video_streams:
            raise InputProbeError(
                f"no video stream found in {path}",
                stage="01_conform",
            )
        v = video_streams[0]

        fps_num, fps_den = _parse_fps(v.get("r_frame_rate", "0/1"))
        nb_frames_raw = v.get("nb_frames")
        if nb_frames_raw is not None:
            total_frames = int(nb_frames_raw)
        else:
            duration_for_count = float(v.get("duration") or data.get("format", {}).get("duration") or 0.0)
            total_frames = int(duration_for_count * fps_num / fps_den) if fps_den else 0

        duration_s = float(v.get("duration") or data.get("format", {}).get("duration") or 0.0)

        return VideoMetadata(
            width=int(v["width"]),
            height=int(v["height"]),
            fps_num=fps_num,
            fps_den=fps_den or 1,
            total_frames=total_frames,
            duration_s=duration_s,
            pix_fmt=v.get("pix_fmt", ""),
            colorspace=v.get("color_space"),
        )

    def transcode(self, args: TranscodeArgs) -> None:
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-y",
            "-i", str(args.input_path),
            "-an", "-sn",
            "-vf", args.filter_chain,
            "-c:v", "ffv1",
            "-pix_fmt", args.target_pix_fmt,
            str(args.output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise InputProbeError(
                f"ffmpeg binary not found: {self.ffmpeg_bin}",
                stage="01_conform",
            ) from e
        except subprocess.CalledProcessError as e:
            raise InputProbeError(
                f"ffmpeg transcode failed for {args.input_path}: {e.stderr.strip()}",
                stage="01_conform",
            ) from e

    def extract_audio(self, path: Path, track_index: int, out: Path) -> None:
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-y",
            "-i", str(path),
            "-map", f"0:a:{track_index}",
            "-ac", "1",
            "-ar", "16000",
            "-vn", "-sn",
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise InputProbeError(
                f"ffmpeg binary not found: {self.ffmpeg_bin}",
                stage="01_conform",
            ) from e
        except subprocess.CalledProcessError as e:
            raise InputProbeError(
                f"ffmpeg audio extraction failed for {path} track {track_index}: "
                f"{e.stderr.strip()}",
                stage="01_conform",
            ) from e
