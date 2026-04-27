import json
import subprocess
from pathlib import Path
from subtitles_ocr.models import Frame, VideoInfo


def parse_video_info(ffprobe_json: str) -> VideoInfo:
    data = json.loads(ffprobe_json)
    streams = data.get("streams", [])
    if not streams:
        raise ValueError("ffprobe returned no video streams")
    stream = streams[0]
    try:
        num, den = map(int, stream["r_frame_rate"].split("/"))
    except (KeyError, ValueError) as e:
        raise ValueError(f"Cannot parse r_frame_rate from stream: {stream!r}") from e
    return VideoInfo(
        width=stream["width"],
        height=stream["height"],
        fps=num / den,
    )


def compute_frame_timestamps(paths: list[Path], fps: float) -> list[Frame]:
    return [Frame(path=p, timestamp=i / fps) for i, p in enumerate(paths)]


def get_video_info(video_path: Path) -> VideoInfo:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-select_streams", "v:0",
                str(video_path),
            ],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed for {video_path}: {e.stderr or '(no stderr)'}") from e
    try:
        return parse_video_info(result.stdout)
    except ValueError as e:
        raise RuntimeError(f"Cannot parse video info for {video_path}: {e}") from e


def extract_frames(video_path: Path, output_dir: Path) -> tuple[list[Frame], VideoInfo]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear any existing JPEGs to prevent stale files from corrupting timestamps
    for f in output_dir.glob("*.jpg"):
        f.unlink()
    video_info = get_video_info(video_path)

    try:
        subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-q:v", "3",
                str(output_dir / "%06d.jpg"),
            ],
            capture_output=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else "(no stderr)"
        raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr}") from e

    paths = sorted(output_dir.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"ffmpeg produced no frames in {output_dir}")
    frames = compute_frame_timestamps(paths, video_info.fps)
    return frames, video_info
