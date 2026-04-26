import json
import subprocess
from pathlib import Path
from subtitles_ocr.models import Frame, VideoInfo


def parse_video_info(ffprobe_json: str) -> VideoInfo:
    data = json.loads(ffprobe_json)
    stream = data["streams"][0]
    num, den = map(int, stream["r_frame_rate"].split("/"))
    return VideoInfo(
        width=stream["width"],
        height=stream["height"],
        fps=num / den,
    )


def compute_frame_timestamps(paths: list[Path], fps: float) -> list[Frame]:
    return [Frame(path=p, timestamp=i / fps) for i, p in enumerate(paths)]


def get_video_info(video_path: Path) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-select_streams", "v:0",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return parse_video_info(result.stdout)


def extract_frames(video_path: Path, output_dir: Path) -> tuple[list[Frame], VideoInfo]:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_info = get_video_info(video_path)

    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-q:v", "3",
            str(output_dir / "%06d.jpg"),
        ],
        check=True,
    )

    paths = sorted(output_dir.glob("*.jpg"))
    frames = compute_frame_timestamps(paths, video_info.fps)
    return frames, video_info
