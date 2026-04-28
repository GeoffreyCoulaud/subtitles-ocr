import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from subtitles_ocr.cli import _read_jsonl, cli


def test_read_jsonl_returns_empty_when_file_missing(tmp_path):
    assert _read_jsonl(tmp_path / "missing.jsonl") == []


def test_read_jsonl_reads_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    assert _read_jsonl(p) == ['{"a": 1}', '{"b": 2}']


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert len(_read_jsonl(p)) == 2


def _minimal_workdir(tmp_path: Path) -> tuple[Path, Path]:
    """Crée un workdir avec manifest + video_info existants. Retourne (video, workdir)."""
    video = tmp_path / "v.mkv"
    video.write_bytes(b"fake")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    frames_dir = workdir / "frames"
    frames_dir.mkdir()

    manifest = [{"path": str(frames_dir / "000001.jpg"), "timestamp": 0.0}]
    (workdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (workdir / "video_info.json").write_text(
        '{"width": 1920, "height": 1080, "fps": 24.0}', encoding="utf-8"
    )
    return video, workdir


def test_extract_skipped_when_manifest_exists(tmp_path):
    video, workdir = _minimal_workdir(tmp_path)

    with patch("subtitles_ocr.cli.extract_frames") as mock_extract, \
         patch("subtitles_ocr.cli.compute_groups", return_value=[]), \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=[]), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        runner = CliRunner()
        runner.invoke(cli, [
            str(video), "--workdir", str(workdir),
            "--output", str(tmp_path / "out.ass"),
        ])

    mock_extract.assert_not_called()
