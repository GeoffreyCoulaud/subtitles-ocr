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
         patch("subtitles_ocr.cli.analyze_group"), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        runner = CliRunner()
        runner.invoke(cli, [
            str(video), "--workdir", str(workdir),
            "--output", str(tmp_path / "out.ass"),
        ])

    mock_extract.assert_not_called()


def test_phash_skipped_when_groups_exist(tmp_path):
    video, workdir = _minimal_workdir(tmp_path)
    fake_group = {"start_time": 0.0, "end_time": 1.0, "frame": "frames/000001.jpg"}
    (workdir / "groups.jsonl").write_text(json.dumps(fake_group) + "\n", encoding="utf-8")

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups") as mock_compute, \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=[False]), \
         patch("subtitles_ocr.cli.analyze_group"), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        runner = CliRunner()
        runner.invoke(cli, [
            str(video), "--workdir", str(workdir),
            "--output", str(tmp_path / "out.ass"),
        ])

    mock_compute.assert_not_called()


def test_prefilter_is_called_with_all_groups(tmp_path):
    video, workdir = _minimal_workdir(tmp_path)
    # groups.jsonl avec 2 groupes
    fake_group = {"start_time": 0.0, "end_time": 1.0, "frame": "frames/000001.jpg"}
    (workdir / "groups.jsonl").write_text(
        json.dumps(fake_group) + "\n" + json.dumps(fake_group) + "\n",
        encoding="utf-8",
    )

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups"), \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=[False, False]) as mock_pf, \
         patch("subtitles_ocr.cli.analyze_group") as mock_analyze, \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        runner = CliRunner()
        runner.invoke(cli, [
            str(video), "--workdir", str(workdir),
            "--output", str(tmp_path / "out.ass"),
        ])

    mock_pf.assert_called_once()
    # analyze_group ne doit pas être appelé si tous les groupes sont filtrés
    mock_analyze.assert_not_called()


def test_analyze_resumes_from_existing_analysis(tmp_path):
    """Si analysis.jsonl a N lignes, seuls les groupes N+ sont envoyés à analyze_group."""
    video, workdir = _minimal_workdir(tmp_path)

    # 3 groupes dans groups.jsonl
    fake_group = {"start_time": 0.0, "end_time": 1.0, "frame": "frames/000001.jpg"}
    (workdir / "groups.jsonl").write_text(
        "\n".join([json.dumps(fake_group)] * 3) + "\n", encoding="utf-8"
    )
    # filter.jsonl : tous has_text=True
    fake_filter = {"frame": "frames/000001.jpg", "has_text": True}
    (workdir / "filter.jsonl").write_text(
        "\n".join([json.dumps(fake_filter)] * 3) + "\n", encoding="utf-8"
    )
    # analysis.jsonl : 2 groupes déjà analysés
    done = {"start_time": 0.0, "end_time": 1.0, "elements": []}
    (workdir / "analysis.jsonl").write_text(
        "\n".join([json.dumps(done)] * 2) + "\n", encoding="utf-8"
    )

    mock_analysis = MagicMock()
    mock_analysis.start_time = 0.0
    mock_analysis.end_time = 1.0
    mock_analysis.elements = []
    mock_analysis.model_dump_json.return_value = json.dumps(done)

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups"), \
         patch("subtitles_ocr.cli.prefilter_groups"), \
         patch("subtitles_ocr.cli.analyze_group", return_value=mock_analysis) as mock_analyze, \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        runner = CliRunner()
        runner.invoke(cli, [
            str(video), "--workdir", str(workdir),
            "--output", str(tmp_path / "out.ass"),
        ])

    # Seul le 3e groupe doit être analysé
    assert mock_analyze.call_count == 1
