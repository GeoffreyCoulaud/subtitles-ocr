import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from subtitles_ocr.cli import _read_jsonl, cli, _resolve_workers, FILTER_WORKERS_DEFAULT
from subtitles_ocr.models import FrameAnalysis


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
    """Crée un workdir avec manifest + video_info + filtered_manifest existants. Retourne (video, workdir)."""
    video = tmp_path / "v.mkv"
    video.write_bytes(b"fake")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    frames_dir = workdir / "001-frames"
    frames_dir.mkdir()

    manifest = [{"path": str(frames_dir / "000001.jpg"), "timestamp": 0.0}]
    (workdir / "001-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (workdir / "001-video_info.json").write_text(
        '{"width": 1920, "height": 1080, "fps": 24.0}', encoding="utf-8"
    )
    # Step 2 (frame filtering) is pre-seeded so tests skip it by default.
    (workdir / "002-filtered_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return video, workdir


def test_extract_skipped_when_manifest_exists(tmp_path):
    video, workdir = _minimal_workdir(tmp_path)

    with patch("subtitles_ocr.cli.extract_frames") as mock_extract, \
         patch("subtitles_ocr.cli.compute_groups", return_value=[]), \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=[]), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=[]), \
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
    (workdir / "003-groups.jsonl").write_text(json.dumps(fake_group) + "\n", encoding="utf-8")

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups") as mock_compute, \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=[False]), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=[]), \
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
    (workdir / "003-groups.jsonl").write_text(
        json.dumps(fake_group) + "\n" + json.dumps(fake_group) + "\n",
        encoding="utf-8",
    )

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups"), \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=[False, False]) as mock_pf, \
         patch("subtitles_ocr.cli.analyze_groups", return_value=[]) as mock_analyze, \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        runner = CliRunner()
        runner.invoke(cli, [
            str(video), "--workdir", str(workdir),
            "--output", str(tmp_path / "out.ass"),
        ])

    mock_pf.assert_called_once()
    # analyze_groups est appelé (même avec tous les groupes filtrés, il retourne des analyses vides)
    mock_analyze.assert_called_once()


def test_analyze_resumes_from_existing_analysis(tmp_path):
    """Si analysis.jsonl a N lignes, seuls les groupes N+ sont envoyés à analyze_groups."""
    video, workdir = _minimal_workdir(tmp_path)

    # 3 groupes dans groups.jsonl
    fake_group = {"start_time": 0.0, "end_time": 1.0, "frame": "frames/000001.jpg"}
    (workdir / "003-groups.jsonl").write_text(
        "\n".join([json.dumps(fake_group)] * 3) + "\n", encoding="utf-8"
    )
    # filter.jsonl : tous has_text=True
    fake_filter = {"frame": "frames/000001.jpg", "has_text": True}
    (workdir / "004-filter.jsonl").write_text(
        "\n".join([json.dumps(fake_filter)] * 3) + "\n", encoding="utf-8"
    )
    # analysis.jsonl : 2 groupes déjà analysés
    done = {"start_time": 0.0, "end_time": 1.0, "elements": []}
    (workdir / "005-analysis.jsonl").write_text(
        "\n".join([json.dumps(done)] * 2) + "\n", encoding="utf-8"
    )

    fake_analysis = FrameAnalysis(start_time=0.0, end_time=1.0, elements=[])

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups"), \
         patch("subtitles_ocr.cli.prefilter_groups"), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=[fake_analysis]) as mock_analyze, \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        runner = CliRunner()
        runner.invoke(cli, [
            str(video), "--workdir", str(workdir),
            "--output", str(tmp_path / "out.ass"),
        ])

    # analyze_groups called once with the 1 remaining group
    assert mock_analyze.call_count == 1
    assert len(mock_analyze.call_args[0][0]) == 1


def test_prefilter_writes_results_incrementally(tmp_path):
    """Results are flushed to filter.jsonl as they arrive, so a crash mid-run preserves progress."""
    video, workdir = _minimal_workdir(tmp_path)

    fake_group = {"start_time": 0.0, "end_time": 1.0, "frame": "frames/000001.jpg"}
    (workdir / "003-groups.jsonl").write_text(
        "\n".join([json.dumps(fake_group)] * 3) + "\n", encoding="utf-8"
    )

    def partial_prefilter(groups, client, prompt, workers):
        yield False
        yield True
        raise RuntimeError("simulated crash")

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups"), \
         patch("subtitles_ocr.cli.prefilter_groups", side_effect=partial_prefilter), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=[]), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        runner = CliRunner()
        runner.invoke(cli, [
            str(video), "--workdir", str(workdir),
            "--output", str(tmp_path / "out.ass"),
        ])

    # The 2 results yielded before the crash must be persisted
    filter_lines = _read_jsonl(workdir / "004-filter.jsonl")
    assert len(filter_lines) == 2


def test_inference_url_propagated_to_clients(tmp_path):
    video, workdir = _minimal_workdir(tmp_path)
    fake_group = {"start_time": 0.0, "end_time": 1.0, "frame": "frames/000001.jpg"}
    (workdir / "003-groups.jsonl").write_text(json.dumps(fake_group) + "\n", encoding="utf-8")

    with patch("subtitles_ocr.cli.OllamaClient") as MockClient, \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=iter([False])), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=iter([])), \
         patch("subtitles_ocr.cli.group_events", return_value=[]), \
         patch("subtitles_ocr.cli.fuzzy_group_events", return_value=[]), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        runner = CliRunner()
        runner.invoke(cli, [
            str(video), "--workdir", str(workdir),
            "--output", str(tmp_path / "out.ass"),
            "--inference-url", "http://proxy:4000",
        ])

    assert MockClient.call_count >= 1
    for call in MockClient.call_args_list:
        assert call.kwargs["host"] == "http://proxy:4000"


def test_resolve_workers_explicit_wins(tmp_path):
    config = tmp_path / "litellm.yaml"
    config.write_text("model_list: []", encoding="utf-8")
    result = _resolve_workers("llava:7b", explicit=7, config=config, default=FILTER_WORKERS_DEFAULT)
    assert result == 7


def test_resolve_workers_no_config_uses_default():
    result = _resolve_workers("llava:7b", explicit=None, config=None, default=FILTER_WORKERS_DEFAULT)
    assert result == FILTER_WORKERS_DEFAULT


def test_resolve_workers_config_used_when_no_explicit(tmp_path):
    config = tmp_path / "litellm.yaml"
    with patch("subtitles_ocr.cli.get_workers_from_litellm", return_value=12) as mock:
        result = _resolve_workers("llava:7b", explicit=None, config=config, default=FILTER_WORKERS_DEFAULT)
    assert result == 12
    mock.assert_called_once_with(config, "llava:7b")


def test_skip_filters_frames_in_range(tmp_path):
    """When --skip 0-1 is passed, frames with timestamps in [0.0, 1.0] are dropped."""
    video = tmp_path / "v.mkv"
    video.write_bytes(b"fake")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    frames_dir = workdir / "001-frames"
    frames_dir.mkdir()

    # Create frames at timestamps 0.0, 0.5, 1.0, and 2.0
    manifest = [
        {"path": str(frames_dir / "000001.jpg"), "timestamp": 0.0},
        {"path": str(frames_dir / "000002.jpg"), "timestamp": 0.5},
        {"path": str(frames_dir / "000003.jpg"), "timestamp": 1.0},
        {"path": str(frames_dir / "000004.jpg"), "timestamp": 2.0},
    ]
    (workdir / "001-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (workdir / "001-video_info.json").write_text(
        '{"width": 1920, "height": 1080, "fps": 24.0}', encoding="utf-8"
    )
    # Note: do NOT pre-seed 002-filtered_manifest.json — let step 2 run

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups", return_value=[]), \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=[]), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=[]), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        runner = CliRunner()
        result = runner.invoke(cli, [
            str(video), "--workdir", str(workdir),
            "--output", str(tmp_path / "out.ass"),
            "--skip", "0-1",
        ])

    # Verify the CLI succeeded
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Read the filtered manifest and verify only the frame at t=2.0 remains
    filtered_manifest_path = workdir / "002-filtered_manifest.json"
    assert filtered_manifest_path.exists(), "002-filtered_manifest.json should be created"
    filtered = json.loads(filtered_manifest_path.read_text(encoding="utf-8"))
    assert len(filtered) == 1, f"Expected 1 frame, got {len(filtered)}"
    assert filtered[0]["timestamp"] == 2.0, f"Expected timestamp 2.0, got {filtered[0]['timestamp']}"


def test_skip_invalid_range_exits_with_error(tmp_path):
    """Invalid --skip range (abc-xyz) causes CLI to exit with non-zero exit code."""
    video = tmp_path / "v.mkv"
    video.write_bytes(b"fake")
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    runner = CliRunner()
    result = runner.invoke(cli, [
        str(video), "--workdir", str(workdir),
        "--output", str(tmp_path / "out.ass"),
        "--skip", "abc-xyz",
    ])

    # Verify the CLI exited with non-zero exit code
    assert result.exit_code != 0, f"CLI should fail for invalid skip range, but exited with {result.exit_code}"
