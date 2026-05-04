import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from subtitles_ocr.cli import _read_jsonl, cli, _resolve_workers, FILTER_WORKERS_DEFAULT
from subtitles_ocr.models import FrameAnalysis, SubtitleEvent, SubtitleElement


def _el() -> SubtitleElement:
    return SubtitleElement(
        text="X", style="regular", color="#FFFFFF",
        border_color="#000000", position="bottom", alignment="center",
    )


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
    """Creates workdir with manifest + video_info + filtered_manifest. Returns (video, workdir)."""
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
    (workdir / "002-filtered_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return video, workdir


def test_extract_skipped_when_manifest_exists(tmp_path):
    video, workdir = _minimal_workdir(tmp_path)
    with patch("subtitles_ocr.cli.extract_frames") as mock_extract, \
         patch("subtitles_ocr.cli.compute_groups", return_value=[]), \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=[]), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=[]), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        CliRunner().invoke(cli, [str(video), "--workdir", str(workdir), "--output", str(tmp_path / "out.ass")])
    mock_extract.assert_not_called()


def test_phash_skipped_when_groups_exist(tmp_path):
    video, workdir = _minimal_workdir(tmp_path)
    fake_group = {"start_time": 0.0, "end_time": 1.0, "frame": "frames/000001.jpg"}
    (workdir / "003-groups.jsonl").write_text(json.dumps(fake_group) + "\n", encoding="utf-8")

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups") as mock_compute, \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=iter([False])), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=iter([])), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        CliRunner().invoke(cli, [str(video), "--workdir", str(workdir), "--output", str(tmp_path / "out.ass")])
    mock_compute.assert_not_called()


def test_prefilter_is_called_with_all_groups(tmp_path):
    video, workdir = _minimal_workdir(tmp_path)
    # 2 groups with unique frame paths
    groups = [
        {"start_time": 0.0, "end_time": 1.0, "frame": "frames/000001.jpg"},
        {"start_time": 1.0, "end_time": 2.0, "frame": "frames/000002.jpg"},
    ]
    (workdir / "003-groups.jsonl").write_text(
        "\n".join(json.dumps(g) for g in groups) + "\n", encoding="utf-8"
    )

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups"), \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=iter([False, False])) as mock_pf, \
         patch("subtitles_ocr.cli.analyze_groups", return_value=iter([])), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        CliRunner().invoke(cli, [str(video), "--workdir", str(workdir), "--output", str(tmp_path / "out.ass")])
    mock_pf.assert_called_once()


def test_analyze_resumes_from_existing_analysis(tmp_path):
    """ID-based resume: groups with existing analysis entries are skipped."""
    video, workdir = _minimal_workdir(tmp_path)

    # 3 groups with unique frame paths
    groups = [
        {"start_time": float(i), "end_time": float(i + 1), "frame": f"frames/{i:06d}.jpg"}
        for i in range(3)
    ]
    (workdir / "003-groups.jsonl").write_text(
        "\n".join(json.dumps(g) for g in groups) + "\n", encoding="utf-8"
    )
    # filter.jsonl with new "id" format
    (workdir / "004-filter.jsonl").write_text(
        "\n".join(
            json.dumps({"id": f"frames/{i:06d}.jpg", "has_text": True}) for i in range(3)
        ) + "\n",
        encoding="utf-8",
    )
    # analysis.jsonl with "id" field — first 2 already done
    (workdir / "005-analysis.jsonl").write_text(
        "\n".join(
            json.dumps({"id": f"frames/{i:06d}.jpg", "start_time": float(i), "end_time": float(i + 1), "elements": []})
            for i in range(2)
        ) + "\n",
        encoding="utf-8",
    )

    fake_analysis = FrameAnalysis(start_time=2.0, end_time=3.0, elements=[])

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups"), \
         patch("subtitles_ocr.cli.prefilter_groups"), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=iter([fake_analysis])) as mock_analyze, \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        CliRunner().invoke(cli, [str(video), "--workdir", str(workdir), "--output", str(tmp_path / "out.ass")])

    assert mock_analyze.call_count == 1
    assert len(mock_analyze.call_args[0][0]) == 1  # only the 3rd group passed


def test_prefilter_writes_results_incrementally(tmp_path):
    """Successful results are written to filter.jsonl even if a later element crashes."""
    video, workdir = _minimal_workdir(tmp_path)

    groups = [
        {"start_time": float(i), "end_time": float(i + 1), "frame": f"frames/{i:06d}.jpg"}
        for i in range(3)
    ]
    (workdir / "003-groups.jsonl").write_text(
        "\n".join(json.dumps(g) for g in groups) + "\n", encoding="utf-8"
    )

    def partial_prefilter(groups, client, prompt, workers, retry_config=None):
        yield False
        yield True
        raise RuntimeError("simulated crash")

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups"), \
         patch("subtitles_ocr.cli.prefilter_groups", side_effect=partial_prefilter), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=iter([])), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        CliRunner().invoke(cli, [str(video), "--workdir", str(workdir), "--output", str(tmp_path / "out.ass")])

    filter_lines = _read_jsonl(workdir / "004-filter.jsonl")
    assert len(filter_lines) == 2


def test_failed_prefilter_element_not_written_to_jsonl(tmp_path):
    """None results from prefilter_groups are not written to filter.jsonl."""
    video, workdir = _minimal_workdir(tmp_path)

    groups = [
        {"start_time": float(i), "end_time": float(i + 1), "frame": f"frames/{i:06d}.jpg"}
        for i in range(2)
    ]
    (workdir / "003-groups.jsonl").write_text(
        "\n".join(json.dumps(g) for g in groups) + "\n", encoding="utf-8"
    )

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups"), \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=iter([True, None])), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=iter([])), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        result = CliRunner().invoke(cli, [str(video), "--workdir", str(workdir), "--output", str(tmp_path / "out.ass")])

    assert result.exit_code != 0
    filter_lines = _read_jsonl(workdir / "004-filter.jsonl")
    assert len(filter_lines) == 1  # only the successful one


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
        CliRunner().invoke(cli, [
            str(video), "--workdir", str(workdir), "--output", str(tmp_path / "out.ass"),
            "--inference-url", "http://proxy:4000",
        ])

    for call in MockClient.call_args_list:
        assert call.kwargs["host"] == "http://proxy:4000"


def test_resolve_workers_explicit_wins(tmp_path):
    config = tmp_path / "litellm.yaml"
    config.write_text("model_list: []", encoding="utf-8")
    assert _resolve_workers("llava:7b", explicit=7, config=config, default=FILTER_WORKERS_DEFAULT) == 7


def test_resolve_workers_no_config_uses_default():
    assert _resolve_workers("llava:7b", explicit=None, config=None, default=FILTER_WORKERS_DEFAULT) == FILTER_WORKERS_DEFAULT


def test_resolve_workers_config_used_when_no_explicit(tmp_path):
    config = tmp_path / "litellm.yaml"
    with patch("subtitles_ocr.cli.get_workers_from_litellm", return_value=12) as mock:
        result = _resolve_workers("llava:7b", explicit=None, config=config, default=FILTER_WORKERS_DEFAULT)
    assert result == 12
    mock.assert_called_once_with(config, "llava:7b")


def test_skip_filters_frames_in_range(tmp_path):
    video = tmp_path / "v.mkv"
    video.write_bytes(b"fake")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    frames_dir = workdir / "001-frames"
    frames_dir.mkdir()

    manifest = [
        {"path": str(frames_dir / "000001.jpg"), "timestamp": 0.0},
        {"path": str(frames_dir / "000002.jpg"), "timestamp": 0.5},
        {"path": str(frames_dir / "000003.jpg"), "timestamp": 1.0},
        {"path": str(frames_dir / "000004.jpg"), "timestamp": 2.0},
    ]
    (workdir / "001-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (workdir / "001-video_info.json").write_text('{"width": 1920, "height": 1080, "fps": 24.0}', encoding="utf-8")

    with patch("subtitles_ocr.cli.extract_frames"), \
         patch("subtitles_ocr.cli.compute_groups", return_value=[]), \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=[]), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=[]), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        result = CliRunner().invoke(cli, [
            str(video), "--workdir", str(workdir), "--output", str(tmp_path / "out.ass"), "--skip", "0-1",
        ])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    filtered = json.loads((workdir / "002-filtered_manifest.json").read_text(encoding="utf-8"))
    assert len(filtered) == 1
    assert filtered[0]["timestamp"] == 2.0


def test_skip_invalid_range_exits_with_error(tmp_path):
    video = tmp_path / "v.mkv"
    video.write_bytes(b"fake")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    result = CliRunner().invoke(cli, [
        str(video), "--workdir", str(workdir), "--output", str(tmp_path / "out.ass"), "--skip", "abc-xyz",
    ])
    assert result.exit_code != 0


def test_retry_options_accepted(tmp_path):
    """--retry-max-attempts, --retry-base-delay, --retry-max-delay are accepted."""
    video, workdir = _minimal_workdir(tmp_path)
    with patch("subtitles_ocr.cli.compute_groups", return_value=[]), \
         patch("subtitles_ocr.cli.prefilter_groups", return_value=iter([])), \
         patch("subtitles_ocr.cli.analyze_groups", return_value=iter([])), \
         patch("subtitles_ocr.cli.build_ass_content", return_value=""):
        result = CliRunner().invoke(cli, [
            str(video), "--workdir", str(workdir), "--output", str(tmp_path / "out.ass"),
            "--retry-max-attempts", "5",
            "--retry-base-delay", "0.5",
            "--retry-max-delay", "15.0",
        ])
    assert result.exit_code == 0, result.output
