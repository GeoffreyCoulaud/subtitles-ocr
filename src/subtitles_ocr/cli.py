import json
import logging
import threading
import time
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from subtitles_ocr.models import Frame, FrameAnalysis, FrameGroup, SubtitleEvent, VideoInfo
from subtitles_ocr.pipeline.extract import extract_frames
from subtitles_ocr.pipeline.filter import compute_groups
from subtitles_ocr.pipeline.prefilter import prefilter_groups
from subtitles_ocr.pipeline.analyze import analyze_groups
from subtitles_ocr.pipeline.group import group_events
from subtitles_ocr.pipeline.fuzzy_group import fuzzy_group_events
from subtitles_ocr.pipeline.reconcile import reconcile_groups
from subtitles_ocr.pipeline.serialize import build_ass_content
from subtitles_ocr.pipeline.retry import RetryConfig
from subtitles_ocr.pipeline.resume import resume_from_jsonl
from subtitles_ocr.vlm.client import OllamaClient
from subtitles_ocr.vlm.prompt import SYSTEM_PROMPT, PREFILTER_PROMPT
from subtitles_ocr.litellm_config import get_workers_from_litellm
from subtitles_ocr.pipeline.skip import parse_skip_range, normalize_ranges, filter_frames, format_time


def _read_jsonl(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [s for line in path.read_text(encoding="utf-8").splitlines() if (s := line.strip())]


FILTER_WORKERS_DEFAULT = 4
ANALYZE_WORKERS_DEFAULT = 1
RECONCILE_WORKERS_DEFAULT = 8


def _resolve_workers(model: str, explicit: int | None, config: Path | None, default: int) -> int:
    if explicit is not None:
        logging.debug("Workers for %s: %d (explicit)", model, explicit)
        return explicit
    if config is not None:
        count = get_workers_from_litellm(config, model)
        logging.debug("Workers for %s: %d (litellm config)", model, count)
        return count
    logging.debug("Workers for %s: %d (default)", model, default)
    return default


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Path to the output .ass file (default: <video>.ass)")
@click.option("--workdir", "-w", type=click.Path(path_type=Path), default=None,
              help="Working directory for intermediate files")
@click.option("--filter-model", default="llava:7b",
              help="Model for pre-filtering (default: llava:7b)")
@click.option("--filter-workers", default=None, type=click.IntRange(min=1),
              help="Parallel workers for pre-filtering (default: 4)")
@click.option("--analyze-model", default="qwen3-vl:4b",
              help="Model for VLM analysis (default: qwen3-vl:4b)")
@click.option("--analyze-workers", default=None, type=click.IntRange(min=1),
              help="Parallel workers for VLM analysis (default: 1).")
@click.option("--reconcile-model", default="gemma3:1b-it-qat",
              help="Model for text reconciliation (default: gemma3:1b-it-qat)")
@click.option("--reconcile-workers", default=None, type=click.IntRange(min=1),
              help="Parallel workers for reconciliation (default: 8)")
@click.option("--edge-diff-threshold", default=8.0, type=click.FloatRange(min=0.0),
              help="Edge difference threshold for frame grouping (default: 8.0)")
@click.option("--similarity-threshold", default=0.75, type=click.FloatRange(min=0.0, max=1.0),
              help="Trigram similarity threshold for fuzzy grouping (default: 0.75)")
@click.option("--gap-tolerance", default=0.5, type=click.FloatRange(min=0.0),
              help="Gap tolerance (seconds) between similar events (default: 0.5)")
@click.option("--inference-url", default="http://localhost:11434",
              help="Base URL of the OpenAI-compatible inference server (default: http://localhost:11434)")
@click.option("--litellm-config", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a litellm.yaml; auto-derives worker counts per model")
@click.option("--skip", "skip_ranges_raw", multiple=True, metavar="START-END",
              help="Skip frames in this time range (HH:MM:SS, MM:SS, or SS). Can be repeated.")
@click.option("--retry-max-attempts", default=10, type=click.IntRange(min=1),
              help="Max retry attempts per element for LLM calls (default: 10)")
@click.option("--retry-base-delay", default=1.0, type=click.FloatRange(min=0.0),
              help="Base delay in seconds for exponential backoff (default: 1.0)")
@click.option("--retry-max-delay", default=30.0, type=click.FloatRange(min=0.0),
              help="Maximum delay cap in seconds for retry backoff (default: 30.0)")
@click.option("--debug", is_flag=True, default=False,
              help="Enable debug logging (VLM model outputs, etc.)")
def cli(
    video: Path,
    output: Path | None,
    workdir: Path | None,
    analyze_model: str,
    filter_model: str,
    filter_workers: int | None,
    analyze_workers: int | None,
    edge_diff_threshold: float,
    similarity_threshold: float,
    gap_tolerance: float,
    reconcile_model: str,
    reconcile_workers: int | None,
    inference_url: str,
    litellm_config: Path | None,
    skip_ranges_raw: tuple[str, ...],
    retry_max_attempts: int,
    retry_base_delay: float,
    retry_max_delay: float,
    debug: bool,
) -> None:
    """Extract hardcoded subtitles from an anime video and produce a .ass file."""
    if debug:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    skip_ranges: list[tuple[float, float]] = []
    for raw in skip_ranges_raw:
        try:
            skip_ranges.append(parse_skip_range(raw))
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="'--skip'") from e
    skip_ranges = normalize_ranges(skip_ranges)

    if output is None:
        output = video.with_suffix(".ass")
    if workdir is None:
        workdir = video.parent / (video.stem + "_subtitles_ocr")

    filter_workers    = _resolve_workers(filter_model,    filter_workers,    litellm_config, FILTER_WORKERS_DEFAULT)
    analyze_workers   = _resolve_workers(analyze_model,   analyze_workers,   litellm_config, ANALYZE_WORKERS_DEFAULT)
    reconcile_workers = _resolve_workers(reconcile_model, reconcile_workers, litellm_config, RECONCILE_WORKERS_DEFAULT)

    retry_config = RetryConfig(
        max_attempts=retry_max_attempts,
        base_delay=retry_base_delay,
        max_delay=retry_max_delay,
    )

    workdir.mkdir(parents=True, exist_ok=True)
    step = 0

    # Step 1: extraction
    step += 1
    frames_dir = workdir / f"{step:03d}-frames"
    manifest_path = workdir / f"{step:03d}-manifest.json"
    video_info_path = workdir / f"{step:03d}-video_info.json"
    if manifest_path.exists() and video_info_path.exists():
        click.echo("[1/9] Extraction skipped (resuming).")
        frames = [Frame.model_validate(f) for f in json.loads(manifest_path.read_text(encoding="utf-8"))]
        video_info = VideoInfo.model_validate_json(video_info_path.read_text(encoding="utf-8"))
    else:
        result_holder: dict = {}
        exc_holder: dict = {}

        def _run_extract() -> None:
            try:
                result_holder["frames"], result_holder["video_info"] = extract_frames(video, frames_dir)
            except Exception as e:
                exc_holder["exc"] = e

        thread = threading.Thread(target=_run_extract)
        thread.start()
        with tqdm(total=None, desc="[1/9] Extracting frames") as pbar:
            while thread.is_alive():
                pbar.update(1)
                time.sleep(0.1)
            thread.join()
        if "exc" in exc_holder:
            raise exc_holder["exc"]
        frames = result_holder["frames"]
        video_info = result_holder["video_info"]
        manifest_path.write_text(
            json.dumps([f.model_dump(mode="json") for f in frames], indent=2),
            encoding="utf-8",
        )
        video_info_path.write_text(video_info.model_dump_json(indent=2), encoding="utf-8")
        click.echo(f"      {len(frames)} frames extracted.")

    # Step 2: frame filtering
    step += 1
    filtered_manifest_path = workdir / f"{step:03d}-filtered_manifest.json"
    if filtered_manifest_path.exists():
        click.echo("[2/9] Frame filtering skipped (resuming).")
        frames = [Frame.model_validate(f) for f in json.loads(filtered_manifest_path.read_text(encoding="utf-8"))]
    else:
        filtered = filter_frames(frames, skip_ranges)
        filtered_manifest_path.write_text(
            json.dumps([f.model_dump(mode="json") for f in filtered], indent=2),
            encoding="utf-8",
        )
        n_dropped = len(frames) - len(filtered)
        if skip_ranges:
            click.echo(f"[2/9] Frame filtering — {len(skip_ranges)} range(s), {n_dropped} frames dropped, {len(filtered)} kept.")
            for start, end in skip_ranges:
                n = sum(1 for f in frames if start <= f.timestamp <= end)
                click.echo(f"      {format_time(start)}–{format_time(end)}: {n} frames dropped")
        else:
            click.echo(f"[2/9] Frame filtering — no ranges specified ({len(filtered)} frames kept).")
        frames = filtered

    # Step 3: edge-similarity grouping
    step += 1
    groups_path = workdir / f"{step:03d}-groups.jsonl"
    if groups_path.exists():
        click.echo("[3/9] Grouping skipped (resuming).")
        groups = [FrameGroup.model_validate_json(line) for line in _read_jsonl(groups_path)]
    else:
        groups = compute_groups(
            tqdm(frames, desc="[3/9] Grouping", total=len(frames), unit="frame"),
            diff_threshold=edge_diff_threshold,
        )
        with groups_path.open("w", encoding="utf-8") as f:
            for g in groups:
                f.write(g.model_dump_json() + "\n")
        click.echo(f"      {len(groups)} groups found.")

    # Step 4: VLM pre-filtering
    step += 1
    filter_path = workdir / f"{step:03d}-filter.jsonl"
    filter_lines, remaining_for_filter = resume_from_jsonl(
        groups, filter_path, lambda g: str(g.frame)
    )
    filter_results: list[bool] = [json.loads(line)["has_text"] for line in filter_lines]

    if remaining_for_filter:
        filter_client = OllamaClient(model=filter_model, host=inference_url)
        mode = "a" if filter_path.exists() else "w"
        failed_filter = 0
        with filter_path.open(mode, encoding="utf-8") as f, logging_redirect_tqdm():
            for group, has_text in zip(
                remaining_for_filter,
                tqdm(
                    prefilter_groups(remaining_for_filter, filter_client, PREFILTER_PROMPT, filter_workers, retry_config),
                    total=len(remaining_for_filter),
                    desc=f"[4/9] Pre-filtering ({filter_model})",
                    unit="group",
                ),
            ):
                if has_text is None:
                    failed_filter += 1
                else:
                    f.write(json.dumps({"id": str(group.frame), "has_text": has_text}) + "\n")
                    filter_results.append(has_text)
        if failed_filter:
            raise click.ClickException(
                f"[4/9] {failed_filter} group(s) failed pre-filter after max retries. Resume to retry."
            )
        kept = sum(filter_results)
        click.echo(f"      {kept}/{len(groups)} groups kept for analysis.")
    else:
        click.echo("[4/9] Pre-filtering skipped (resuming).")

    # Step 5: VLM analysis
    step += 1
    analysis_path = workdir / f"{step:03d}-analysis.jsonl"
    filter_by_id = {json.loads(line)["id"]: json.loads(line)["has_text"] for line in _read_jsonl(filter_path)}
    analysis_lines, remaining_groups = resume_from_jsonl(
        groups, analysis_path, lambda g: str(g.frame)
    )
    analyses: list[FrameAnalysis] = [FrameAnalysis.model_validate_json(line) for line in analysis_lines]
    remaining_filter = [filter_by_id[str(g.frame)] for g in remaining_groups]

    if remaining_groups:
        client = OllamaClient(model=analyze_model, host=inference_url)
        mode = "a" if analysis_path.exists() else "w"
        failed_analyze = 0
        with analysis_path.open(mode, encoding="utf-8") as f, logging_redirect_tqdm():
            for group, analysis in zip(
                remaining_groups,
                tqdm(
                    analyze_groups(remaining_groups, remaining_filter, client, SYSTEM_PROMPT, analyze_workers, retry_config),
                    total=len(remaining_groups),
                    desc=f"[5/9] VLM analysis ({analyze_model})",
                    unit="group",
                ),
            ):
                if analysis is None:
                    failed_analyze += 1
                else:
                    data = analysis.model_dump(mode="json")
                    data["id"] = str(group.frame)
                    f.write(json.dumps(data) + "\n")
                    analyses.append(analysis)
        if failed_analyze:
            raise click.ClickException(
                f"[5/9] {failed_analyze} group(s) failed analysis after max retries. Resume to retry."
            )
    else:
        click.echo("[5/9] Analysis skipped (resuming).")

    # Step 6: temporal grouping
    step += 1
    events_path = workdir / f"{step:03d}-events.json"
    if events_path.exists():
        click.echo("[6/9] Temporal grouping skipped (resuming).")
        events = [
            SubtitleEvent.model_validate(e)
            for e in json.loads(events_path.read_text(encoding="utf-8"))
        ]
    else:
        click.echo("[6/9] Grouping events temporally...")
        events = group_events(analyses)
        events_path.write_text(
            json.dumps([e.model_dump(mode="json") for e in events], indent=2),
            encoding="utf-8",
        )
        click.echo(f"      {len(events)} events.")

    # Step 7: fuzzy grouping
    step += 1
    fuzzy_groups_path = workdir / f"{step:03d}-fuzzy_groups.jsonl"
    if fuzzy_groups_path.exists():
        click.echo("[7/9] Fuzzy grouping skipped (resuming).")
        fuzzy_groups = [
            [SubtitleEvent.model_validate(e) for e in json.loads(line)]
            for line in _read_jsonl(fuzzy_groups_path)
        ]
    else:
        click.echo("[7/9] Fuzzy grouping events...")
        fuzzy_groups = fuzzy_group_events(
            events,
            similarity_threshold=similarity_threshold,
            gap_tolerance=gap_tolerance,
        )
        with fuzzy_groups_path.open("w", encoding="utf-8") as f:
            for cluster in fuzzy_groups:
                f.write(json.dumps([e.model_dump(mode="json") for e in cluster]) + "\n")
        click.echo(f"      {len(fuzzy_groups)} fuzzy groups.")

    # Step 8: reconciliation
    step += 1
    reconciled_path = workdir / f"{step:03d}-reconciled.jsonl"
    reconciled_lines, remaining_clusters = resume_from_jsonl(
        fuzzy_groups, reconciled_path, lambda cluster: str(cluster[0].start_time)
    )
    reconciled: list[SubtitleEvent] = [SubtitleEvent.model_validate_json(line) for line in reconciled_lines]

    if remaining_clusters:
        reconcile_client = OllamaClient(model=reconcile_model, host=inference_url)
        mode = "a" if reconciled_path.exists() else "w"
        failed_reconcile = 0
        with reconciled_path.open(mode, encoding="utf-8") as f, logging_redirect_tqdm():
            for cluster, event in zip(
                remaining_clusters,
                tqdm(
                    reconcile_groups(remaining_clusters, reconcile_client, reconcile_workers, retry_config),
                    total=len(remaining_clusters),
                    desc=f"[8/9] Reconciliation ({reconcile_model})",
                    unit="group",
                ),
            ):
                if event is None:
                    failed_reconcile += 1
                else:
                    data = event.model_dump(mode="json")
                    data["id"] = str(cluster[0].start_time)
                    f.write(json.dumps(data) + "\n")
                    reconciled.append(event)
        if failed_reconcile:
            raise click.ClickException(
                f"[8/9] {failed_reconcile} cluster(s) failed reconciliation after max retries. Resume to retry."
            )
    else:
        click.echo("[8/9] Reconciliation skipped (resuming).")

    # Step 9: serialization
    click.echo(f"[9/9] Writing .ass file → {output}")
    ass_content = build_ass_content(reconciled, video_info)
    output.write_text(ass_content, encoding="utf-8")

    click.echo(f"\nDone. Intermediate files in: {workdir}")
