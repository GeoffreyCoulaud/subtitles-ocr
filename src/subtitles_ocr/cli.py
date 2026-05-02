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
from subtitles_ocr.vlm.client import OllamaClient
from subtitles_ocr.vlm.prompt import SYSTEM_PROMPT, PREFILTER_PROMPT


def _read_jsonl(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [s for line in path.read_text(encoding="utf-8").splitlines() if (s := line.strip())]


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Path to the output .ass file (default: <video>.ass)")
@click.option("--workdir", "-w", type=click.Path(path_type=Path), default=None,
              help="Working directory for intermediate files")
@click.option("--model", "-m", default="qwen2.5vl:3b",
              help="Ollama model for analysis (default: qwen2.5vl:3b)")
@click.option("--filter-model", default="llava:7b",
              help="Ollama model for pre-filtering (default: llava:7b)")
@click.option("--filter-workers", default=4, type=click.IntRange(min=1),
              help="Parallel workers for pre-filtering (default: 4)")
@click.option("--analyze-workers", default=1, type=click.IntRange(min=1),
              help="Parallel workers for VLM analysis (default: 1). "
                   "Values > 1 require OLLAMA_NUM_PARALLEL >= value in the Ollama env.")
@click.option("--edge-diff-threshold", default=8.0, type=click.FloatRange(min=0.0),
              help="Edge difference threshold for frame grouping (default: 8.0)")
@click.option("--similarity-threshold", default=0.75, type=click.FloatRange(min=0.0, max=1.0),
              help="Trigram similarity threshold for fuzzy grouping (default: 0.75)")
@click.option("--gap-tolerance", default=0.5, type=click.FloatRange(min=0.0),
              help="Gap tolerance (seconds) between similar events (default: 0.5)")
@click.option("--reconcile-model", default="gemma3:1b-it-qat",
              help="Ollama model for text reconciliation (default: gemma3:1b-it-qat)")
@click.option("--reconcile-workers", default=8, type=click.IntRange(min=1),
              help="Parallel workers for reconciliation (default: 8)")
@click.option("--ollama-host", default="http://localhost:11434",
              help="Base URL of the Ollama server or LiteLLM proxy (default: http://localhost:11434)")
@click.option("--debug", is_flag=True, default=False,
              help="Enable debug logging (VLM model outputs, etc.)")
def cli(
    video: Path,
    output: Path | None,
    workdir: Path | None,
    model: str,
    filter_model: str,
    filter_workers: int,
    analyze_workers: int,
    edge_diff_threshold: float,
    similarity_threshold: float,
    gap_tolerance: float,
    reconcile_model: str,
    reconcile_workers: int,
    ollama_host: str,
    debug: bool,
) -> None:
    """Extract hardcoded subtitles from an anime video and produce a .ass file."""
    if debug:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    if output is None:
        output = video.with_suffix(".ass")
    if workdir is None:
        workdir = video.parent / (video.stem + "_subtitles_ocr")

    workdir.mkdir(parents=True, exist_ok=True)
    frames_dir = workdir / "frames"
    manifest_path = workdir / "manifest.json"
    video_info_path = workdir / "video_info.json"
    groups_path = workdir / "groups.jsonl"
    filter_path = workdir / "filter.jsonl"
    analysis_path = workdir / "analysis.jsonl"
    events_path = workdir / "events.json"
    fuzzy_groups_path = workdir / "fuzzy_groups.jsonl"
    reconciled_path = workdir / "reconciled.jsonl"

    # Step 1: extraction
    if manifest_path.exists() and video_info_path.exists():
        click.echo("[1/8] Extraction skipped (resuming).")
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
        with tqdm(total=None, desc="[1/8] Extracting frames") as pbar:
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

    # Step 2: edge-similarity grouping
    if groups_path.exists():
        click.echo("[2/8] Grouping skipped (resuming).")
        groups = [FrameGroup.model_validate_json(line) for line in _read_jsonl(groups_path)]
    else:
        groups = compute_groups(
            tqdm(frames, desc="[2/8] Grouping", total=len(frames), unit="frame"),
            diff_threshold=edge_diff_threshold,
        )
        with groups_path.open("w", encoding="utf-8") as f:
            for g in groups:
                f.write(g.model_dump_json() + "\n")
        click.echo(f"      {len(groups)} groups found.")

    # Step 3: VLM pre-filtering
    filter_lines = _read_jsonl(filter_path)
    filter_results: list[bool] = [json.loads(line)["has_text"] for line in filter_lines]
    n_filter_done = len(filter_results)
    remaining_for_filter = groups[n_filter_done:]

    if remaining_for_filter:
        filter_client = OllamaClient(model=filter_model, host=ollama_host)
        mode = "a" if n_filter_done > 0 else "w"
        with filter_path.open(mode, encoding="utf-8") as f, logging_redirect_tqdm():
            for group, has_text in zip(
                remaining_for_filter,
                tqdm(
                    prefilter_groups(remaining_for_filter, filter_client, PREFILTER_PROMPT, filter_workers),
                    total=len(remaining_for_filter),
                    desc=f"[3/8] Pre-filtering ({filter_model})",
                    unit="group",
                ),
            ):
                f.write(json.dumps({"frame": str(group.frame), "has_text": has_text}) + "\n")
                filter_results.append(has_text)
        kept = sum(filter_results)
        click.echo(f"      {kept}/{len(groups)} groups kept for analysis.")
    else:
        click.echo("[3/8] Pre-filtering skipped (resuming).")

    # Step 4: VLM analysis
    assert len(filter_results) == len(groups), (
        f"filter.jsonl has {len(filter_results)} entries but groups.jsonl has "
        f"{len(groups)} — delete filter.jsonl to rerun pre-filtering."
    )
    analysis_lines = _read_jsonl(analysis_path)
    analyses: list[FrameAnalysis] = [FrameAnalysis.model_validate_json(line) for line in analysis_lines]
    n_analysis_done = len(analyses)
    assert n_analysis_done <= len(groups), (
        f"analysis.jsonl has {n_analysis_done} entries but groups.jsonl has "
        f"{len(groups)} — delete analysis.jsonl to rerun analysis."
    )
    remaining_groups = groups[n_analysis_done:]
    remaining_filter = filter_results[n_analysis_done:]

    if remaining_groups:
        client = OllamaClient(model=model, host=ollama_host)
        mode = "a" if n_analysis_done > 0 else "w"
        with analysis_path.open(mode, encoding="utf-8") as f, logging_redirect_tqdm():
            for analysis in tqdm(
                analyze_groups(remaining_groups, remaining_filter, client, SYSTEM_PROMPT, analyze_workers),
                total=len(remaining_groups),
                desc=f"[4/8] VLM analysis ({model})",
                unit="group",
            ):
                f.write(analysis.model_dump_json() + "\n")
                analyses.append(analysis)
    else:
        click.echo("[4/8] Analysis skipped (resuming).")

    # Step 5: temporal grouping
    if events_path.exists():
        click.echo("[5/8] Temporal grouping skipped (resuming).")
        events = [
            SubtitleEvent.model_validate(e)
            for e in json.loads(events_path.read_text(encoding="utf-8"))
        ]
    else:
        click.echo("[5/8] Grouping events temporally...")
        events = group_events(analyses)
        events_path.write_text(
            json.dumps([e.model_dump(mode="json") for e in events], indent=2),
            encoding="utf-8",
        )
        click.echo(f"      {len(events)} events.")

    # Step 6: fuzzy grouping
    if fuzzy_groups_path.exists():
        click.echo("[6/8] Fuzzy grouping skipped (resuming).")
        fuzzy_groups = [
            [SubtitleEvent.model_validate(e) for e in json.loads(line)]
            for line in _read_jsonl(fuzzy_groups_path)
        ]
    else:
        click.echo("[6/8] Fuzzy grouping events...")
        fuzzy_groups = fuzzy_group_events(
            events,
            similarity_threshold=similarity_threshold,
            gap_tolerance=gap_tolerance,
        )
        with fuzzy_groups_path.open("w", encoding="utf-8") as f:
            for cluster in fuzzy_groups:
                f.write(json.dumps([e.model_dump(mode="json") for e in cluster]) + "\n")
        click.echo(f"      {len(fuzzy_groups)} fuzzy groups.")

    # Step 7: reconciliation
    reconciled_lines = _read_jsonl(reconciled_path)
    reconciled: list[SubtitleEvent] = [SubtitleEvent.model_validate_json(line) for line in reconciled_lines]
    n_reconciled_done = len(reconciled)
    assert n_reconciled_done <= len(fuzzy_groups), (
        f"reconciled.jsonl has {n_reconciled_done} entries but fuzzy_groups.jsonl has "
        f"{len(fuzzy_groups)} — delete reconciled.jsonl to rerun reconciliation."
    )
    remaining_clusters = fuzzy_groups[n_reconciled_done:]

    if remaining_clusters:
        reconcile_client = OllamaClient(model=reconcile_model, host=ollama_host)
        mode = "a" if n_reconciled_done > 0 else "w"
        with reconciled_path.open(mode, encoding="utf-8") as f, logging_redirect_tqdm():
            for event in tqdm(
                reconcile_groups(remaining_clusters, reconcile_client, reconcile_workers),
                total=len(remaining_clusters),
                desc=f"[7/8] Reconciliation ({reconcile_model})",
                unit="group",
            ):
                f.write(event.model_dump_json() + "\n")
                reconciled.append(event)
    else:
        click.echo("[7/8] Reconciliation skipped (resuming).")

    # Step 8: serialization
    click.echo(f"[8/8] Writing .ass file → {output}")
    ass_content = build_ass_content(reconciled, video_info)
    output.write_text(ass_content, encoding="utf-8")

    click.echo(f"\nDone. Intermediate files in: {workdir}")
