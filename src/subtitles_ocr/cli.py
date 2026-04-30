import json
import sys
import threading
import time
from pathlib import Path

import click
from tqdm import tqdm

from subtitles_ocr.models import Frame, FrameAnalysis, FrameGroup, SubtitleEvent, VideoInfo
from subtitles_ocr.pipeline.extract import extract_frames
from subtitles_ocr.pipeline.filter import compute_groups
from subtitles_ocr.pipeline.prefilter import prefilter_groups
from subtitles_ocr.pipeline.analyze import analyze_group
from subtitles_ocr.pipeline.group import group_events
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
              help="Chemin du fichier .ass de sortie (défaut: <video>.ass)")
@click.option("--workdir", "-w", type=click.Path(path_type=Path), default=None,
              help="Dossier de travail pour les fichiers intermédiaires")
@click.option("--model", "-m", default="qwen3-vl:8b",
              help="Modèle Ollama pour l'analyse (défaut: qwen3-vl:8b)")
@click.option("--filter-model", default="moondream",
              help="Modèle Ollama pour le pré-filtrage (défaut: moondream)")
@click.option("--filter-workers", default=4, type=click.IntRange(min=1),
              help="Workers parallèles pour le pré-filtrage (défaut: 4)")
def cli(
    video: Path,
    output: Path | None,
    workdir: Path | None,
    model: str,
    filter_model: str,
    filter_workers: int,
) -> None:
    """Extrait les sous-titres incrustés d'une vidéo anime et produit un fichier .ass."""
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

    # Étape 1 : extraction
    if manifest_path.exists() and video_info_path.exists():
        click.echo("[1/6] Extraction ignorée (reprise).")
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
        with tqdm(total=None, desc="[1/6] Extraction des frames", unit="it") as pbar:
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
        click.echo(f"      {len(frames)} frames extraites.")

    # Étape 2 : filtrage pHash
    if groups_path.exists():
        click.echo("[2/6] Groupement ignoré (reprise).")
        groups = [FrameGroup.model_validate_json(line) for line in _read_jsonl(groups_path)]
    else:
        groups = compute_groups(
            tqdm(frames, desc="[2/6] Groupement pHash", total=len(frames), unit="frame")
        )
        with groups_path.open("w", encoding="utf-8") as f:
            for g in groups:
                f.write(g.model_dump_json() + "\n")
        click.echo(f"      {len(groups)} groupes trouvés.")

    # Étape 3 : pré-filtrage VLM
    filter_lines = _read_jsonl(filter_path)
    filter_results: list[bool] = [json.loads(line)["has_text"] for line in filter_lines]
    n_filter_done = len(filter_results)
    remaining_for_filter = groups[n_filter_done:]

    if remaining_for_filter:
        filter_client = OllamaClient(model=filter_model)
        new_results = list(tqdm(
            prefilter_groups(remaining_for_filter, filter_client, PREFILTER_PROMPT, filter_workers),
            total=len(remaining_for_filter),
            desc=f"[3/6] Pré-filtrage ({filter_model})",
            unit="groupe",
        ))
        mode = "a" if n_filter_done > 0 else "w"
        with filter_path.open(mode, encoding="utf-8") as f:
            for group, has_text in zip(remaining_for_filter, new_results):
                f.write(json.dumps({"frame": str(group.frame), "has_text": has_text}) + "\n")
        filter_results.extend(new_results)
        kept = sum(filter_results)
        click.echo(f"      {kept}/{len(groups)} groupes conservés pour l'analyse.")
    else:
        click.echo("[3/6] Pré-filtrage ignoré (reprise).")

    # Étape 4 : analyse VLM
    assert len(filter_results) == len(groups), (
        f"filter.jsonl a {len(filter_results)} entrées mais groups.jsonl en a "
        f"{len(groups)} — supprimez filter.jsonl pour relancer le pré-filtrage."
    )
    analysis_lines = _read_jsonl(analysis_path)
    analyses: list[FrameAnalysis] = [FrameAnalysis.model_validate_json(line) for line in analysis_lines]
    n_analysis_done = len(analyses)
    assert n_analysis_done <= len(groups), (
        f"analysis.jsonl a {n_analysis_done} entrées mais groups.jsonl en a "
        f"{len(groups)} — supprimez analysis.jsonl pour relancer l'analyse."
    )
    remaining_groups = groups[n_analysis_done:]
    remaining_filter = filter_results[n_analysis_done:]

    if remaining_groups:
        client = OllamaClient(model=model)
        mode = "a" if n_analysis_done > 0 else "w"
        with analysis_path.open(mode, encoding="utf-8") as f:
            with tqdm(total=len(remaining_groups), desc=f"[4/6] Analyse VLM ({model})", unit="groupe") as pbar:
                for group, has_text in zip(remaining_groups, remaining_filter):
                    if not has_text:
                        analysis = FrameAnalysis(
                            start_time=group.start_time,
                            end_time=group.end_time,
                            elements=[],
                        )
                    else:
                        try:
                            analysis = analyze_group(group, client, SYSTEM_PROMPT)
                            pbar.set_postfix(elements=len(analysis.elements))
                        except RuntimeError as e:
                            tqdm.write(f"ERREUR [{group.frame.name}]: {e}", file=sys.stderr)
                            analysis = FrameAnalysis(
                                start_time=group.start_time,
                                end_time=group.end_time,
                                elements=[],
                            )
                    f.write(analysis.model_dump_json() + "\n")
                    analyses.append(analysis)
                    pbar.update(1)
    else:
        click.echo("[4/6] Analyse ignorée (reprise).")

    # Étape 5 : groupement temporel
    if events_path.exists():
        click.echo("[5/6] Groupement temporel ignoré (reprise).")
        events = [
            SubtitleEvent.model_validate(e)
            for e in json.loads(events_path.read_text(encoding="utf-8"))
        ]
    else:
        click.echo("[5/6] Groupement temporel des événements...")
        events = group_events(analyses)
        events_path.write_text(
            json.dumps([e.model_dump(mode="json") for e in events], indent=2),
            encoding="utf-8",
        )
        click.echo(f"      {len(events)} événements.")

    # Étape 6 : sérialisation
    click.echo(f"[6/6] Écriture du fichier .ass → {output}")
    ass_content = build_ass_content(events, video_info)
    output.write_text(ass_content, encoding="utf-8")

    click.echo(f"\nTerminé. Fichiers intermédiaires dans : {workdir}")
