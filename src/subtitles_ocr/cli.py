import json
from pathlib import Path

import click

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
@click.option("--filter-model", default="smolvlm2:256m",
              help="Modèle Ollama pour le pré-filtrage (défaut: smolvlm2:256m)")
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
        click.echo(f"[1/6] Extraction des frames vers {frames_dir}...")
        frames, video_info = extract_frames(video, frames_dir)
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
        click.echo("[2/6] Groupement des frames similaires (pHash)...")
        groups = compute_groups(frames)
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
        click.echo(f"[3/6] Pré-filtrage ({filter_model}) — {len(remaining_for_filter)} groupes restants...")
        filter_client = OllamaClient(model=filter_model)
        new_results = prefilter_groups(remaining_for_filter, filter_client, PREFILTER_PROMPT, filter_workers)
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
    click.echo(f"[4/6] Analyse VLM ({model}) — {len(groups)} groupes à traiter...")
    client = OllamaClient(model=model)
    analyses: list[FrameAnalysis] = []
    with analysis_path.open("w", encoding="utf-8") as f:
        for i, (group, has_text) in enumerate(zip(groups, filter_results), 1):
            if not has_text:
                analysis = FrameAnalysis(
                    start_time=group.start_time,
                    end_time=group.end_time,
                    elements=[],
                )
                f.write(analysis.model_dump_json() + "\n")
                analyses.append(analysis)
                continue
            click.echo(f"      [{i}/{len(groups)}] {group.frame.name}...", nl=False)
            try:
                analysis = analyze_group(group, client, SYSTEM_PROMPT)
                f.write(analysis.model_dump_json() + "\n")
                analyses.append(analysis)
                click.echo(f" {len(analysis.elements)} élément(s)")
            except RuntimeError as e:
                click.echo(f" ERREUR: {e}", err=True)
                fallback = FrameAnalysis(
                    start_time=group.start_time,
                    end_time=group.end_time,
                    elements=[],
                )
                f.write(fallback.model_dump_json() + "\n")
                analyses.append(fallback)

    # Étape 5 : groupement temporel
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
