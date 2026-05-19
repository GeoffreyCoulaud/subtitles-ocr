# Agent P3.1 — ConformStage + SubprocessFfmpegRunner

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0002 §3 Stage 1.

**Pré-requis** : P2 terminée.

## Tâche

### 1. `src/subtitles_ocr/ffmpeg/subprocess_runner.py`

Implémente `SubprocessFfmpegRunner` (implémente `FfmpegRunner`) :
- `probe(path)` : `ffprobe -v error -print_format json -show_streams -show_format`. Parse JSON → `VideoMetadata`.
- `transcode(args)` : subprocess `ffmpeg` avec la filter chain de `args`.
- `extract_audio(path, track_index, out)` : subprocess `ffmpeg -map 0:a:{idx} -ac 1 -ar 16000 ...`.

### 2. `pipeline/conform.py` — `ConformStage.run`

- Probe raw via `self.ffmpeg.probe(globals.raw_path)`.
- Probe fansub via `self.ffmpeg.probe(globals.hardsub_path)`.
- Calcule AR fansub et AR raw. Si mismatch et `ar_strategy="error"` → lève `AspectRatioMismatch`.
- Construit `TranscodeArgs` :
  - Filter chain : `scale=W:H:flags=area, format=yuv420p` (no dither), + colorspace conversion si détectée.
  - Audio/subs drop (`-an -sn`).
  - Output codec ffv1, conteneur mkv.
- Vérifie le cache via `BaseMeta`/sidecar : si match, skip transcode.
- Appelle `self.ffmpeg.transcode`.
- Écrit `01_conform/raw.mkv` + `01_conform/raw.meta.json`.

## Dépendance

Au choix : `uv add ffmpeg-python` OU `uv add av` (PyAV). Justifie le choix dans le commit.

## Tests (`tests/pipeline/test_conform.py`)

Utiliser `class FakeFfmpegRunner` qui implémente le Protocol et enregistre les appels :
- AR match : `transcode` appelé exactement 1 fois ; sidecar écrit.
- AR mismatch + `ar_strategy="error"` : lève `AspectRatioMismatch`, pas de `transcode`.
- Re-run après sidecar valide : `transcode` PAS appelé.
- Modification de la mtime du raw → sidecar invalide → `transcode` rappelé.
- `STAGE_VERSION` bump simulé → invalide.
- Filter chain inclut `scale=...:flags=area`, `format=yuv420p`.

## Périmètre strict

Tu ne touches PAS aux autres stages. Édite `config.py` UNIQUEMENT pour compléter `ConformConfig` (déjà scaffoldé en P2).
