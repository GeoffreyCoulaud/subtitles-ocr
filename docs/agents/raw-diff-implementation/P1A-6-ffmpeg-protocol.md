# Agent P1.A.6 — ffmpeg/protocol.py

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0004 §10.3.

Tâche : créer le sous-package `src/subtitles_ocr/ffmpeg/`.

## Contrat

### `src/subtitles_ocr/ffmpeg/__init__.py`
Ré-exporte `FfmpegRunner`, `VideoMetadata`, `TranscodeArgs`.

### `src/subtitles_ocr/ffmpeg/protocol.py`

- `class VideoMetadata(BaseModel)` (champs minimaux, sera complété par P3.1) :
  - `width: int`
  - `height: int`
  - `fps_num: int`
  - `fps_den: int`
  - `total_frames: int`
  - `duration_s: float`
  - `pix_fmt: str`
  - `colorspace: str | None`

- `class TranscodeArgs(BaseModel)` (complétable plus tard par P3.1) :
  - `input_path: Path`
  - `output_path: Path`
  - `filter_chain: str`
  - `target_width: int`
  - `target_height: int`
  - `target_pix_fmt: str`

- `class FfmpegRunner(Protocol)` :
  - `def probe(self, path: Path) -> VideoMetadata: ...`
  - `def transcode(self, args: TranscodeArgs) -> None: ...`
  - `def extract_audio(self, path: Path, track_index: int, out: Path) -> None: ...`

## Tests

Smoke test d'import + round-trip Pydantic `model_validate_json(model_dump_json())` sur `VideoMetadata` et `TranscodeArgs` avec valeurs factices.

## Périmètre strict

Tu ne touches à AUCUN autre fichier hors `src/subtitles_ocr/ffmpeg/`.
