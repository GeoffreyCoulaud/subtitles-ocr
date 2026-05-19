# Agent P3.2 — AlignmentStage (audio + phash)

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0002 §3 Stage 2 (sous-stages 2a audio coarse, 2b phash refinement, 2c phash fallback) + arbre de décision + politique de skip ranges.

**Pré-requis** : P2 terminée, P3.1 mergée si possible (pour `SubprocessFfmpegRunner.extract_audio`).

## Tâche

### 1. `pipeline/alignment/audio.py`
- VAD via `silero-vad` ONNX (signature : VAD probabilities z-score normalisées par source).
- Cross-corrélation hiérarchique 3 passes :
  - Pass 1 : windows ~30–60 s, normalized cross-correlation, record `(offset, peak_height, peak_to_noise)`.
  - Pass 2 : identify suspicious windows.
  - Pass 3 : subdivise récursivement jusqu'à window floor.
- Verdict par window : `peak_height < THRESH_LOW` → `no_match` ; `peak_height ≥ THRESH_HIGH` ET `peak_to_noise ≥ THRESH_SNR` → `confident_match` ; sinon `ambiguous` → subdivise.
- Post-Pass-3 filter : reclassify tiny isolated matches as ORPHAN.

### 2. `pipeline/alignment/phash.py`
- Refinement (sub-stage 2b) avec `cv2.img_hash.PHash` et fenêtre W=2.
- Fallback (sub-stage 2c) avec fenêtre adaptative (`W_INITIAL`, `W_MIN`, `W_MAX`, `grow_step`, `shrink_step`).

### 3. `pipeline/alignment/stage.py` — `AlignmentStage.run`
- Orchestre 2a → 2b → 2c selon l'arbre de décision ADR-0002 §3 Stage 2 :
  - Audio absent / extraction failed → skip 2a/2b, run 2c.
  - 2a `aligned_ratio < 70%` → hard stop.
  - 2b disagreement > 30% → fallback 2c avec warning.
- Respecte `--hardsub-skip` / `--raw-skip` (range parsing "HH:MM:SS-HH:MM:SS"). Ranges exclues du dénominateur du 70%.
- Lève `AlignmentRatioTooLow` si orphan_ratio > 0.30 post-traitement.
- Écrit `02_alignment/alignment.json` (atomique) + sidecar.

## Dépendances

```
uv add silero-vad onnxruntime opencv-contrib-python scipy
```

## Tests (`tests/pipeline/alignment/`)

Signaux numpy synthétiques uniquement, PAS de WAV réel.

- Audio path : sinus + bursts → VAD détecte les bursts → cross-corrélation trouve l'offset connu (signal décalé de N frames).
- Tiny isolated match : insère un faux pic dans une région orphan → post-filter le reclasse en ORPHAN.
- 70% threshold : fixture d'AlignmentResult intermédiaire avec 35% orphans → `AlignmentRatioTooLow`.
- `--*-skip` ranges : zones skippées exclues du dénominateur du ratio.
- Phash fallback : sans flags audio → 2c directement.
- Phash refinement disagreement > 30% → bascule en 2c avec warning.
- Resume : `alignment.json` existant + sidecar valide → skip toute computation.

## Périmètre strict

Tu ne touches PAS aux autres stages.
