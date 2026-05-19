# Agent P3.3 — frame_processing + OcrStage + PaddleOcrEngine

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0002 §3 Stages 3, 4, 5, 6. ADR-0004 §3.2 (OcrStage est l'owner du sidecar qui inclut `FrameProcessingConfig`).

**Pré-requis** : P2 terminée.

## Tâche

### 1. `pipeline/frame_processing/diff.py`
`def compute_diff(fansub: np.ndarray, raw: np.ndarray, config) -> np.ndarray` : grayscale BT.709 luma + LCN (`(pixel - mean_local) / max(std_local, std_floor)`) + Sobel 3×3 + `abs(gmag_fansub - gmag_raw)`, float32.

### 2. `pipeline/frame_processing/mask.py`
`def make_mask(diff_map, config) -> np.ndarray` :
- Gaussian smoothing (σ = `config.mask_smoothing_sigma`).
- Hysteresis thresholding (Canny-style : seeds `> T_high`, accepted iff connected 8-neighbor à un seed).
- Connected-components filter par taille (`AREA_MIN`, `AREA_MAX`).
- Morphological dilation 3×3, 1 itération.

### 3. `pipeline/frame_processing/compose.py`
`def compose(fansub, mask) -> np.ndarray` : `composed = fansub * (mask / 255)[..., None]`, fond noir RGB(0,0,0).

### 4. `pipeline/frame_processing/iterator.py`
`iter_composed_frames` orchestre diff → mask → compose en streaming (yield un `ComposedFrame` par fansub frame ALIGNED). Lit fansub via OpenCV ou PyAV, raw via `01_conform/raw.mkv`. Sous `--debug-images`, sauvegarde PNGs intermédiaires dans `03_diff/debug/`, `04_mask/frames/`, `05_compose/frames/`.

### 5. `src/subtitles_ocr/ocr_engine/paddle.py`
`class PaddleOcrEngine` : appelle PaddleOCR PP-OCRv5 server. Init avec :
```python
PaddleOCR(
    lang=lang,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
)
```
Gère `device` (auto/cuda/rocm/cpu) :
- `auto` : detect, prefer GPU, fallback CPU avec warning.
- `cuda`/`rocm` : hard fail via `OcrDeviceInitError` si init/test échoue.
- `cpu` : baseline.

### 6. `pipeline/ocr.py` — `OcrStage.run`
- Consomme `iter_composed_frames`.
- Appelle `self.ocr_engine.detect` par frame.
- Écrit `06_ocr/results.jsonl` via `JsonlWriter` (`chunk_size=500`, fsync à chaque chunk).
- Resume via `JsonlWriter.resume_index`.
- Sidecar inclut `OcrConfig` ET `FrameProcessingConfig` (les deux invalident le cache).

## Dépendances

```
uv add paddleocr paddlepaddle
```

CPU baseline. Documenter dans le commit que CUDA/ROCm sont expérimentaux.

## Tests (`tests/pipeline/test_frame_processing.py`, `tests/pipeline/test_ocr.py`)

- Diff : 2 images numpy identiques → diff ≈ 0 ; images avec rectangle de glyph → diff > 0 sur les bords du rectangle.
- Mask : diff binarisé donne un mask cohérent ; aire trop petite → filtré ; dilation appliquée.
- Compose : pixels hors mask = (0,0,0) ; pixels in-mask = fansub original.
- `iter_composed_frames` avec un `AlignmentResult` synthétique de 5 frames → produit 5 `ComposedFrame` dans l'ordre.
- `FakeOcrEngine(responses=[...])` : `OcrStage.run` écrit 5 lignes dans le jsonl, sidecar valide.
- Resume : tronque le jsonl à 3 lignes → `OcrStage` reprend à frame 4.
- `--ocr-device=cuda` avec FakeOcrEngine qui lève à l'init → `OcrDeviceInitError`.
- `--ocr-device=auto` avec fail → warning logged, fallback (test via une factory qui simule).

## Périmètre

Tu peux toucher à `config.py` UNIQUEMENT pour compléter `OcrConfig` et `FrameProcessingConfig` (déjà scaffoldés en P2 mais ajouts mineurs possibles). Pas d'autres stages.
