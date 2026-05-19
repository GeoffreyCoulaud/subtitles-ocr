# Agent P2 — Scaffold des schémas Pydantic de tous les stages

> Lis d'abord [`preamble.md`](./preamble.md).

**Pré-requis** : Phases 0, 1.A, 1.B, 1.C terminées.

**Objectif** : rendre la Phase 3 100% parallèle. Tu écris pour CHAQUE stage le module Python avec :

1. `STAGE_VERSION: int = 1` en tête de module.
2. Tous les Pydantic models de l'ADR (`Result`, sous-types) avec champs typés complets.
3. La classe de config `XConfig(BaseModel)` réelle (champs concrets, defaults, `Annotated[..., NoCacheKey]` sur device/parallelism).
4. La `class XStage` avec :
   - `CONFIG_FIELD: ClassVar[str] = "x"`
   - `GLOBALS_USED: ClassVar[tuple[str, ...]] = ("...", ...)`
   - `__init__` minimal avec injection de dépendances optionnelles (`None` defaults — NE pas instancier l'impl prod, laisse le `NotImplementedError`).
   - `def run(self, globals: PipelineGlobals, config: XConfig) -> XResult: raise NotImplementedError`.
5. Édite `src/subtitles_ocr/config.py` pour remplacer chaque `class XConfig(BaseModel): pass` par un import depuis le module de stage.
   - **Si import cyclique** : déplace les `XConfig` dans `config.py` directement et importe-les depuis les modules de stage. Choisis l'option non-cyclique et documente.

## DAG d'imports

Un stage ne peut importer QUE depuis ses prédécesseurs (cf. ADR-0004 §12).

## Modules à scaffolder

### `pipeline/conform.py`
- `ConformConfig` (vide ou minimal — Stage 1 utilise surtout les flags racine).
- `ConformResult` : `raw_conformed_path: Path`, `target_width: int`, `target_height: int`, `pix_fmt: str`.
- `ConformStage(ffmpeg: FfmpegRunner | None = None)`.

### `pipeline/alignment/stage.py` (+ `audio.py`, `phash.py` placeholders)
- `AlignmentConfig` : tous les tunables d'ADR-0002 §3 Stage 2 + §9 avec defaults raisonnables (`audio_thresh_low`, `audio_thresh_high`, `audio_thresh_snr`, `min_match_s`, `offset_tolerance_frames`, `thresh_agree=10`, `threshold_disagree=0.30`, `w_initial`, `w_min`, `w_max`, `grow_step`, `shrink_step`, `thresh_match`, `orphan_ratio_max=0.30`). Marquer `# baseline, à tuner` les défauts non spécifiés.
- `class AlignmentSegment(BaseModel)` (schéma exact ADR-0002 §3 Stage 2).
- `class AlignmentResult(BaseModel)` (schéma exact ADR-0002 §3 Stage 2).
- `class AlignmentStage(ffmpeg: FfmpegRunner | None = None)`.

### `pipeline/frame_processing/iterator.py` (+ `diff.py`, `mask.py`, `compose.py` stubs)
- `FrameProcessingConfig` : `lcn_sigma`, `std_floor`, `mask_smoothing_sigma`, `mask_t_high`, `mask_t_low`, `mask_area_min`, `mask_area_max`, `mask_dilation_iter=1`.
- `@dataclass(frozen=True) class ComposedFrame` : `fansub_frame_idx: int`, `image: np.ndarray`. **PAS Pydantic** (runtime-only, ADR-0004 §3.2).
- `def iter_composed_frames(globals, alignment_result, config, start_at_fansub_idx=0) -> Iterator[ComposedFrame]: raise NotImplementedError`.

### `pipeline/ocr.py`
- `OcrConfig` :
  - `language: str = "latin"`
  - `device: Annotated[Literal["auto","cuda","rocm","cpu"], NoCacheKey] = "auto"`
  - `parallelism: Annotated[int, NoCacheKey] = 1`
  - `chunk_size: int = 500`
- `class OcrDetection(BaseModel)` (schéma exact ADR-0002 §3 Stage 6).
- `class FrameOcrResult(BaseModel)`.
- `class OcrStage(ocr_engine: OcrEngine | None = None)`.

### `pipeline/group.py`
- `GroupConfig` : `text_levenshtein_max: float = 0.2`, `quad_iou_min: float = 0.5`.
- `class SubtitleEvent(BaseModel)` avec `quads_per_frame: dict[int, list[tuple[int, int]]]` (ADR-0003 §4.1) ET `quad_median: list[tuple[int, int]]`.
- `class GroupResult(BaseModel)`.
- `class GroupStage()`.

### `pipeline/animation.py`
- `AnimationConfig` : tous les tunables ADR-0003 §8 avec defaults exacts (`min_move_displacement_px=8`, `move_gap_tolerance_ms=200`, `move_r2_threshold=0.95`, `move_text_levenshtein_max=0.2`, `fade_search_window_ms=1250`, `min_fade_duration_ms=125`, `fade_duration_cap_ms=1000`, `fade_score_fit_range=(0.05, 0.95)`, `fade_fit_r2_threshold=0.7`).
- `class AnimatedEvent(BaseModel)` (schéma exact ADR-0003 §4.2).
- `class AnimationAnalysisResult(BaseModel)`.
- `class AnimationStage()` (impl MVP en P3.9).

### `pipeline/color.py`
- `ColorConfig` : `interior_hue_var_max`, `outline_hue_var_max`, `interior_min_pixels=100`, `pool_ratio_min=0.1`, `pool_ratio_max=10.0`, `crop_padding_pct=0.10`, `erosion_factor=0.4`, `hsv_bins=16`.
- `class EventColors(BaseModel)`.
- `class ColorExtractionResult(BaseModel)`.
- `class ColorStage()`.

### `pipeline/event_cleanup.py`
- `EventCleanupConfig` : `model: str | None = None`, `parallelism: Annotated[int, NoCacheKey] = 4`, `chunk_size: int = 100`.
- `class CleanedEvent(BaseModel)` : `text: str`.
- `class EventCleanupItem(BaseModel)` : `event_id: int`, `cleaned_text: str`, `skipped_llm: bool`.
- `class EventCleanupResult(BaseModel)`.
- `class EventCleanupStage(llm: LlmClient | None = None)`.

### `pipeline/doc_cleanup.py`
- `DocCleanupConfig` : `model: str | None = None`, `parallelism: Annotated[int, NoCacheKey] = 1`.
- `class FinalEvent(BaseModel)` : `event_id: int`, `cleaned_text: str`.
- `class DocCleanupResult(BaseModel)` : `events: list[FinalEvent]`.
- `class DocCleanupStage(llm: LlmClient | None = None)`.

### `pipeline/export.py`
- `ExportConfig` : `default_font: str = "Arial"`, `default_font_size: int = 60`.
- `class ExportStage()`.

## Tests (`tests/pipeline/test_scaffold.py`)

- Pour chaque module : import marche.
- Pour chaque `XResult` ou model Pydantic non-trivial : round-trip `model_validate_json(model_dump_json())` sur une instance minimale.
- `XStage().run(mock_globals, XConfig())` lève `NotImplementedError`.
- `PipelineConfig().ocr.language == "latin"` (un check par sous-config).

## Commit

`feat: scaffold Pydantic schemas for all pipeline stages`.
