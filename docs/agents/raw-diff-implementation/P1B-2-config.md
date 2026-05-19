# Agent P1.B.2 — config.py (squelette)

> Lis d'abord [`preamble.md`](./preamble.md).

**Pré-requis** : P1.A terminée, P1.B.1 terminée (`NoCacheKey` importable depuis `meta.py`).

**Référence** : ADR-0004 §4.1 et §4.2.

Tâche : créer `src/subtitles_ocr/config.py`.

## Contrat

### `PipelineGlobals(BaseModel)` — complet
- `workdir: Path`
- `hardsub_path: Path`
- `raw_path: Path`
- `out_path: Path`
- `fps: Fraction` — nécessite `model_config = ConfigDict(arbitrary_types_allowed=True)` + un `field_serializer` / `field_validator` custom pour la sérialisation JSON.
- `fansub_width: int`, `fansub_height: int`, `fansub_total_frames: int`
- `debug_images: bool = False`

**Test critique** : `model_dump_json()` / `model_validate_json()` round-trip préserve la valeur exacte de la `Fraction`.

### Sous-configs par stage — squelettes vides

Pour CHAQUE stage, déclare un `class XConfig(BaseModel): pass` (vide) :
`ConformConfig`, `AlignmentConfig`, `FrameProcessingConfig`, `OcrConfig`, `GroupConfig`, `AnimationConfig`, `ColorConfig`, `EventCleanupConfig`, `DocCleanupConfig`, `ExportConfig`.

**Ces classes SERONT COMPLÉTÉES par P2 puis P3** ; laisse-les vides pour ne pas créer de conflit.

### `PipelineConfig(BaseModel)` racine

Champs racine :
- `ar_strategy: Literal["error","letterbox","crop"] = "error"`
- `synopsis_path: Path | None = None`
- `color_cluster_threshold: float = 10.0`
- `hardsub_audio_track: int | None = None`
- `raw_audio_track: int | None = None`
- `hardsub_skip_ranges: list[str] = Field(default_factory=list)`
- `raw_skip_ranges: list[str] = Field(default_factory=list)`

Un champ par sous-config :
```python
conform: ConformConfig = Field(default_factory=ConformConfig)
alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
# ... etc pour tous les stages
```

### Helper

`def section_for(config: PipelineConfig, stage) -> BaseModel` : lit `stage.CONFIG_FIELD: ClassVar[str]` et retourne `getattr(config, stage.CONFIG_FIELD)`.

### Re-export

Ré-exporte `NoCacheKey` depuis `meta` pour que les stages l'importent depuis `config`.

## Tests (`tests/test_config.py`)

- `PipelineGlobals` instanciable avec valeurs valides.
- Round-trip JSON de `PipelineGlobals` préserve `Fraction(24000, 1001)` exactement.
- `PipelineConfig()` instanciable sans argument (tous les defaults).
- `section_for` lookup : créer un objet stub `class S: CONFIG_FIELD = "ocr"` ; `section_for(cfg, S()) is cfg.ocr`.

## Périmètre strict

Tu ne touches à AUCUN autre fichier.
