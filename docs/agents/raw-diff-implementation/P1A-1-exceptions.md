# Agent P1.A.1 — exceptions.py

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0004 §7.1.

Tâche : créer `src/subtitles_ocr/exceptions.py` avec la hiérarchie d'exceptions de la pipeline.

## Contrat

- `PipelineError(Exception)` avec `__init__(message, *, stage: str, hint: str | None = None)`.
- Sous-classes (vides, héritent juste pour le typage) :
  - `InputProbeError`
  - `AspectRatioMismatch`
  - `AlignmentRatioTooLow`
  - `OcrDeviceInitError`
  - `LlmRetryExhausted`
  - `LlmResponseSchemaError`
  - `LlmPromptTooLarge`
  - `CacheCorruptionError`
- Toutes dans CE fichier uniquement, pas ailleurs.

## Tests (`tests/test_exceptions.py`)

- Toutes les sous-classes héritent bien de `PipelineError`.
- `PipelineError("x", stage="07", hint="y").stage == "07"` et `.hint == "y"`.
- `PipelineError("x", stage="07").hint is None`.
- L'import du module est sans effet de bord.

## Périmètre strict

Tu ne touches à AUCUN autre fichier.
