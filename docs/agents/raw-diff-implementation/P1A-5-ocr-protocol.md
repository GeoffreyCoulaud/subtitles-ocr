# Agent P1.A.5 — ocr_engine/protocol.py

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0004 §10.2.

Tâche : créer le sous-package `src/subtitles_ocr/ocr_engine/`.

## Contrat

### `src/subtitles_ocr/ocr_engine/__init__.py`
Ré-exporte `OcrEngine`.

### `src/subtitles_ocr/ocr_engine/protocol.py`
- `class OcrEngine(Protocol)` :
  ```python
  def detect(self, image: np.ndarray) -> list["OcrDetection"]: ...
  ```
- `OcrDetection` n'est PAS défini ici (il sera défini dans `pipeline/ocr.py` par P2).
- Pour éviter la dépendance circulaire à l'import, utiliser `from __future__ import annotations` ou une annotation string. Ajouter un commentaire `# OcrDetection: défini dans pipeline/ocr.py (P2)`.

## Tests

Smoke test d'import optionnel.

## Périmètre strict

Tu ne touches à AUCUN autre fichier hors `src/subtitles_ocr/ocr_engine/`.
