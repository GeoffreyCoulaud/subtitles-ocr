# Agent P3.4 — GroupStage

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0002 §3 Stage 7 + ADR-0003 §4.1 (champ `quads_per_frame`).

**Pré-requis** : P2 terminée.

## Tâche

Implémente `GroupStage.run` dans `pipeline/group.py`.

- Lit `06_ocr/results.jsonl` via `JsonlWriter.iter_persisted`.
- Forme des trajectoires par quad selon critère N → N+1 :
  - Levenshtein normalisé `< config.text_levenshtein_max` (default 0.2)
  - ET quad IoU `> config.quad_iou_min` (default 0.5).
- ORPHAN / USER_SKIPPED au milieu d'une trajectoire ne brisent PAS la trajectoire (lit `02_alignment/alignment.json`).
- Une frame sans détection casse la trajectoire (préserve les répétitions artistiques).
- Agrégations par event :
  - `raw_ocr_texts: list[str]` (un par ALIGNED member frame).
  - `raw_ocr_confidences: list[float]`.
  - `quads_per_frame: dict[int, list[tuple[int,int]]]` (clé = `fansub_frame_idx`).
  - `quad_median: list[tuple[int, int]]` (médiane par coordonnée).
  - `member_frame_indices: list[int]`.
- Écrit `07_group/events.json` (atomique) + sidecar.

## Dépendance

```
uv add rapidfuzz
```

## Tests (`tests/pipeline/test_group.py`)

- 3 frames consécutives, même texte + même quad → 1 event.
- 2 events distincts (textes différents) sur la même frame → 2 trajectoires (test critique : top + bottom subs simultanés).
- ORPHAN au milieu d'une trajectoire textuellement continue → 1 event qui enjambe la zone orphan.
- Frame sans détection au milieu → 2 events distincts.
- Levenshtein > seuil → cassure.
- IoU < seuil → cassure.
- `quads_per_frame` correctement peuplé (clé = `fansub_frame_idx`).
- `quad_median` = médiane par coordonnée des quads members.
- Resume : `07_group/events.json` existe + sidecar valide → skip.

## Périmètre strict

Tu ne touches PAS aux autres stages.
