# Agent P0 — Suppression du code VLM legacy

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0001 §10 et ADR-0004 §14.

Tâche : supprimer le code du pipeline VLM (legacy) qui est remplacé par le nouveau pipeline diff/OCR.

## Fichiers/répertoires à supprimer

- `src/subtitles_ocr/pipeline/prefilter.py`
- `src/subtitles_ocr/vlm/` (récursivement)
- `src/subtitles_ocr/pipeline/filter.py`
- `src/subtitles_ocr/models.py`
- `src/subtitles_ocr/cli.py`
- `src/subtitles_ocr/litellm_config.py`
- `tests/test_prefilter.py`
- `tests/test_vlm_client.py`
- `tests/test_analyze.py`
- `tests/test_reconcile.py`
- `tests/test_fuzzy_group.py`
- `tests/test_filter.py`
- `tests/test_group.py`
- `tests/test_models.py`
- `tests/test_serialize.py`
- `tests/test_cli.py`
- `tests/test_extract.py`
- `tests/test_resume.py`
- `tests/test_skip.py`
- `tests/test_litellm_config.py`
- `tests/test_retry.py` (sera réécrit en P1.A.3)

Aucun fichier ne doit être déplacé hors de cette liste. Si tu trouves un import cassé dans un fichier non listé, signale-le mais ne le corrige pas.

## Critère d'arrêt

- `uv run python -m pytest -q` termine sans erreur de collecte (suite vide ou réduite à `tests/conftest.py` et `tests/fixtures/`).
- `uv run python -c "import subtitles_ocr"` marche (paquet vide acceptable).

## Commit

`chore: delete VLM-pipeline code per ADR-0001 §10` (commit unique).
