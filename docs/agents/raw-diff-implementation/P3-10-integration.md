# Agent P3.10 — Intégration croisée + cache invalidation

> Lis d'abord [`preamble.md`](./preamble.md).

**Pré-requis** : tous les agents P3.1 à P3.9 ont mergé.

## Tâche

Écris UNIQUEMENT des tests d'intégration dans `tests/integration/`. Tu ne modifies AUCUN module de stage.

### `tests/integration/test_cache_invalidation_chain.py`

- Lance les 9 stages séquentiellement avec fakes (`FakeFfmpeg`, `FakeOcrEngine`, `FakeLlm`) sur un workdir frais.
- Vérifie que tous les sidecars sont écrits.
- Re-run identique : aucun stage ne ré-exécute (assert sur compteurs des fakes).
- Modifie une mtime de fichier source → tous les stages aval ré-exécutent, pas les amont.
- Bumpe `STAGE_VERSION` d'un stage → ce stage + tous les avals invalident, pas les amont.
- Change un champ `NoCacheKey` (ex. `parallelism`) → AUCUN stage ré-exécute.
- Change un champ cache-invalidating (ex. `OcrConfig.language`) → OCR + aval ré-exécutent.

### `tests/integration/test_resume_jsonl_stages.py`

- Lance OCR partiellement (3 frames sur 5), tue le run, re-lance → reprend à frame 4.
- Idem pour EventCleanup.
- Corruption mi-fichier d'un jsonl → `CacheCorruptionError` propagée.

### `tests/integration/test_e2e_smoke.py`

- 5 frames fansub synthétiques + 5 frames raw synthétiques + un fake stack complet → produit un `.ass` parsable par `pysubs2`.
- Tous les artefacts intermédiaires existent dans le workdir.

### Fixtures

Le `conftest.py` du paquet `tests/integration/` fournit les fakes spécifiques (workdir avec vidéos numpy → ffmpeg via `FakeFfmpeg`).

## Périmètre strict

Pas de modifications aux modules de stage. Pas de modifications à `cli.py` (P4 s'en chargera).
