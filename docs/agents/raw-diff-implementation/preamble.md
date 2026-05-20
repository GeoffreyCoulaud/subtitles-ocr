# Préambule commun

À lire (par l'agent) avant toute action. Vaut pour tous les prompts P0–P6.

Tu travailles sur le projet **subtitles-ocr** (extraction de sous-titres hardsubbed par diff avec une vidéo raw). Branche cible : `feat/raw-diff-implementation`.

## Lectures obligatoires avant de coder

1. `CLAUDE.md` à la racine — règles non négociables (uv, TDD, Pydantic v2, `uv run python -m pytest`, pas de scripts print pour vérifier).
2. `docs/ADR-0004-Shared-Infrastructure.md` — conventions transverses (signatures de stage, exceptions, persistence, logging, DI, frame indexing).
3. Les sections d'ADR mentionnées dans ton prompt spécifique.

## Règles de fer

- **TDD obligatoire** : test rouge → code → test vert. Pas de code de production sans test qui échoue d'abord.
- **Pydantic v2 partout** : `model_validate`, `model_dump`, `model_dump_json`, `model_validate_json`.
- **Jamais de monkeypatch pour mocker une dépendance externe** : utilise les Protocols (`LlmClient`, `OcrEngine`, `FfmpegRunner`) avec injection par constructeur. Pour `httpx`, utilise `httpx.MockTransport`. Le monkeypatch sur stdlib (`time.sleep`, `os.fsync`) est toléré quand il est testé.
- **Jamais `uv run pytest`** — toujours `uv run python -m pytest`.
- **Jamais `pip` / `python` direct** — `uv add`, `uv remove`, `uv run`.
- **Pas de scripts qui se contentent d'imprimer pour "vérifier"** : toute vérification est un test pytest avec `assert`.
- **Ne touche pas aux fichiers hors de ton périmètre explicite** (évite les conflits de merge avec les agents parallèles).
- **Pas de commentaires explicatifs** de ce que fait le code ; uniquement des commentaires sur du POURQUOI non-évident.
- **Conventions de frame indexing** (ADR-0004 §9) : zéro-based, intervalles half-open `[start, end)`, `fansub_frame_idx` / `raw_frame_idx`, `fps` est toujours `Fraction`, jamais `float`. Toutes conversions passent par `timing.frame_to_ms` / `ms_to_frame`.

## Commits

Message conventionnel (`feat:`, `test:`, `refactor:`, `chore:`). Suit le style des commits récents (`git log --oneline -10`). Un seul commit par tâche sauf indication contraire dans le prompt spécifique.
