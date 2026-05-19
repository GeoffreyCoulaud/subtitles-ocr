# Agent P1.C.2 — cli.py squelette + conftest + setup_logging

> Lis d'abord [`preamble.md`](./preamble.md).

**Pré-requis** : tout P1.A et P1.B (exceptions, config, meta, io, timing, retry, les 3 protocols).

**Référence** : ADR-0004 §3.3, §8.

Tâche : créer le squelette d'orchestration et l'infra de tests.

## Fichiers à créer

### `src/subtitles_ocr/cli.py`

- `def parse_args(argv: list[str] | None = None) -> tuple[PipelineGlobals, PipelineConfig]`
  - argparse avec TOUS les flags d'ADR-0002 §4 : `--hardsub`, `--raw`, `--out`, `--language`, `--synopsis`, `--workdir`, `--debug-images`, `--ar-strategy`, `--hardsub-audio-track`, `--raw-audio-track`, `--hardsub-skip` (`action="append"`), `--raw-skip` (`action="append"`), `--ocr-device`, `--event-cleanup-model`, `--event-cleanup-parallelism`, `--doc-cleanup-model`, `--doc-cleanup-parallelism`, `--color-cluster-threshold`, `--debug` (log level).
  - Note transitoire : `fps` et dimensions du fansub seront probés via `FfmpegRunner` (en P3.1). En attendant, accepte un `--fps-num`/`--fps-den` optionnel avec defaults qui correspondent à un fake. Commenter en TODO que ces flags seront retirés quand `SubprocessFfmpegRunner` sera dispo.
  - Map argparse → `PipelineGlobals` + `PipelineConfig`.

- `def setup_logging(stdout_level: int, log_file: Path) -> None` — exactement la référence ADR-0004 §8.2 (`TzAwareFormatter`, `ShortNameFilter`, stdout + workdir/pipeline.log append, header `==== run started ... ====`).

- `def build_stages() -> list` — retourne `[]` (sera rempli en P4).

- `def run_pipeline(globals: PipelineGlobals, config: PipelineConfig, stages: list | None = None) -> None` — itère sur `stages or build_stages()`, appelle `stage.run(globals, section_for(config, stage))`. Le paramètre `stages` permet l'injection pour les tests.

- `def main(argv: list[str] | None = None) -> int` — parse, setup logging, try `run_pipeline` / except `PipelineError as e` : log formaté `[stage <e.stage>] <type(e).__name__>: <e.args[0]>. Hint: <e.hint>` puis `return 1`. Autres exceptions propagent (bugs).

### `tests/conftest.py`

- `tmp_workdir(tmp_path) -> Path` : crée `01_conform/`, `02_alignment/`, `04_mask/`, `06_ocr/`, `07_group/`, `08_animation/`, `09_color/`, `10_event_cleanup/`, `11_doc_cleanup/`.
- `mock_globals(tmp_workdir) -> PipelineGlobals` : valeurs factices raisonnables (`fps=Fraction(24, 1)`, 1920×1080, 1000 frames, `debug_images=False`).

### `tests/test_cli.py`

- argparse : un test par flag majeur, asserte que la valeur arrive dans `PipelineGlobals` ou `PipelineConfig`.
- `--hardsub-skip` répétable : 2 occurrences → liste de 2 strings.
- Logging : `setup_logging` crée le fichier log, écrit le header de run.
- Orchestrateur (test happy) :
  ```python
  class FakeStage:
      CONFIG_FIELD = "ocr"
      GLOBALS_USED = ()
      STAGE_VERSION = 1
      def run(self, g, c):
          raise AlignmentRatioTooLow("x", stage="02")
  ```
  `main([...])` retourne 1, log contient `AlignmentRatioTooLow`.
- Orchestrateur (bug) : `class BugStage` qui lève `RuntimeError("bug")` → `main` propage (NE pas catcher).
- Smoke `def test_fixtures_smoke(tmp_workdir, mock_globals): assert (tmp_workdir / "06_ocr").is_dir()`.

## Périmètre strict

Tu ne touches PAS aux modules de stage (vides ou inexistants à ce stade).
