# Agent P1.A.4 — llm/protocol.py

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0004 §10.1.

Tâche : créer le sous-package `src/subtitles_ocr/llm/`.

## Contrat

### `src/subtitles_ocr/llm/__init__.py`
Ré-exporte `LlmClient`, `LlmCallFailed`.

### `src/subtitles_ocr/llm/protocol.py`
- `T = TypeVar("T", bound=BaseModel)`
- `class LlmClient(Protocol)` :
  ```python
  def complete(self, prompt: str, response_schema: type[T], *, model: str) -> T: ...
  ```
- `class LlmCallFailed(Exception)` (PAS un `PipelineError` — c'est l'erreur du client, le caller convertit en `LlmRetryExhausted`).

## Tests

Pas de tests obligatoires (Protocol pur). Smoke test optionnel qui vérifie que `LlmClient` et `LlmCallFailed` sont importables depuis `subtitles_ocr.llm`.

## Périmètre strict

Tu ne touches à AUCUN autre fichier hors `src/subtitles_ocr/llm/`.
