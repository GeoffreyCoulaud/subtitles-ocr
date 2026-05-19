# Agent P1.A.3 — retry.py

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0004 §11.1.

Tâche : créer `src/subtitles_ocr/retry.py` avec `RetryConfig` (Pydantic v2) et le décorateur `@retry_method`.

## Contrat

- `RetryConfig(BaseModel)` Pydantic v2 :
  - `max_retries: int`
  - `backoff_seconds: tuple[float, ...]`
  - `@model_validator(mode="after")` vérifie que `len(backoff_seconds) == max_retries`.
- `retry_method(is_retryable: Callable[[Exception], bool])` : décorateur de méthode :
  - Lit `self.retry: RetryConfig`.
  - Réessaye sur exception retryable avec `time.sleep(...)` selon `backoff_seconds[attempt]`.
  - Sur exception non-retryable, propage immédiatement.
  - Sur exhaustion, propage la dernière exception (le caller l'enveloppera).
  - Log d'avertissement à chaque retry via `logging.getLogger(__name__)`.

## Tests (`tests/test_retry.py`)

Utiliser `backoff_seconds=(0.0, 0.0)` pour la vitesse.

- `RetryConfig(max_retries=2, backoff_seconds=(1.0,))` lève `ValidationError` (longueur mismatch).
- Happy path : méthode qui réussit du premier coup, appelée une fois.
- Retry-puis-succès : échoue au 1er appel, réussit au 2e (appelée 2 fois).
- Exhaustion : échoue 3 fois sur 2 retries (3 appels), exception finale propagée.
- Non-retryable : `is_retryable` retourne False → 1 seul appel, propage.
- Vérifier que `time.sleep` est appelé avec les bonnes valeurs (monkeypatch AUTORISÉ ici uniquement parce que c'est stdlib).

## Périmètre strict

Tu ne touches à AUCUN autre fichier.
