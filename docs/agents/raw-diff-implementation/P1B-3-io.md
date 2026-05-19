# Agent P1.B.3 — io.py (JsonlWriter)

> Lis d'abord [`preamble.md`](./preamble.md).

**Pré-requis** : P1.A terminée (`exceptions.py` existe).

**Référence** : ADR-0004 §5.4.

Tâche : créer `src/subtitles_ocr/io.py` avec `JsonlWriter`.

## Contrat

`class JsonlWriter[T: BaseModel]` (générique Python 3.12+ syntax) :

- `__init__(self, path: Path, model_cls: type[T], fsync_every: int = 1)`
- `def resume_index(self) -> int` :
  - Compte les lignes JSON valides du fichier existant.
  - Si une ligne corrompue (JSON invalide) n'est PAS la dernière → lève `CacheCorruptionError`.
  - Si la dernière ligne est tronquée (JSON incomplet sans newline) → la rejette silencieusement, retourne le compte des lignes précédentes (resume reprend après).
- `def iter_persisted(self) -> Iterator[T]` : yield des items déjà persistés.
- `def append(self, item: T) -> None` : écrit `item.model_dump_json() + "\n"`. `fsync` tous les `fsync_every` appends.
- Context manager : `__enter__` ouvre le fichier en mode `"a"` ; `__exit__` `close()` flush + fsync + ferme.
- Pas de buffer interne : chaque `append` écrit immédiatement.

## Tests (`tests/test_io.py`)

Utiliser `class FakeItem(BaseModel): n: int` pour les fixtures.

- Happy path : write 5 items, ferme, relit avec `iter_persisted` → mêmes 5.
- Resume après écriture partielle : write 3, fermer ; rouvrir, `resume_index() == 3`, ajouter 2 de plus, relire → 5 items.
- Dernière ligne tronquée : écrire 3 items + écrire manuellement `'{"n": 4'` (sans newline), rouvrir → `resume_index() == 3`. Append d'un 4e item OK.
- Corruption mi-fichier : 3 items, écrire manuellement `'!!!garbage!!!\n'`, puis 1 item ; rouvrir → lève `CacheCorruptionError`.
- `fsync_every=2` : compteur via `monkeypatch.setattr(os, "fsync", spy)` → après 5 appends, `spy` appelé 2 fois (cycles complets) + 1 fois au close.

## Périmètre strict

Tu ne touches à AUCUN autre fichier.
