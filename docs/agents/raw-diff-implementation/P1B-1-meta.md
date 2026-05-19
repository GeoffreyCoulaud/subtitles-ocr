# Agent P1.B.1 — meta.py

> Lis d'abord [`preamble.md`](./preamble.md).

**Pré-requis** : P1.A terminée (`exceptions.py` existe).

**Référence** : ADR-0004 §4.3, §5.1, §5.2.

Tâche : créer `src/subtitles_ocr/meta.py` avec la machinerie de cache.

## Contrat

- `class NoCacheKey: """marker"""` (PAS un dataclass, juste une classe vide utilisée comme annotation `Annotated[T, NoCacheKey]`). **Vit dans CE fichier** — c'est `config.py` qui l'importera depuis ici.

- `class FileFingerprint(BaseModel)` :
  - `path: Path`
  - `size: int`
  - `mtime: float`
  - `head_tail_hash: str | None = None`
  - `full_hash: str | None = None`

- `def fingerprint(path: Path, *, treat_as_intermediate: bool = False, full_hash_max_bytes: int = 1_000_000) -> FileFingerprint` :
  - `treat_as_intermediate=True` (workdir intermediates) : `mtime + size` seulement.
  - Sinon, taille `<= full_hash_max_bytes` : `mtime + size + full_hash` (SHA-256 du contenu complet).
  - Sinon (taille `> full_hash_max_bytes`) : `mtime + size + head_tail_hash` (SHA-256 de `head_bytes + tail_bytes`, où head/tail = `full_hash_max_bytes` chaque).

- `class BaseMeta(BaseModel)` :
  - `stage_name: str`
  - `stage_version: int`
  - `config: dict`
  - `globals_subset: dict`
  - `input_fingerprints: dict[str, FileFingerprint]`
  - `written_at: datetime`
  - `def matches(self, other: "BaseMeta") -> bool` : compare TOUS les champs SAUF `written_at`.

- `def cache_invalidating_dict(config: BaseModel) -> dict` : itère sur `config.model_fields`, exclut les champs dont `field.metadata` contient une instance de `NoCacheKey`.

## Tests (`tests/test_meta.py`)

- `fingerprint` sur petit fichier (≤1MB) : `full_hash` présent, `head_tail_hash` None.
- `fingerprint` sur grand fichier (>1MB, créer avec `f.seek(2*1024*1024); f.write(b'x')`) : `head_tail_hash` présent, `full_hash` None.
- `fingerprint(treat_as_intermediate=True)` : ni `full_hash` ni `head_tail_hash`.
- `cache_invalidating_dict` exclut bien les champs `Annotated[..., NoCacheKey]` (utiliser un modèle de test minimal local).
- `BaseMeta.matches` ignore `written_at` mais détecte tout autre changement.

## Périmètre strict

Tu ne touches à AUCUN autre fichier.
