"""Deterministic de-duplication keys for the Fact-Intent DAG and the intent bridge.

This module owns *every* canonical hash used to suppress duplicate state in
Intentest.  It is deliberately pure (stdlib only: :mod:`hashlib`, :mod:`json`,
:mod:`unicodedata`) so that both the scheduler plane (``core/scheduler``) and the
worker plane (``worker/``) can compute identical keys without importing
SQLAlchemy, networkx or an HTTP client.

Paper / plan contract
---------------------
``intent.dedupe_key``
    ``sha256(tactic | target_entity | sorted(predecessor_fact_ids))``.
    Stored under ``UNIQUE(project_id, dedupe_key)`` so two concurrent explores
    that would perform the same action from the same verified facts collapse
    into a single intent (the second insert is a no-op via
    ``ON CONFLICT DO NOTHING``).

``fact.fact_key``
    ``sha256(kind | json.dumps(non_volatile_attributes, sort_keys=True))``.
    Volatile observation fields (:data:`VOLATILE_ATTRS`) are stripped so that
    re-observing the same endpoint twice does not create a second immutable
    fact (invariant I1: corrections must go through ``supersedes_fact_id``).

``fact.content_hash``
    ``sha256(kind | attributes | value | evidence_hash)`` — used to detect that
    a *superseding* fact genuinely carries different content.

``bridge`` dedupe
    ``sha256(recommendation_id | hash_blocks(B1, B2, B3))`` — Algorithm 3 must
    not inject the same guidance twice for the same recommendation and the same
    three payload blocks.

Design notes
------------
* All hashing is **stable across processes** (no ``PYTHONHASHSEED`` influence):
  JSON is serialised with ``sort_keys=True`` and
  ``separators=(",", ":")`` and every part is UTF-8 encoded explicitly.
* Text parts are Unicode NFKC-normalised and whitespace-collapsed before
  hashing, so ``"Get  /admin"`` and ``"get /admin"`` collide only when they
  genuinely describe the same action (case folding is *opt-in* per helper).
* :class:`SeenSet` provides a bounded in-process LRU for hot-path suppression
  (e.g. the bridge loop inside a single worker) while the database UNIQUE
  constraints remain the authoritative exactly-once mechanism.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "VOLATILE_ATTRS",
    "canonical_json",
    "normalise_text",
    "hash_parts",
    "intent_dedupe_key",
    "dedupe_key_for_intent",
    "fact_dedupe_key",
    "fact_content_hash",
    "bridge_dedupe_key",
    "block_hash",
    "hash_blocks",
    "pattern_dedupe_key",
    "normalise_fact_ids",
    "SeenSet",
    "DedupeStats",
    "suppress_duplicates",
    "is_duplicate_intent",
    "find_duplicate_intent",
]

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Attribute keys that describe *when* something was observed rather than *what*
#: was observed.  They are excluded from ``fact_key`` so that repeated scans of
#: the same target collapse onto the existing immutable fact.
VOLATILE_ATTRS: frozenset[str] = frozenset(
    {
        "timestamp",
        "ts",
        "scanned_at",
        "observed_at",
        "request_id",
        "duration_ms",
        "elapsed_ms",
        "nonce",
        "date",
        "time",
        "created_at",
        "updated_at",
    }
)

#: Separator used between logical parts.  ``\x1f`` (unit separator) cannot occur
#: in normalised text, which prevents ``("a|b", "c")`` from colliding with
#: ``("a", "b|c")``.
_PART_SEP = "\x1f"

_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Canonicalisation helpers
# --------------------------------------------------------------------------- #


def normalise_text(value: Any, *, casefold: bool = False) -> str:
    """Return a canonical, deterministic string form of ``value``.

    ``None`` becomes the empty string.  Unicode is NFKC-normalised, runs of
    whitespace collapse to a single space and the result is stripped.
    ``casefold=True`` additionally lower-cases (used for tactics/entities where
    the paper's vocabulary is case-insensitive).
    """

    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text).strip()
    if casefold:
        text = text.casefold()
    return text


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe projection used by :func:`canonical_json`."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(item) for item in value]
        if isinstance(value, (set, frozenset)):
            try:
                items.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))
            except Exception:  # pragma: no cover - defensive
                items.sort(key=str)
        return items
    # Enums, UUIDs, datetimes, Decimal, ORM scalars.
    enum_value = getattr(value, "value", None)
    if enum_value is not None and not callable(enum_value):
        return _jsonable(enum_value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialise ``value`` with sorted keys and no insignificant whitespace."""

    try:
        return json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("dedupe.canonical_json_failed", extra={"error": repr(exc)})
        return json.dumps(str(value), ensure_ascii=False)


def hash_parts(*parts: Any, length: Optional[int] = None) -> str:
    """SHA-256 over the canonical text of ``parts`` joined by a unit separator.

    ``length`` truncates the hex digest (useful for compact log identifiers);
    it must be a positive integer.
    """

    chunks: list[str] = []
    for part in parts:
        if isinstance(part, (Mapping, list, tuple, set, frozenset)):
            chunks.append(canonical_json(part))
        else:
            chunks.append(normalise_text(part))
    digest = hashlib.sha256(_PART_SEP.join(chunks).encode("utf-8")).hexdigest()
    if length is not None:
        if length <= 0:
            raise ValueError("length must be a positive integer")
        digest = digest[:length]
    return digest


def normalise_fact_ids(fact_ids: Optional[Iterable[Any]]) -> list[str]:
    """Return a sorted, deduplicated list of stringified fact ids.

    Ordering is lexicographic on the *string* form so the value is stable
    regardless of whether callers pass ``uuid.UUID`` objects, integers or plain
    strings.
    """

    if not fact_ids:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in fact_ids:
        text = normalise_text(raw)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    out.sort()
    return out


# --------------------------------------------------------------------------- #
# Intent keys
# --------------------------------------------------------------------------- #


def intent_dedupe_key(
    tactic: Any,
    target_entity: Any,
    predecessor_fact_ids: Optional[Iterable[Any]] = None,
    *,
    action_type: Any = None,
) -> str:
    """Compute ``intent.dedupe_key`` exactly as specified in the paper plan.

    ``sha256(tactic | target_entity | sorted(predecessor_fact_ids))``.

    ``action_type`` is accepted for callers that want to disambiguate a REASON
    from an EXPLORE on the same tactic/target; when it is ``None`` the key
    matches the published definition verbatim (backwards compatible).
    """

    tactic_text = normalise_text(tactic, casefold=True)
    target_text = normalise_text(target_entity, casefold=True)
    ids = normalise_fact_ids(predecessor_fact_ids)
    parts: list[Any] = [tactic_text, target_text, ids]
    if action_type is not None and normalise_text(action_type):
        parts.append(normalise_text(action_type, casefold=True))
    return hash_parts(*parts)


def dedupe_key_for_intent(intent: Any, predecessor_fact_ids: Optional[Iterable[Any]] = None) -> str:
    """Compute the dedupe key for an intent row/dict.

    If ``predecessor_fact_ids`` is not supplied the helper looks for the ids on
    the object itself under ``predecessor_fact_ids`` / ``fact_ids`` /
    ``source_fact_ids`` (attributes or mapping keys).  Intent sources are *not*
    loaded from the database here — this stays a pure function.
    """

    tactic = _attr(intent, "tactic", "action", "action_type_name")
    target = _attr(intent, "target_entity", "target", "entity")
    action_type = _attr(intent, "action_type", "kind", "task_type")
    if predecessor_fact_ids is None:
        predecessor_fact_ids = _attr(intent, "predecessor_fact_ids", "fact_ids", "source_fact_ids")
    return intent_dedupe_key(tactic, target, predecessor_fact_ids, action_type=action_type)


# --------------------------------------------------------------------------- #
# Fact keys
# --------------------------------------------------------------------------- #


def _strip_volatile(attributes: Any, volatile: Iterable[str] = VOLATILE_ATTRS) -> dict[str, Any]:
    """Return ``attributes`` minus volatile observation fields (recursively)."""

    volatile_set = {str(name).casefold() for name in volatile}
    if not isinstance(attributes, Mapping):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in attributes.items():
        key_text = str(key)
        if key_text.casefold() in volatile_set:
            continue
        if isinstance(value, Mapping):
            cleaned[key_text] = _strip_volatile(value, volatile_set)
        elif isinstance(value, (list, tuple)):
            cleaned[key_text] = [
                _strip_volatile(item, volatile_set) if isinstance(item, Mapping) else item for item in value
            ]
        else:
            cleaned[key_text] = value
    return cleaned


def fact_dedupe_key(
    kind: Any,
    attributes: Any = None,
    *,
    volatile: Iterable[str] = VOLATILE_ATTRS,
) -> str:
    """Compute ``fact.fact_key``: ``sha256(kind | sorted-JSON attributes)``.

    Matches :func:`app.db.repositories.fact_repo.FactRepository` so that facts
    inserted through the API and through the worker collapse identically.
    """

    kind_text = normalise_text(kind, casefold=True)
    cleaned = _strip_volatile(attributes, volatile)
    return hash_parts(kind_text, cleaned)


def fact_content_hash(
    kind: Any,
    attributes: Any = None,
    value: Any = None,
    evidence_hash: Any = None,
) -> str:
    """Compute ``fact.content_hash`` over kind/attributes/value/evidence hash."""

    payload = {
        "kind": normalise_text(kind, casefold=True),
        "attributes": _jsonable(attributes if attributes is not None else {}),
        "value": _jsonable(value),
        "evidence_hash": normalise_text(evidence_hash),
    }
    return hash_parts(payload)


# --------------------------------------------------------------------------- #
# Bridge (Algorithm 3) keys
# --------------------------------------------------------------------------- #


def block_hash(block: Any) -> str:
    """Hash one bridge block (mapping, string or sequence) deterministically."""

    if isinstance(block, str) or block is None:
        return hash_parts(normalise_text(block))
    return hash_parts(block if isinstance(block, (Mapping, list, tuple)) else str(block))


def hash_blocks(*blocks: Any) -> str:
    """``hash_blocks(B1, B2, B3)`` — stable hash over the ordered blocks."""

    return hash_parts([block_hash(block) for block in blocks])


def bridge_dedupe_key(recommendation_id: Any, blocks: Any, *extra: Any) -> str:
    """``sha256(i*.id | hash_blocks)`` — Algorithm 3 injection dedupe key.

    ``blocks`` may be:
      * a mapping of block-name -> block payload (``{"B1": ..., "B2": ..., "B3": ...}``),
      * a sequence of three blocks, or
      * an already-computed hash string (returned unchanged).
    """

    if isinstance(blocks, str) and len(blocks) == 64:
        blocks_hash = blocks
    elif isinstance(blocks, Mapping):
        ordered = [blocks[key] for key in sorted(blocks, key=lambda k: str(k))]
        blocks_hash = hash_blocks(*ordered)
    elif isinstance(blocks, (list, tuple)):
        blocks_hash = hash_blocks(*blocks)
    else:
        blocks_hash = hash_blocks(blocks)
    parts: list[Any] = [normalise_text(recommendation_id), blocks_hash]
    parts.extend(normalise_text(extra_part) for extra_part in extra)
    return hash_parts(*parts)


def pattern_dedupe_key(pattern: Any) -> str:
    """Identity for an offline-graph node (``intent_pattern``).

    Uses name/target/vulnerability/tactic/MITRE id so re-ingesting the same
    corpus record is idempotent.  ``UNIQUE(src, dst, relation)`` on
    ``intent_pattern_edge`` handles edges separately.
    """

    if isinstance(pattern, Mapping):
        payload = pattern
    else:
        payload = {
            "name": _attr(pattern, "name", "title", "pattern"),
            "target_entity_type": _attr(pattern, "target_entity_type", "target_type", "entity_type"),
            "vulnerability_type": _attr(pattern, "vulnerability_type", "vulnerability", "vuln"),
            "tactic": _attr(pattern, "tactic", "technique", "action"),
            "mitre_attack_id": _attr(pattern, "mitre_attack_id", "mitre_id", "attack_id"),
        }
    return hash_parts(
        normalise_text(payload.get("name"), casefold=True),
        normalise_text(payload.get("target_entity_type"), casefold=True),
        normalise_text(payload.get("vulnerability_type"), casefold=True),
        normalise_text(payload.get("tactic"), casefold=True),
        normalise_text(payload.get("mitre_attack_id"), casefold=True),
    )


# --------------------------------------------------------------------------- #
# Bounded in-process seen-set
# --------------------------------------------------------------------------- #


@dataclass
class DedupeStats:
    """Counters exposed by :class:`SeenSet` for metrics/tests."""

    added: int = 0
    duplicates: int = 0
    evicted: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"added": self.added, "duplicates": self.duplicates, "evicted": self.evicted}


class SeenSet:
    """Bounded FIFO/LRU set of previously-seen keys.

    Used as a *fast path* only: the database UNIQUE constraint
    (``UNIQUE(project_id, dedupe_key)``) remains authoritative.  When the set is
    full the oldest key is evicted, so a long-running worker keeps memory flat.

    Exact constructor: ``SeenSet(maxsize: int = 4096, *, name: str = "dedupe")``.
    """

    def __init__(self, maxsize: int = 4096, *, name: str = "dedupe") -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")
        self.maxsize = int(maxsize)
        self.name = name
        self._seen: "OrderedDict[str, None]" = OrderedDict()
        self.stats = DedupeStats()

    # -- core ------------------------------------------------------------- #
    def add(self, key: Any) -> bool:
        """Record ``key``; return ``True`` if it is new, ``False`` if seen."""

        text = normalise_text(key)
        if not text:
            return False
        if text in self._seen:
            self._seen.move_to_end(text)
            self.stats.duplicates += 1
            return False
        self._seen[text] = None
        self.stats.added += 1
        while len(self._seen) > self.maxsize:
            self._seen.popitem(last=False)
            self.stats.evicted += 1
        return True

    def seen(self, key: Any) -> bool:
        """Return ``True`` when ``key`` was already recorded (does not add)."""

        return normalise_text(key) in self._seen

    def discard(self, key: Any) -> None:
        self._seen.pop(normalise_text(key), None)

    def clear(self) -> None:
        self._seen.clear()

    def __len__(self) -> int:
        return len(self._seen)

    def __contains__(self, key: Any) -> bool:  # pragma: no cover - trivial
        return self.seen(key)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size": len(self._seen), "maxsize": self.maxsize, **self.stats.to_dict()}


def suppress_duplicates(items: Iterable[Any], key: Any = None) -> list[Any]:
    """Return ``items`` with duplicates removed, preserving first occurrence.

    ``key`` is either a callable ``item -> hashable`` or a string naming an
    attribute/mapping key on each item.  Falling back to
    :func:`canonical_json` keeps behaviour deterministic for dicts.
    """

    seen: set[str] = set()
    out: list[Any] = []
    for item in items:
        if key is None:
            raw = canonical_json(item)
        elif callable(key):
            raw = key(item)
        else:
            raw = _attr(item, key)
        marker = normalise_text(raw) if not isinstance(raw, str) else raw
        if marker in seen:
            continue
        seen.add(marker)
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Database-backed duplicate lookups (tolerant, duck-typed)
# --------------------------------------------------------------------------- #


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Tolerant attribute/mapping accessor (skips ``None`` values)."""

    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            value = obj[name]
        else:
            value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def is_duplicate_intent(
    session_or_repo: Any,
    project_id: Any,
    dedupe_key: Any,
) -> bool:
    """Return ``True`` when an intent with ``dedupe_key`` already exists.

    Best-effort: any failure (missing table, driver error) returns ``False`` so
    the caller proceeds to insert, where the UNIQUE constraint is the real
    guarantee.
    """

    try:
        existing = await find_duplicate_intent(session_or_repo, project_id, dedupe_key)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("dedupe.lookup_failed", extra={"error": repr(exc)})
        return False
    return existing is not None


async def find_duplicate_intent(session_or_repo: Any, project_id: Any, dedupe_key: Any) -> Any:
    """Fetch the existing intent for ``(project_id, dedupe_key)`` or ``None``.

    Tries a repository method first (``find_by_dedupe_key`` / ``get_by_dedupe_key``
    / ``find_duplicate``), then falls back to a raw SQLAlchemy ``select``.
    """

    key_text = normalise_text(dedupe_key)
    if not key_text:
        return None

    for name in ("find_by_dedupe_key", "get_by_dedupe_key", "find_duplicate", "get_by_key"):
        func = getattr(session_or_repo, name, None)
        if not callable(func):
            continue
        try:
            result = await _maybe_await(func(project_id, key_text))
        except TypeError:
            try:
                result = await _maybe_await(func(dedupe_key=key_text))
            except Exception:
                continue
        except Exception:
            continue
        if result is not None:
            return result

    # Raw SQLAlchemy fallback.
    try:
        from sqlalchemy import select  # noqa: PLC0415 - lazy, keeps module import-light

        from app.db.models.intent import Intent  # noqa: PLC0415

        session = session_or_repo
        execute = getattr(session, "execute", None)
        if not callable(execute):
            return None
        stmt = (
            select(Intent)
            .where(Intent.project_id == project_id, Intent.dedupe_key == key_text)
            .limit(1)
        )
        result = await _maybe_await(execute(stmt))
        scalar = getattr(result, "scalar_one_or_none", None)
        if callable(scalar):
            return await _maybe_await(scalar())
        first = getattr(result, "first", None)
        if callable(first):
            row = await _maybe_await(first())
            return row[0] if row else None
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("dedupe.orm_lookup_failed", extra={"error": repr(exc)})
    return None
