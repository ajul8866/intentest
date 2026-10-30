"""Algorithm 2, Stage S1 - semantic probing (paper §4.4, Eq. 2, plan C9).

S1 is the *entry* of the five-stage intent retrieval pipeline.  Given the
predecessor facts of the intent that is about to be created, it

1. builds the predecessor **attribute vector text** (a deterministic, sorted
   ``key=value`` serialisation of every verified predecessor fact), embeds it
   with :math:`\\varphi` (default ``Qwen3-Embedding-4B``, 2560-d), and
2. scores every node of the offline MITRE-mapped graph :math:`G_o` by
   Equation 2::

       sim_cos(v, phi(i)) = (v . phi(i)) / (||v|| ||phi(i)||)

3. keeps the top-k candidates (:data:`SUBGRAPH_TOP_K <app.core.intent.DEFAULT_PARAMETERS>`
   default ``10``) and returns them, with the full per-candidate score trace,
   to the orchestrator.

Nothing in this module mutates the DAG and nothing here creates an intent: a
candidate produced by S1 is only ever a *suggestion* and must pass through
:func:`app.core.intent.retrieval.submit_for_validation` before it can exist as
an ``intent`` row (invariant I3, the IPIA defence).

Design notes
------------
* Embedding backend resolution is delegated to
  :mod:`app.core.intent.embedding`, whose fallback chain is
  ``api -> sentence-transformers -> hash``.  The backend that actually produced
  the vectors is recorded on the result (and therefore in the persisted
  retrieval trace) so a degraded run is visible rather than silent.
* Pattern vectors are cached twice: in-process by the provider's LRU
  (:class:`~app.core.intent.embedding.EmbeddingCache`) and, when the pattern row
  already carries a persisted ``embedding`` column (pgvector), that vector is
  reused instead of calling the endpoint again.
* The query vector is the L2-normalised mean of the per-fact vectors, which
  keeps the similarity bounded in ``[-1, 1]`` and makes a single very long
  predecessor dominate less than it would in a naive concatenation.
* S1 never raises for missing data: no facts -> empty result with reason
  ``"no_predecessors"``; no patterns -> ``"no_patterns"``; embedding failure ->
  ``"embedding_failed"`` with the exception detail in the trace.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from app.core.intent.embedding import (
    EmbeddingProvider,
    attribute_vector_text,
    cosine_similarity,
    get_embedding_provider,
    normalise_vector,
    objective_text,
)

logger = logging.getLogger(__name__)

__all__ = [
    "STAGE",
    "DEFAULT_TOP_K",
    "S1Candidate",
    "S1Result",
    "build_query_text",
    "build_query_vector",
    "pattern_text",
    "pattern_vector",
    "score_patterns",
    "run_s1",
    "s1_semantic_probing",
]

STAGE = "S1"
STAGE_NAME = "semantic_probing"

#: Default ``SUBGRAPH_TOP_K`` (documented default, see LIMITATIONS.md §2).
DEFAULT_TOP_K = 10
#: Scores at or below this are considered noise and are still returned in the
#: trace but flagged as ``excluded`` (they are *not* candidates).
DEFAULT_MIN_SCORE = 0.0


# ---------------------------------------------------------------------------
# tolerant accessors (ORM row / mapping / dataclass all accepted)
# ---------------------------------------------------------------------------
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first present, non-``None`` attribute/key among ``names``."""

    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
            continue
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _pattern_id(pattern: Any) -> str:
    raw = _attr(pattern, "id", "pattern_id", "uuid", default=None)
    if raw is None:
        raw = _attr(pattern, "name", default="")
    return str(raw) if raw is not None else ""


def _coerce_vector(value: Any) -> Optional[List[float]]:
    """Convert a persisted embedding (list/np.ndarray/str) to ``list[float]``."""

    if value is None:
        return None
    if isinstance(value, str):
        # pgvector returns e.g. "[0.1,0.2,...]"
        text = value.strip().strip("[]")
        if not text:
            return None
        try:
            return [float(part) for part in text.split(",") if part.strip()]
        except (TypeError, ValueError):
            return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def _mean_vector(vectors: Sequence[Sequence[float]], *, dimension: Optional[int] = None) -> List[float]:
    """L2-normalised mean of ``vectors`` (zero vector when empty)."""

    vectors = [v for v in vectors if v]
    if not vectors:
        return []
    dim = dimension or max(len(v) for v in vectors)
    acc = [0.0] * dim
    for vec in vectors:
        for index, value in enumerate(vec[:dim]):
            acc[index] += float(value)
    acc = [value / len(vectors) for value in acc]
    return normalise_vector(acc, dimension=dim)


# ---------------------------------------------------------------------------
# query construction
# ---------------------------------------------------------------------------
def build_query_text(facts: Iterable[Any]) -> str:
    """Deterministically serialise the predecessor facts into one query string.

    Each fact contributes ``<KIND> key=value key2=value2 value=<value>`` in a
    stable order; facts are sorted by their node id so two callers with the
    same predecessor set embed byte-identical text (and therefore hit the same
    embedding cache entry).
    """

    rows: List[Tuple[str, str]] = []
    for fact in facts or ():
        if fact is None:
            continue
        fact_id = str(_attr(fact, "id", "fact_id", default="") or "")
        kind = str(_attr(fact, "kind", default="") or "").upper()
        value = _attr(fact, "value", default=None)
        attributes = _attr(fact, "attributes", "attrs", default=None)
        text = attribute_vector_text(attributes, kind=kind, value=value)
        rows.append((fact_id, f"{kind} {text}".strip()))
    rows.sort(key=lambda item: (item[0], item[1]))
    return "\n".join(text for _, text in rows if text).strip()


def build_query_vector(
    facts: Iterable[Any],
    *,
    provider: Optional[EmbeddingProvider] = None,
    dimension: Optional[int] = None,
) -> List[float]:
    """Return :math:`v = \\varphi(\\text{predecessor attributes})`.

    ``provider`` defaults to the process-wide provider.  The vector is the
    normalised mean of the per-fact embeddings so that a multi-predecessor
    intent (a hyperedge with 3 facts) yields one comparable vector.
    """

    fact_list = [f for f in (facts or ()) if f is not None]
    if not fact_list:
        return []
    provider = provider or get_embedding_provider()
    vectors: List[List[float]] = []
    for fact in fact_list:
        kind = str(_attr(fact, "kind", default="") or "").upper()
        value = _attr(fact, "value", default=None)
        attributes = _attr(fact, "attributes", "attrs", default=None)
        text = attribute_vector_text(attributes, kind=kind, value=value)
        if not text:
            continue
        try:
            vectors.append(provider.embed_vector(text))
        except Exception as exc:  # pragma: no cover - hash backend never fails
            logger.warning(
                "s1.fact_embedding_failed",
                extra={"fact_id": str(_attr(fact, "id", "fact_id", default="")), "error": str(exc)},
            )
    if not vectors:
        # Fall back to embedding the concatenated query text so S1 still works.
        text = build_query_text(fact_list)
        if not text:
            return []
        try:
            vectors = [provider.embed_vector(text)]
        except Exception:  # pragma: no cover - defensive
            return []
    return _mean_vector(vectors, dimension=dimension or getattr(provider, "dimension", None))


# ---------------------------------------------------------------------------
# pattern side
# ---------------------------------------------------------------------------
def pattern_text(pattern: Any) -> str:
    """Canonical embeddable text for one offline-graph pattern node."""

    return objective_text(
        _attr(pattern, "objective", "name", "description", "summary", default="") or "",
        tactic=_attr(pattern, "tactic", default=None),
        target=_attr(pattern, "target_entity_type", "target_entity", "target", default=None),
        action_type=_attr(pattern, "action_type", default=None),
    )


def pattern_vector(
    pattern: Any,
    *,
    provider: Optional[EmbeddingProvider] = None,
    use_stored: bool = True,
) -> Tuple[List[float], str]:
    """Return ``(vector, source)`` for a pattern.

    ``source`` is ``"stored"`` when the persisted pgvector column was used,
    ``"cache"``/``"computed"`` otherwise (as reported by the provider), or
    ``"empty"`` when no text and no vector are available.
    """

    if use_stored:
        stored = _coerce_vector(_attr(pattern, "embedding", "vector", default=None))
        if stored:
            return stored, "stored"
    text = pattern_text(pattern)
    if not text:
        return [], "empty"
    provider = provider or get_embedding_provider()
    try:
        result = provider.embed(text)
    except Exception as exc:  # pragma: no cover - hash backend never fails
        logger.warning(
            "s1.pattern_embedding_failed",
            extra={"pattern_id": _pattern_id(pattern), "error": str(exc)},
        )
        return [], "failed"
    vector = list(getattr(result, "vector", result) or [])
    return vector, "cache" if getattr(result, "cached", False) else "computed"


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------
@dataclass
class S1Candidate:
    """One scored offline-graph node produced by S1."""

    pattern_id: str
    pattern: Any = None
    score: float = 0.0
    confidence: float = 0.0
    name: str = ""
    tactic: str = ""
    objective: str = ""
    target_entity_type: str = ""
    vulnerability_type: str = ""
    mitre_attack_id: str = ""
    vector_source: str = ""
    excluded: bool = False
    reason: str = ""

    def to_dict(self, *, include_pattern: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "pattern_id": self.pattern_id,
            "score": round(float(self.score), 6),
            "confidence": float(self.confidence or 0.0),
            "name": self.name,
            "tactic": self.tactic,
            "objective": self.objective,
            "target_entity_type": self.target_entity_type,
            "vulnerability_type": self.vulnerability_type,
            "mitre_attack_id": self.mitre_attack_id,
            "vector_source": self.vector_source,
            "excluded": bool(self.excluded),
        }
        if self.reason:
            payload["reason"] = self.reason
        if include_pattern and self.pattern is not None:
            payload["pattern"] = self.pattern
        return payload


@dataclass
class S1Result:
    """Outcome of the S1 stage (also the persisted trace fragment)."""

    stage: str = STAGE
    stage_name: str = STAGE_NAME
    query_text: str = ""
    candidates: List[S1Candidate] = field(default_factory=list)
    excluded: List[S1Candidate] = field(default_factory=list)
    top_k: int = DEFAULT_TOP_K
    backend: str = ""
    model: str = ""
    dimension: int = 0
    duration_ms: float = 0.0
    reason: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.candidates

    def candidate_dicts(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.candidates]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_name": self.stage_name,
            "query_text": self.query_text,
            "candidates": self.candidate_dicts(),
            "excluded": [c.to_dict() for c in self.excluded],
            "top_k": self.top_k,
            "backend": self.backend,
            "model": self.model,
            "dimension": self.dimension,
            "duration_ms": round(self.duration_ms, 3),
            "reason": self.reason,
            "stats": dict(self.stats),
        }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score_patterns(
    query_vector: Sequence[float],
    patterns: Iterable[Any],
    *,
    provider: Optional[EmbeddingProvider] = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    use_stored_embeddings: bool = True,
) -> Tuple[List[S1Candidate], List[S1Candidate]]:
    """Score every pattern against ``query_vector`` (Eq. 2).

    Returns ``(top_k_candidates, excluded_candidates)``.  Both lists are sorted
    by score desc, then pattern id asc, so traces are reproducible.  The cost is
    ``O(|G_o|)``; S2's bounded isomorphism then runs over ``top_k`` only.
    """

    scored: List[S1Candidate] = []
    for pattern in patterns or ():
        pid = _pattern_id(pattern)
        if not pid:
            continue
        vector, source = pattern_vector(pattern, provider=provider, use_stored=use_stored_embeddings)
        score = cosine_similarity(list(query_vector or []), vector) if vector else 0.0
        candidate = S1Candidate(
            pattern_id=pid,
            pattern=pattern,
            score=float(score),
            confidence=float(_attr(pattern, "confidence", default=0.0) or 0.0),
            name=str(_attr(pattern, "name", default="") or ""),
            tactic=str(_attr(pattern, "tactic", default="") or ""),
            objective=str(_attr(pattern, "objective", "description", default="") or ""),
            target_entity_type=str(_attr(pattern, "target_entity_type", default="") or ""),
            vulnerability_type=str(_attr(pattern, "vulnerability_type", default="") or ""),
            mitre_attack_id=str(_attr(pattern, "mitre_attack_id", default="") or ""),
            vector_source=source,
        )
        if not vector:
            candidate.excluded = True
            candidate.reason = "no_vector"
        elif score <= min_score:
            candidate.excluded = True
            candidate.reason = "below_min_score"
        scored.append(candidate)

    scored.sort(key=lambda c: (-c.score, c.pattern_id))
    limit = max(0, int(top_k or 0))
    accepted = [c for c in scored if not c.excluded][:limit]
    rejected = [c for c in scored if c.excluded]
    return accepted, rejected


# ---------------------------------------------------------------------------
# stage entry point
# ---------------------------------------------------------------------------
def run_s1(
    facts: Iterable[Any] = (),
    patterns: Iterable[Any] = (),
    *,
    provider: Optional[EmbeddingProvider] = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    dimension: Optional[int] = None,
    use_stored_embeddings: bool = True,
) -> S1Result:
    """Execute Stage S1 (synchronous; the orchestrator wraps it in a thread).

    This is a pure function of ``(facts, patterns, provider)`` — no I/O other
    than the (cached) embedding calls and no database writes.
    """

    started = time.perf_counter()
    fact_list = [f for f in (facts or ()) if f is not None]
    pattern_list = [p for p in (patterns or ()) if p is not None]
    result = S1Result(top_k=max(0, int(top_k or 0)))
    if not fact_list:
        result.reason = "no_predecessors"
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        return result
    if not pattern_list:
        result.reason = "no_patterns"
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        return result

    try:
        provider = provider or get_embedding_provider()
        result.backend = str(getattr(provider, "active_backend", "") or getattr(provider, "backend", "") or "")
        result.model = str(getattr(provider, "model", "") or "")
        result.dimension = int(getattr(provider, "dimension", 0) or 0)
        result.query_text = build_query_text(fact_list)
        query_vector = build_query_vector(fact_list, provider=provider, dimension=dimension)
    except Exception as exc:
        logger.warning("s1.query_embedding_failed", extra={"error": str(exc)})
        result.reason = "embedding_failed"
        result.stats["error"] = f"{type(exc).__name__}: {exc}"
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        return result

    if not query_vector:
        result.reason = "empty_query_vector"
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        return result

    candidates, excluded = score_patterns(
        query_vector,
        pattern_list,
        provider=provider,
        top_k=result.top_k,
        min_score=min_score,
        use_stored_embeddings=use_stored_embeddings,
    )
    result.candidates = candidates
    result.excluded = excluded
    result.stats = {
        "patterns": len(pattern_list),
        "facts": len(fact_list),
        "accepted": len(candidates),
        "excluded": len(excluded),
        "stored_vectors": sum(1 for c in candidates if c.vector_source == "stored"),
        "top_score": round(candidates[0].score, 6) if candidates else 0.0,
        "query_dim": len(query_vector),
    }
    if not candidates:
        result.reason = "no_candidates"
    result.duration_ms = (time.perf_counter() - started) * 1000.0
    logger.debug(
        "s1.completed",
        extra={
            "backends": result.backend,
            "candidates": len(candidates),
            "top_k": result.top_k,
            "duration_ms": round(result.duration_ms, 2),
            "reason": result.reason,
        },
    )
    return result


async def s1_semantic_probing(
    facts: Iterable[Any] = (),
    patterns: Iterable[Any] = (),
    **kwargs: Any,
) -> S1Result:
    """Async alias used by the retrieval orchestrator."""

    return run_s1(facts, patterns, **kwargs)
