"""Algorithm 2 - stage package (S1..S5) for intent retrieval & prediction.

Paper: "Staying on the Attack Path" (Algorithm 2, plan C9).

The five stages are *narrowing* filters.  Each stage consumes the candidate set
produced by the previous one and returns a :class:`StageResult` carrying a full
trace (candidates, scores, rejections) so the Retrieval Trace UI can explain the
funnel and so ``test_retrieval_pipeline`` can assert the S2->S3->S5 degradation
path:

    S1  ``s1_embedding``   semantic probing: top-k by Eq. 2 cosine similarity
                           ``sim_cos(v, phi(i)) = (v . phi(i)) / (||v|| ||phi(i)||)``
                           over the offline pattern graph ``G_o``.
    S2  ``s2_isomorphism`` bounded subgraph isomorphism: depth-2 predecessor
                           pattern checked only against the S1 Top-k candidates
                           (cost ``O(k)``, <=12 query nodes, <=60 pattern nodes,
                           5s timeout).
    S3  ``s3_jaccard``     degraded fuzzy inference: only when S2 is empty;
                           ``J(A,B) = |A n B| / |A u B|`` over attributed
                           neighbourhood token sets.
    S4  ``s4_alignment``   strategic-intent alignment: prune out-of-scope /
                           unauthorised lateral movement / goal-destructive
                           candidates, then
                           ``score = a*cos(obj, goal) + b*cos(obj, chain) + c*conf``
                           ``>= GOAL_ALIGN_THRESHOLD``.
    S5  ``s5_llm``         LLM heuristic reasoning on a reduced log; only after
                           S1-S4 yield nothing.  Its output is *still* routed
                           through ``retrieval.submit_for_validation()``.

Design rules honoured here:

* **Import light.**  No stage module is imported eagerly, so ``import
  app.core.intent.stages`` never pulls in networkx, numpy, httpx or an LLM
  client.  ``import_module`` is used by the lazy accessors.
* **Vocabulary only.**  Stage names / order / human readable descriptions come
  from :mod:`app.core.intent` (``STAGE_NAMES``, ``RETRIEVAL_STAGES``) so the API
  endpoint ``GET /projects/{id}/retrieval/stages`` stays consistent.
* **Uniform result contract.**  Every stage returns ``StageResult`` (never a
  bare list) and never raises for "no candidates"; a stage error is captured in
  ``StageResult.error`` so the orchestrator can decide between the documented
  degradations (S2 empty -> S3, S1-S4 empty -> S5).
* **No validation here.**  Stage outputs are *recommendations*.  Only
  :func:`app.core.intent.retrieval.submit_for_validation` may turn one into an
  intent (IPIA defence).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "STAGE_MODULES",
    "STAGE_FUNCTIONS",
    "RETRIEVAL_STAGES",
    "STAGE_NAMES",
    "TERMINAL_STAGE",
    "Rejection",
    "StageResult",
    "StageContext",
    "stage_names",
    "stage_module_name",
    "load_stage",
    "normalise_stage_result",
    "rejection",
    "empty_result",
    "describe_stages",
]

#: ``stage -> module`` mapping for the lazy loader.
STAGE_MODULES: Dict[str, str] = {
    "S1": "s1_embedding",
    "S2": "s2_isomorphism",
    "S3": "s3_jaccard",
    "S4": "s4_alignment",
    "S5": "s5_llm",
}

#: ``stage -> canonical entry-point name`` inside each stage module.
#:
#: Each stage module *must* expose one of these names.  ``run_stage`` is the
#: preferred uniform entry point; the historical per-stage names are accepted so
#: an individually implemented stage can still be used without a shim.
STAGE_FUNCTIONS: Dict[str, Tuple[str, ...]] = {
    "S1": ("run_stage", "run", "semantic_probing", "retrieve", "s1_semantic_probing"),
    "S2": ("run_stage", "run", "run_isomorphism", "bounded_isomorphism", "match"),
    "S3": ("run_stage", "run", "run_jaccard", "fuzzy_inference", "degraded_fuzzy_inference"),
    "S4": ("run_stage", "run", "run_alignment", "align", "strategic_alignment"),
    "S5": ("run_stage", "run", "run_llm", "heuristic_reasoning", "llm_heuristic_reasoning"),
}

#: Pipeline order (S1..S5) - mirrors :data:`app.core.intent.STAGE_NAMES`.
STAGE_NAMES: Tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")

#: Stage names with names/descriptions for the API + UI funnel.
RETRIEVAL_STAGES: Tuple[Dict[str, str], ...] = (
    {
        "stage": "S1",
        "name": "semantic_probing",
        "description": (
            "Embed the predecessor attribute vector with phi (Qwen3-Embedding-4B, "
            "2560-d) and keep the top-k offline patterns by Eq. 2 cosine "
            "similarity sim_cos(v, phi(i))."
        ),
    },
    {
        "stage": "S2",
        "name": "bounded_subgraph_isomorphism",
        "description": (
            "Depth-2 predecessor pattern matched only against the S1 Top-k "
            "candidates (O(k)) with a bounded DiGraphMatcher: <=12 query nodes, "
            "<=60 pattern nodes, 5s timeout."
        ),
    },
    {
        "stage": "S3",
        "name": "degraded_fuzzy_inference",
        "description": (
            "Run only when S2 is empty: J(A,B)=|A n B|/|A u B| over attributed "
            "neighbourhood token sets (labels + key=value attributes + edge "
            "relations)."
        ),
    },
    {
        "stage": "S4",
        "name": "strategic_intent_alignment",
        "description": (
            "Prune out-of-scope / unauthorised lateral movement and destructive "
            "conflicts with the goal; keep score = alpha*cos(obj,goal) + "
            "beta*cos(obj,primary_chain) + gamma*confidence >= "
            "GOAL_ALIGN_THRESHOLD."
        ),
    },
    {
        "stage": "S5",
        "name": "llm_heuristic_reasoning",
        "description": (
            "LLM heuristic reasoning over a reduced/belief-projected log, only "
            "after S1-S4 yield nothing; output still passes "
            "submit_for_validation()."
        ),
    },
)

#: The last stage of the pipeline (LLM heuristic reasoning).
TERMINAL_STAGE: str = "S5"


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------
@dataclass
class Rejection:
    """One candidate rejected by a stage, with the reason for the trace.

    ``candidate_id`` is the pattern/intent identifier; ``reason`` is a short
    machine-readable code (``out_of_scope``, ``lateral_movement``,
    ``destructive_conflict``, ``below_threshold``, ``isomorphism_mismatch``,
    ``timeout``, ...).  ``details`` carries stage-specific evidence.
    """

    candidate_id: str = ""
    reason: str = ""
    stage: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": str(self.candidate_id or ""),
            "reason": str(self.reason or ""),
            "stage": str(self.stage or ""),
            "detail": dict(self.detail or {}),
        }


@dataclass
class StageResult:
    """Uniform output of one Algorithm 2 stage.

    Attributes
    ----------
    stage:
        ``"S1"`` .. ``"S5"``.
    candidates:
        Surviving candidates in descending score order.  Each element is
        normally a mapping (pattern row projection) but may be any object the
        orchestrator can serialise.
    scores:
        ``candidate_id -> score`` for every candidate that was scored, which is
        a superset of ``candidates`` (rejected-but-scored entries stay here for
        the trace).
    rejections:
        Rejected candidates with reasons.
    examined:
        Number of candidates considered (cost accounting: S2 must only examine
        the Top-k, i.e. ``examined <= k``).
    trace:
        Free-form stage trace (query graph size, thresholds, timing, matching
        backend, ...).
    degraded / error:
        ``degraded`` marks a documented fallback path (S3 after empty S2, S5
        after empty S4); ``error`` is set instead of raising when a stage blows
        up.
    """

    stage: str = ""
    candidates: List[Any] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    rejections: List[Rejection] = field(default_factory=list)
    examined: int = 0
    trace: Dict[str, Any] = field(default_factory=dict)
    degraded: bool = False
    error: Optional[str] = None

    # -- helpers ---------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.candidates)

    def is_empty(self) -> bool:
        return not self.candidates

    def candidate_ids(self) -> List[str]:
        return [_candidate_id(item) for item in self.candidates]

    def top(self, n: int = 1) -> List[Any]:
        if n <= 0:
            return []
        return list(self.candidates[:n])

    def best(self) -> Optional[Any]:
        return self.candidates[0] if self.candidates else None

    def reject(self, candidate: Any, reason: str, **detail: Any) -> None:
        """Record a rejection using this result's stage name."""

        self.rejections.append(
            Rejection(
                candidate_id=_candidate_id(candidate),
                reason=str(reason),
                stage=self.stage,
                detail=dict(detail or {}),
            )
        )

    def note(self, **values: Any) -> None:
        """Merge extra keys into the trace (stage-local diagnostics)."""

        for key, value in values.items():
            self.trace[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "candidates": [_jsonable(item) for item in self.candidates],
            "scores": {str(k): float(v) for k, v in (self.scores or {}).items()},
            "rejections": [r.to_dict() for r in self.rejections],
            "examined": int(self.examined or 0),
            "trace": _jsonable(self.trace or {}),
            "degraded": bool(self.degraded),
            "error": self.error,
            "count": len(self.candidates),
        }


@dataclass
class StageContext:
    """Inputs shared by every stage.

    The orchestrator (:mod:`app.core.intent.retrieval`) builds one context and
    passes it down the funnel.  ``candidates`` is the previous stage's surviving
    set (empty for S1); ``top_k`` only binds S1.  All other fields are optional
    because stages must remain usable in unit tests with hand-built objects.
    """

    project_id: Optional[str] = None
    predecessor_fact_ids: List[str] = field(default_factory=list)
    state_vector: Optional[Sequence[float]] = None
    state_text: str = ""
    top_k: int = 10
    goal_text: str = ""
    phase: str = ""
    scope: Optional[Any] = None
    attack_graph: Optional[Any] = None
    merged_graph: Optional[Any] = None
    patterns: Sequence[Any] = field(default_factory=list)
    facts: Sequence[Any] = field(default_factory=list)
    pending_intents: Sequence[Any] = field(default_factory=list)
    embedding_provider: Optional[Any] = None
    llm_client: Optional[Any] = None
    settings: Optional[Any] = None
    candidates: List[Any] = field(default_factory=list)
    session: Optional[Any] = None
    trace: Dict[str, Any] = field(default_factory=dict)

    def previous_ids(self) -> List[str]:
        return [_candidate_id(item) for item in (self.candidates or [])]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def stage_names() -> Tuple[str, ...]:
    """Return the pipeline order (``("S1", ..., "S5")``)."""

    return STAGE_NAMES


def stage_module_name(stage: Any) -> str:
    """Map ``"s1"``/``1``/``"S1"``/``"stage1"`` to ``"s1_embedding"``.

    Returns ``""`` for an unknown stage so callers can degrade instead of
    raising (retrieval must never crash on a bad trace value).
    """

    key = _normalise_stage(stage)
    return STAGE_MODULES.get(key, "")


def load_stage(stage: Any) -> Callable[..., Any]:
    """Lazily import and return a stage's entry point.

    Raises
    ------
    KeyError
        Unknown stage name.
    ImportError
        Stage module missing.
    AttributeError
        Stage module present but exposes none of the accepted entry points.
    """

    key = _normalise_stage(stage)
    module_name = STAGE_MODULES.get(key)
    if not module_name:
        raise KeyError(f"unknown retrieval stage: {stage!r}")

    module = importlib.import_module(f"{__name__}.{module_name}")
    for attr in STAGE_FUNCTIONS.get(key, ("run_stage", "run")):
        func = getattr(module, attr, None)
        if callable(func):
            return func
    raise AttributeError(
        f"stage module {module.__name__!r} exposes none of "
        f"{STAGE_FUNCTIONS.get(key, ())!r}"
    )


def normalise_stage_result(
    value: Any,
    *,
    stage: str,
    examined: Optional[int] = None,
) -> StageResult:
    """Coerce a stage's return value into :class:`StageResult`.

    Accepted shapes: ``StageResult``, ``None``/``False`` (empty), a sequence of
    candidates, or a mapping with ``candidates``/``scores``/``rejections``/
    ``trace``/``degraded``/``error`` keys (the shape written by hand-built
    stages and by ``s5_llm``).  This keeps the orchestrator tolerant while still
    guaranteeing one uniform contract.
    """

    if isinstance(value, StageResult):
        if not value.stage:
            value.stage = stage
        if examined is not None and not value.examined:
            value.examined = int(examined)
        return value

    result = StageResult(stage=stage, examined=int(examined or 0))
    if value is None or value is False:
        return result

    if isinstance(value, dict):
        candidates = value.get("candidates")
        if candidates is None:
            candidates = value.get("items") or value.get("results") or []
        result.candidates = list(candidates or [])
        result.scores = {
            str(k): float(v)
            for k, v in dict(value.get("scores") or {}).items()
            if _is_number(v)
        }
        result.rejections = [
            _as_rejection(item, stage) for item in (value.get("rejections") or [])
        ]
        result.trace = dict(value.get("trace") or {})
        result.degraded = bool(value.get("degraded"))
        result.error = value.get("error")
        if examined is not None and not result.examined:
            result.examined = int(examined)
        return result

    if isinstance(value, (list, tuple, set, frozenset)):
        result.candidates = list(value)
        return result

    # Anything else is treated as a single candidate.
    result.candidates = [value]
    return result


def rejection(
    candidate: Any,
    reason: str,
    *,
    stage: str = "",
    **detail: Any,
) -> Rejection:
    """Build a :class:`Rejection` for a candidate."""

    return Rejection(
        candidate_id=_candidate_id(candidate),
        reason=str(reason),
        stage=str(stage or ""),
        detail=dict(detail or {}),
    )


def empty_result(stage: str, reason: str = "", **trace: Any) -> StageResult:
    """An empty :class:`StageResult` (used for documented degradations)."""

    result = StageResult(stage=_normalise_stage(stage) or str(stage), trace=dict(trace or {}))
    if reason:
        result.trace.setdefault("reason", reason)
    return result


def describe_stages() -> List[Dict[str, Any]]:
    """Stage descriptors plus the module/entry-point wiring (diagnostics)."""

    out: List[Dict[str, Any]] = []
    for entry in RETRIEVAL_STAGES:
        item: Dict[str, Any] = dict(entry)
        item["module"] = f"app.core.intent.stages.{STAGE_MODULES[entry['stage']]}"
        item["entry_points"] = list(STAGE_FUNCTIONS.get(entry["stage"], ()))
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _normalise_stage(stage: Any) -> str:
    """Normalise ``"s1"``/``1``/``"S1"``/``"stage1"`` to ``"S1"``; else ``""``."""

    if stage is None:
        return ""
    if isinstance(stage, bool):  # bool is an int subclass - never a stage
        return ""
    value = getattr(stage, "value", stage)
    if isinstance(value, int):
        return f"S{value}" if 1 <= value <= 5 else ""
    text = str(value).strip().upper()
    if text.startswith("STAGE"):
        text = text[5:].strip()
    elif text.startswith("S") and text[1:].isdigit():
        text = text[1:]
    if text.isdigit() and 1 <= int(text) <= 5:
        return f"S{int(text)}"
    return ""


def _candidate_id(candidate: Any) -> str:
    """Best-effort identifier for a candidate (mapping or object)."""

    if candidate is None:
        return ""
    if isinstance(candidate, (str, int)):
        return str(candidate)
    if isinstance(candidate, dict):
        for key in ("id", "pattern_id", "intent_id", "recommendation_id", "name", "candidate_id"):
            value = candidate.get(key)
            if value:
                return str(value)
        return ""
    for attr in ("id", "pattern_id", "intent_id", "recommendation_id", "name", "candidate_id"):
        value = getattr(candidate, attr, None)
        if value:
            return str(value)
    return ""


def _as_rejection(item: Any, stage: str) -> Rejection:
    if isinstance(item, Rejection):
        if not item.stage:
            item.stage = stage
        return item
    if isinstance(item, dict):
        return Rejection(
            candidate_id=str(item.get("candidate_id") or item.get("id") or ""),
            reason=str(item.get("reason") or ""),
            stage=str(item.get("stage") or stage),
            detail=dict(item.get("detail") or {}),
        )
    return Rejection(candidate_id=_candidate_id(item), reason=str(item), stage=stage)


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _jsonable(value: Any) -> Any:
    """Recursively coerce a trace value to JSON-safe primitives."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


def _iter_ids(values: Optional[Iterable[Any]]) -> List[str]:
    """Normalise an id iterable (helper shared by stage implementations)."""

    if not values:
        return []
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out
