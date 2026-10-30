"""Intent retrieval & prediction API (paper Algorithm 2).

These routes expose the five-stage retrieval pipeline and its trace:

``POST /projects/{project_id}/retrieval``
    Run Algorithm 2 for a set of predecessor facts (or for the current
    frontier of the runtime attack graph) and return the recommendation
    *only* after it passed ``submit_for_validation`` (precondition + scope +
    acyclicity).  The full per-stage trace (candidate sets, scores, rejection
    reasons) is persisted so the Retrieval Trace UI can render the funnel.

``GET  /projects/{project_id}/retrieval/{trace_id}``
    Fetch one persisted trace.

``GET  /projects/{project_id}/retrievals``
    List persisted traces (newest first).

``GET  /projects/{project_id}/retrieval/stages``
    Static description of S1..S5 plus the configured thresholds, so the UI can
    label the funnel without hard-coding constants.

``GET  /projects/{project_id}/patterns`-adjacent endpoints live in
``routes/patterns.py``; this module only consumes the offline graph.

Everything here is read-mostly: the only writes are the ``retrieval_trace``
row and (optionally) the created intent when ``enqueue=true``.  Neither
mutates the immutable fact set, and a rejected recommendation never creates an
intent (IPIA defence).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import uuid
from typing import Any, Mapping, Optional, Sequence

from fastapi import APIRouter, Body, Depends, Query, status

from app.api import deps
from app.api.errors import (
    AppError,
    ErrorCode,
    NotFoundError,
    ValidationAppError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects/{project_id}/retrieval", tags=["retrieval"])

_MISS = object()
_MAX_LIMIT = 500

#: Stage identifiers used by Algorithm 2 and by the Retrieval Trace UI.
STAGES: tuple[dict[str, str], ...] = (
    {
        "stage": "S1",
        "name": "semantic probing",
        "description": (
            "Embed the predecessor attribute vector with phi and rank the offline "
            "graph G_o by cosine similarity (Eq. 2); keep the top-k candidates."
        ),
    },
    {
        "stage": "S2",
        "name": "bounded subgraph isomorphism",
        "description": (
            "Depth-2 predecessor pattern matched only against the S1 top-k "
            "candidates (cost O(k)); <=12 query nodes, <=60 pattern nodes, 5s."
        ),
    },
    {
        "stage": "S3",
        "name": "degraded fuzzy inference",
        "description": (
            "Jaccard similarity over attributed neighbourhood token sets; only "
            "reached when S2 produced nothing."
        ),
    },
    {
        "stage": "S4",
        "name": "strategic intent alignment",
        "description": (
            "Prune out-of-scope / unauthorised lateral movement and "
            "goal-destructive candidates; score = alpha*cos(objective, goal) + "
            "beta*cos(objective, primary chain) + gamma*confidence."
        ),
    },
    {
        "stage": "S5",
        "name": "LLM heuristic reasoning",
        "description": (
            "Reduced-log LLM reasoning, reached only when S1-S4 yield nothing; "
            "its output is validated exactly like any other candidate."
        ),
    },
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Tolerant attribute/mapping lookup (ORM rows, dataclasses, dicts)."""
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        try:
            value = getattr(obj, name)
        except Exception:  # pragma: no cover - defensive
            continue
        if value is not None:
            return value
    return default


def _iso(value: Any) -> Any:
    """Stringify UUIDs/datetimes/enums for JSON transport."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


def _jsonable(value: Any) -> Any:
    """Recursively convert a value into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (uuid.UUID,)):
        return str(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump())
        except Exception:  # pragma: no cover - defensive
            pass
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        try:
            return _jsonable(as_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return _iso(value)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_first(obj: Any, names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
    """Call the first existing method/function among ``names``.

    ``TypeError`` (signature mismatch) skips to the next candidate; any other
    exception propagates so real failures are not masked.
    """
    if obj is None:
        return _MISS
    for name in names:
        candidate = getattr(obj, name, None)
        if candidate is None or not callable(candidate):
            continue
        try:
            return await _maybe_await(candidate(*args, **kwargs))
        except TypeError:
            continue
    return _MISS


def _coerce_uuid(value: Any, *, field: str = "project_id") -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValidationAppError(
            f"Invalid {field}",
            details={"field": field, "value": str(value)},
        ) from exc


async def _load_project(session: Any, project_id: uuid.UUID) -> Any:
    from sqlalchemy import select

    from app.db.models.project import Project

    result = await session.execute(select(Project).where(Project.id == project_id))
    project = result.scalars().first()
    if project is None:
        raise NotFoundError(
            f"Project {str(project_id)!r} does not exist",
            details={"project_id": str(project_id)},
        )
    return project


def _retrieval_module() -> Any:
    """Import ``app.core.intent.retrieval`` lazily."""
    try:
        return importlib.import_module("app.core.intent.retrieval")
    except Exception as exc:  # pragma: no cover - import guard
        logger.debug("retrieval.module_unavailable", extra={"error": str(exc)})
        return None


def _embedding_module() -> Any:
    try:
        return importlib.import_module("app.core.intent.embedding")
    except Exception:  # pragma: no cover - import guard
        return None


def _bridge_module() -> Any:
    try:
        return importlib.import_module("app.core.intent.bridge")
    except Exception:  # pragma: no cover - import guard
        return None


def _trace_to_dict(trace: Any) -> dict[str, Any]:
    """Normalise a persisted trace row / dict into the API shape."""
    if trace is None:
        return {}
    payload = _attr(trace, "trace", "payload", "data", default=None)
    return {
        "id": _iso(_attr(trace, "id", default=None)),
        "project_id": _iso(_attr(trace, "project_id", default=None)),
        "intent_id": _iso(_attr(trace, "intent_id", "source_intent_id", default=None)),
        "stage": _attr(trace, "stage", "reached_stage", default=None),
        "query": _jsonable(_attr(trace, "query", "query_vector", default=None)),
        "candidate_count": _attr(trace, "candidate_count", default=None),
        "recommendation": _jsonable(
            _attr(trace, "recommendation", "recommended", default=None)
        ),
        "accepted": _attr(trace, "accepted", "validated", default=None),
        "rejections": _jsonable(_attr(trace, "rejections", "rejection_reasons", default=[])),
        "embedding_backend": _attr(trace, "embedding_backend", "backend", default=None),
        "duration_ms": _attr(trace, "duration_ms", default=None),
        "trace": _jsonable(payload if payload is not None else {}),
        "created_at": _iso(_attr(trace, "created_at", default=None)),
    }


def _recommendation_to_dict(candidate: Any) -> dict[str, Any]:
    """Serialise a retrieval candidate / recommendation."""
    if candidate is None:
        return {}
    if isinstance(candidate, Mapping):
        data = dict(candidate)
    else:
        data = {
            key: _attr(candidate, key, default=None)
            for key in (
                "id",
                "pattern_id",
                "name",
                "action_type",
                "tactic",
                "objective",
                "target_entity",
                "vulnerability_type",
                "mitre_attack_id",
                "confidence",
                "state",
                "origin",
                "score",
                "alignment_score",
                "action_spec",
                "preconditions",
                "matching_method",
                "stage",
            )
        }
    return {k: _jsonable(v) for k, v in data.items()}


async def _run_pipeline(
    *,
    session: Any,
    project: Any,
    predecessor_fact_ids: Sequence[str],
    top_k: int,
    enqueue: bool,
) -> dict[str, Any]:
    """Invoke Algorithm 2 and normalise its return value."""
    module = _retrieval_module()
    project_id = _attr(project, "id", default=None)

    if module is not None:
        runner = getattr(module, "run_retrieval", None) or getattr(
            module, "retrieve", None
        )
        if callable(runner):
            for kwargs in (
                {
                    "session": session,
                    "project": project,
                    "project_id": project_id,
                    "predecessor_fact_ids": list(predecessor_fact_ids),
                    "top_k": top_k,
                    "enqueue": enqueue,
                },
                {
                    "session": session,
                    "project_id": project_id,
                    "predecessor_fact_ids": list(predecessor_fact_ids),
                    "top_k": top_k,
                },
            ):
                try:
                    result = runner(**kwargs)
                except TypeError:
                    continue
                result = await _maybe_await(result)
                if isinstance(result, Mapping):
                    return dict(result)
                return {
                    "recommendation": _recommendation_to_dict(result),
                    "trace": {},
                    "stage": getattr(result, "stage", None),
                }

    # Fallback: no orchestrator importable.  Be explicit instead of inventing
    # a recommendation (never fabricate a graph edge).
    return {
        "stage": None,
        "recommendation": None,
        "accepted": False,
        "candidates": [],
        "rejections": [
            {
                "reason": "retrieval_unavailable",
                "detail": "app.core.intent.retrieval is not importable",
            }
        ],
        "trace": {"stages": [], "available": False},
    }


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@router.get("/stages")
async def retrieval_stages() -> dict[str, Any]:
    """Describe the five stages and the configured thresholds."""
    settings = deps.get_settings()
    from app.core.intent import retrieval as _retrieval  # noqa: F401  (optional)

    def _cfg(*names: str, default: Any = None) -> Any:
        for name in names:
            value = getattr(settings, name, None)
            if value is None:
                value = getattr(settings, name.lower(), None)
            if value is not None:
                return value
        return default

    return {
        "stages": [
            {
                **stage,
                "backends": (
                    ["S1_embedding", "S2_isomorphism", "S3_jaccard", "S4_alignment", "S5_llm"]
                    if stage["stage"] == "S1"
                    else None
                ),
            }
            if stage["stage"] == "S1"
            else dict(stage)
            for stage in STAGES
        ],
        "thresholds": {
            "subgraph_top_k": _cfg("subgraph_top_k", default=10),
            "jaccard_threshold": _cfg("jaccard_threshold", default=0.5),
            "goal_align_threshold": _cfg("goal_align_threshold", default=0.55),
            "align_alpha": _cfg("align_alpha", default=0.5),
            "align_beta": _cfg("align_beta", default=0.3),
            "align_gamma": _cfg("align_gamma", default=0.2),
            "crosslink_cosine": _cfg("crosslink_cosine", default=0.75),
            "confidence_threshold": _cfg("confidence_threshold", default=0.6),
            "subgraph_max_query_nodes": _cfg("subgraph_max_query_nodes", default=12),
            "subgraph_max_pattern_nodes": _cfg("subgraph_max_pattern_nodes", default=60),
            "subgraph_timeout_s": _cfg("subgraph_timeout_s", default=5),
            "bridge_theta": _cfg("bridge_theta", default=0.3),
        },
        "notes": [
            "S3 is reached only when S2 returns no candidate.",
            "S5 is reached only when S1-S4 all return nothing.",
            "Every candidate, including S5 output, must pass submit_for_validation().",
        ],
    }


@router.post("", status_code=status.HTTP_200_OK)
async def run_retrieval(
    project_id: str,
    payload: Mapping[str, Any] = Body(
        default_factory=dict,
        description="Retrieval request (predecessor fact ids, top_k, enqueue).",
    ),
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Run Algorithm 2 and return the validated recommendation plus the trace."""
    pid = _coerce_uuid(project_id)
    project = await _load_project(session, pid)

    raw_ids = payload.get("predecessor_fact_ids") or payload.get("fact_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    if not isinstance(raw_ids, (list, tuple)):
        raise ValidationAppError(
            "predecessor_fact_ids must be a list of fact ids",
            details={"field": "predecessor_fact_ids"},
        )
    fact_ids = [str(fid) for fid in raw_ids]

    top_k = payload.get("top_k")
    try:
        top_k = int(top_k) if top_k is not None else 10
    except (TypeError, ValueError):
        raise ValidationAppError(
            "top_k must be an integer",
            details={"field": "top_k", "value": str(payload.get("top_k"))},
        )
    if top_k < 1 or top_k > 100:
        raise ValidationAppError(
            "top_k must be between 1 and 100",
            details={"field": "top_k", "value": top_k},
        )

    enqueue = bool(payload.get("enqueue", False))

    result = await _run_pipeline(
        session=session,
        project=project,
        predecessor_fact_ids=fact_ids,
        top_k=top_k,
        enqueue=enqueue,
    )

    recommendation = result.get("recommendation")
    if not isinstance(recommendation, Mapping):
        recommendation = _recommendation_to_dict(recommendation)

    trace = result.get("trace")
    if not isinstance(trace, Mapping):
        trace = {"raw": _jsonable(trace)}

    response = {
        "project_id": str(pid),
        "stage": result.get("stage"),
        "accepted": bool(result.get("accepted", bool(recommendation))),
        "recommendation": {k: _jsonable(v) for k, v in dict(recommendation).items()},
        "candidates": _jsonable(result.get("candidates") or []),
        "rejections": _jsonable(result.get("rejections") or []),
        "trace": _jsonable(trace),
        "embedding_backend": result.get("embedding_backend")
        or _attr(trace, "embedding_backend", default=None),
        "trace_id": _iso(result.get("trace_id")),
        "intent_id": _iso(result.get("intent_id")),
        "predecessor_fact_ids": fact_ids,
    }
    logger.info(
        "retrieval.executed",
        extra={
            "project_id": str(pid),
            "stage": response["stage"],
            "accepted": response["accepted"],
            "predecessor_count": len(fact_ids),
            "top_k": top_k,
        },
    )
    return response


@router.get("")
async def list_retrievals(
    project_id: str,
    limit: int = Query(100, ge=1, le=_MAX_LIMIT),
    offset: int = Query(0, ge=0),
    stage: Optional[str] = Query(default=None, description="Filter by reached stage."),
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """List persisted retrieval traces for a project (newest first)."""
    pid = _coerce_uuid(project_id)
    await _load_project(session, pid)

    from sqlalchemy import func, select

    trace_model = None
    for module_name, class_name in (
        ("app.db.models.retrieval_trace", "RetrievalTrace"),
        ("app.db.models.intent", "RetrievalTrace"),
        ("app.db.models.event", "RetrievalTrace"),
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        candidate = getattr(module, class_name, None)
        if candidate is not None:
            trace_model = candidate
            break

    if trace_model is None:
        # The retrieval trace store is not part of the initial migration; the
        # API reports an explicit empty list instead of 500-ing.
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    filters = [trace_model.project_id == pid]
    stage_column = getattr(trace_model, "stage", None)
    if stage and stage_column is not None:
        filters.append(stage_column == stage.upper())

    count_stmt = select(func.count()).select_from(trace_model).where(*filters)
    total = int((await session.execute(count_stmt)).scalar() or 0)

    order_column = getattr(trace_model, "created_at", None)
    stmt = select(trace_model).where(*filters)
    if order_column is not None:
        stmt = stmt.order_by(order_column.desc())
    stmt = stmt.limit(limit).offset(offset)

    rows = (await session.execute(stmt)).scalars().all()
    return {
        "items": [_trace_to_dict(row) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{trace_id}")
async def get_retrieval(
    project_id: str,
    trace_id: str,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Fetch one persisted retrieval trace (full per-stage detail)."""
    pid = _coerce_uuid(project_id)
    tid = _coerce_uuid(trace_id, field="trace_id")
    await _load_project(session, pid)

    trace_model = None
    for module_name, class_name in (
        ("app.db.models.retrieval_trace", "RetrievalTrace"),
        ("app.db.models.intent", "RetrievalTrace"),
        ("app.db.models.event", "RetrievalTrace"),
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        candidate = getattr(module, class_name, None)
        if candidate is not None:
            trace_model = candidate
            break
    if trace_model is None:
        raise NotFoundError(
            f"Retrieval trace {trace_id!r} not found",
            details={"trace_id": trace_id},
        )

    from sqlalchemy import select

    stmt = select(trace_model).where(
        trace_model.id == tid, trace_model.project_id == pid
    )
    row = (await session.execute(stmt)).scalars().first()
    if row is None:
        raise NotFoundError(
            f"Retrieval trace {trace_id!r} not found",
            details={"trace_id": trace_id, "project_id": str(pid)},
        )
    return _trace_to_dict(row)


@router.post("/bridge/preview", status_code=status.HTTP_200_OK)
async def preview_bridge(
    project_id: str,
    payload: Mapping[str, Any] = Body(
        default_factory=dict,
        description="Bridge preview request (intent_id of the concluded Explore).",
    ),
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Preview the three bridge blocks without injecting them (Algorithm 3).

    Useful for the Retrieval Trace UI and for debugging the theta gate; the
    real injection happens inside the worker after an EXPLORE conclusion.
    """
    pid = _coerce_uuid(project_id)
    project = await _load_project(session, pid)

    bridge = _bridge_module()
    if bridge is None:
        return {
            "project_id": str(pid),
            "injected": False,
            "reason": "bridge_unavailable",
            "blocks": {},
        }

    intent_id = payload.get("intent_id")
    builder = getattr(bridge, "preview_bridge", None) or getattr(
        bridge, "build_bridge", None
    )
    if not callable(builder):
        return {
            "project_id": str(pid),
            "injected": False,
            "reason": "bridge_api_unavailable",
            "blocks": {},
        }

    for kwargs in (
        {
            "session": session,
            "project": project,
            "project_id": pid,
            "intent_id": intent_id,
            "preview": True,
        },
        {
            "session": session,
            "project_id": pid,
            "intent_id": intent_id,
        },
        {"session": session, "project": project},
    ):
        try:
            result = builder(**kwargs)
        except TypeError:
            continue
        result = await _maybe_await(result)
        if isinstance(result, Mapping):
            return {
                "project_id": str(pid),
                "injected": bool(result.get("injected", True)),
                "reason": result.get("reason"),
                "score": result.get("score"),
                "blocks": _jsonable(result.get("blocks") or {}),
                "payload": _jsonable(result.get("payload")),
            }
        return {
            "project_id": str(pid),
            "injected": bool(result),
            "blocks": {},
        }

    raise AppError(
        "Bridge preview is not available",
        code=ErrorCode.INTERNAL,
    )


@router.get("/config/embedding")
async def embedding_status() -> dict[str, Any]:
    """Report which embedding backend produced the vectors (fallback chain)."""
    settings = deps.get_settings()
    provider = None
    module = _embedding_module()
    if module is not None:
        try:
            provider = deps.get_embedding_provider()
        except Exception as exc:  # pragma: no cover - configuration error
            return {
                "ok": False,
                "backend": None,
                "detail": f"{type(exc).__name__}: {exc}",
            }

    backend = None
    for name in ("backend", "backend_name", "provider"):
        value = getattr(provider, name, None)
        if isinstance(value, str):
            backend = value
            break
    if backend is None:
        backend = getattr(settings, "embedding_backend", None)

    dimension = None
    for name in ("dimension", "dim", "embedding_dim"):
        value = getattr(provider, name, None)
        if isinstance(value, int):
            dimension = value
            break
    if dimension is None:
        dimension = getattr(settings, "embedding_dim", None)

    model = getattr(provider, "model", None) or getattr(settings, "embedding_model", None)

    return {
        "ok": provider is not None,
        "backend": backend,
        "dimension": dimension,
        "model": model,
        "fallback_chain": ["api", "sentence-transformers", "hash"],
    }


__all__ = ["router", "STAGES"]
