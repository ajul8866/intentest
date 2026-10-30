"""Intent routes — the ``I`` half of the Fact–Intent DAG.

Mounted at ``/projects/{project_id}/intents``.  These endpoints expose the
intent nodes together with both hyperedge families of Definition 1:

* ``E_f→i`` — predecessor facts (``intent_source`` rows, roles
  ``PRECONDITION | SUPPORTING | GOAL_ANCHOR | CONTRADICTION``); and
* ``E_i→f`` — facts produced by an intent (``fact.derived_from_intent_id``).

Design notes (mirroring ``app/api/routes/facts.py``):

* routes stay thin — all mutation happens through
  :class:`app.db.repositories.intent_repo.IntentRepository` (CAS claim, FSM
  guards, ``dedupe_key`` UNIQUE) or, when that module cannot be imported
  (partial builds / unit tests), through a minimal ORM fallback;
* every model import is function-local so that ``import app.main`` stays
  dependency-light and tests can override dependencies;
* serializers are attribute-tolerant (ORM row, dataclass or mapping);
* there is **no** endpoint that mutates an ``intent_source`` row: the
  hyperedge is write-once (invariant **I2**).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Mapping, Optional, Sequence

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field

from app.api import deps
from app.api.errors import (
    ConflictError,
    IllegalTransitionError,
    NotFoundError,
    ValidationAppError,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/projects/{project_id}/intents", tags=["intents"])

__all__ = ["router", "IntentCreate", "IntentSourceCreate", "intent_to_dict"]

_MAX_LIMIT = 500
_VALID_STATES = ("CREATED", "CLAIMED", "CONCLUDED", "COMPLETED")
_VALID_ACTION_TYPES = ("BOOTSTRAP", "EXPLORE", "REASON", "VERIFY")
_VALID_SOURCE_ROLES = (
    "PRECONDITION",
    "SUPPORTING",
    "GOAL_ANCHOR",
    "CONTRADICTION",
)


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------
class IntentSourceCreate(BaseModel):
    """One ``E_f→i`` hyperedge to create together with the intent."""

    fact_id: str = Field(..., description="Predecessor fact id (UUID).")
    role: str = Field(
        "PRECONDITION",
        description="PRECONDITION | SUPPORTING | GOAL_ANCHOR | CONTRADICTION",
    )
    weight: float = Field(1.0, ge=0.0, le=1.0)


class IntentCreate(BaseModel):
    """Body for manually enqueueing an intent (expert / API path).

    In normal operation intents are created by the scheduler's
    ``TaskFactory`` or by the retrieval pipeline; this endpoint exists for
    operators and integration tests.  The same invariants apply: at least one
    verified predecessor fact is mandatory (invariant **I3**) and the
    ``dedupe_key`` UNIQUE constraint suppresses duplicates.
    """

    action_type: str = Field(
        "EXPLORE", description="BOOTSTRAP | EXPLORE | REASON | VERIFY"
    )
    tactic: str = Field(..., min_length=1, description="ATT&CK-style tactic.")
    objective: str = Field(..., min_length=1, description="Natural-language objective.")
    target_entity: Optional[str] = Field(None, description="Target host/endpoint/entity.")
    action_spec: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(0, description="Higher runs first within a task type.")
    depth: int = Field(0, ge=0)
    origin: str = Field("api", description="scheduler | retrieval | api | bootstrap")
    pattern_id: Optional[str] = Field(None, description="Offline intent-pattern id.")
    injected_context: Optional[dict[str, Any]] = None
    injection_meta: Optional[dict[str, Any]] = None
    sources: list[IntentSourceCreate] = Field(
        default_factory=list,
        description="E_f→i predecessors; at least one verified fact is required.",
    )


# ---------------------------------------------------------------------------
# serialisation helpers
# ---------------------------------------------------------------------------
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first present attribute/key among ``names``."""
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _iso(value: Any) -> Any:
    """ISO-format datetimes, stringify UUIDs/enums, pass everything else."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        try:
            return value.value
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return value


def _jsonable(value: Any) -> Any:
    """Best-effort JSON conversion for ``result``/``action_spec`` payloads."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return _iso(value)


def intent_to_dict(intent: Any) -> dict[str, Any]:
    """Tolerant serializer used by REST responses and WS payloads."""
    result = _attr(intent, "result")
    return {
        "id": _iso(_attr(intent, "id")),
        "project_id": _iso(_attr(intent, "project_id")),
        "action_type": _iso(_attr(intent, "action_type")),
        "tactic": _attr(intent, "tactic"),
        "objective": _attr(intent, "objective"),
        "target_entity": _attr(intent, "target_entity"),
        "action_spec": _jsonable(_attr(intent, "action_spec", default={})),
        "state": _iso(_attr(intent, "state")),
        "origin": _attr(intent, "origin"),
        "priority": _attr(intent, "priority", default=0),
        "depth": _attr(intent, "depth", default=0),
        "pattern_id": _attr(intent, "pattern_id"),
        "worker_id": _iso(_attr(intent, "worker_id")),
        "task_run_id": _iso(_attr(intent, "task_run_id")),
        "lease_expires_at": _iso(_attr(intent, "lease_expires_at")),
        "lease_heartbeat": _iso(
            _attr(intent, "lease_heartbeat", "lease_heartbeat_at")
        ),
        "round_index": _attr(intent, "round_index"),
        "success": _attr(intent, "success"),
        "result": _jsonable(result),
        "new_fact_ids": _iso_list(
            _attr(intent, "new_fact_ids")
            or (result.get("new_fact_ids") if isinstance(result, dict) else None)
        ),
        "injected_context": _jsonable(_attr(intent, "injected_context")),
        "injection_meta": _jsonable(_attr(intent, "injection_meta")),
        "dedupe_key": _attr(intent, "dedupe_key"),
        "created_at": _iso(_attr(intent, "created_at")),
        "updated_at": _iso(_attr(intent, "updated_at")),
        "concluded_at": _iso(_attr(intent, "concluded_at")),
    }


def _iso_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        return [_iso(v) for v in value]
    return [_iso(value)]


def source_to_dict(source: Any) -> dict[str, Any]:
    return {
        "id": _iso(_attr(source, "id")),
        "intent_id": _iso(_attr(source, "intent_id")),
        "fact_id": _iso(_attr(source, "fact_id")),
        "role": _iso(_attr(source, "role")),
        "weight": _attr(source, "weight", default=1.0),
        "created_at": _iso(_attr(source, "created_at")),
    }


# ---------------------------------------------------------------------------
# async plumbing helpers
# ---------------------------------------------------------------------------
async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _call_first(
    obj: Any, names: Sequence[str], *args: Any, **kwargs: Any
) -> Any:
    """Invoke the first available method on ``obj``; ``None`` if none work."""
    for name in names:
        func = getattr(obj, name, None)
        if not callable(func):
            continue
        try:
            return await _maybe_await(func(*args, **kwargs))
        except TypeError:
            continue
    return None


def _coerce_uuid(value: Any, *, field: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValidationAppError(
            f"{field} must be a UUID: {value!r}", details={"field": field}
        ) from exc


async def _load_project(session: Any, project_id: uuid.UUID) -> Any:
    from app.db.models.project import Project

    result = await session.execute(
        __import__("sqlalchemy").select(Project).where(Project.id == project_id)
    )
    row = result.scalars().first()
    if row is None:
        raise NotFoundError(
            f"project {project_id} not found", details={"project_id": str(project_id)}
        )
    return row


def _normalise(value: Any) -> Optional[str]:
    if value is None:
        return None
    enum_value = getattr(value, "value", value)
    return str(enum_value).upper()


async def _get_intent_or_404(
    session: Any, project_id: uuid.UUID, intent_id: uuid.UUID
) -> Any:
    from sqlalchemy import select

    from app.db.models.intent import Intent

    result = await session.execute(
        select(Intent).where(Intent.id == intent_id, Intent.project_id == project_id)
    )
    row = result.scalars().first()
    if row is None:
        raise NotFoundError(
            f"intent {intent_id} not found", details={"intent_id": str(intent_id)}
        )
    return row


async def _query_intents(
    session: Any,
    project_id: uuid.UUID,
    *,
    state: Optional[str] = None,
    action_type: Optional[str] = None,
    worker_id: Optional[str] = None,
    success: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[Any], int]:
    from sqlalchemy import func, select

    from app.db.models.intent import Intent

    stmt = select(Intent).where(Intent.project_id == project_id)
    if state:
        stmt = stmt.where(Intent.state == state.upper())
    if action_type:
        stmt = stmt.where(Intent.action_type == action_type.upper())
    if worker_id:
        stmt = stmt.where(Intent.worker_id == _coerce_uuid(worker_id, field="worker_id"))
    if success is not None:
        stmt = stmt.where(Intent.success.is_(success))

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = int((await session.execute(count_stmt)).scalar() or 0)

    stmt = stmt.order_by(Intent.created_at.asc(), Intent.id.asc())
    stmt = stmt.limit(max(1, min(int(limit), _MAX_LIMIT))).offset(max(0, int(offset)))
    rows = (await session.execute(stmt)).scalars().all()
    return list(rows), total


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
@router.get("")
async def list_intents(
    project_id: uuid.UUID,
    state: Optional[str] = Query(
        None, description="CREATED | CLAIMED | CONCLUDED | COMPLETED"
    ),
    action_type: Optional[str] = Query(
        None, description="BOOTSTRAP | EXPLORE | REASON | VERIFY"
    ),
    worker_id: Optional[str] = Query(None, description="Filter by worker id."),
    success: Optional[bool] = Query(None, description="Filter by terminal success."),
    limit: int = Query(100, ge=1, le=_MAX_LIMIT),
    offset: int = Query(0, ge=0),
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """List intents of a project with optional filters."""
    pid = _coerce_uuid(project_id, field="project_id")
    await _load_project(session, pid)

    if state and state.upper() not in _VALID_STATES:
        raise ValidationAppError(
            f"unknown intent state {state!r}",
            details={"allowed": list(_VALID_STATES)},
        )
    if action_type and action_type.upper() not in _VALID_ACTION_TYPES:
        raise ValidationAppError(
            f"unknown action_type {action_type!r}",
            details={"allowed": list(_VALID_ACTION_TYPES)},
        )

    rows, total = await _query_intents(
        session,
        pid,
        state=state,
        action_type=action_type,
        worker_id=worker_id,
        success=success,
        limit=limit,
        offset=offset,
    )
    return {
        "items": [intent_to_dict(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/states")
async def intent_state_counts(
    project_id: uuid.UUID, session: Any = Depends(deps.get_session)
) -> dict[str, Any]:
    """Counts per intent state / action type (declared before ``/{intent_id}``)."""
    from sqlalchemy import func, select

    from app.db.models.intent import Intent

    pid = _coerce_uuid(project_id, field="project_id")
    await _load_project(session, pid)

    stmt = (
        select(Intent.state, Intent.action_type, func.count())
        .where(Intent.project_id == pid)
        .group_by(Intent.state, Intent.action_type)
    )
    by_state: dict[str, int] = {s: 0 for s in _VALID_STATES}
    by_action: dict[str, int] = {a: 0 for a in _VALID_ACTION_TYPES}
    total = 0
    for state_value, action_value, count in (await session.execute(stmt)).all():
        state_key = _normalise(state_value) or "UNKNOWN"
        action_key = _normalise(action_value) or "UNKNOWN"
        by_state[state_key] = by_state.get(state_key, 0) + int(count)
        by_action[action_key] = by_action.get(action_key, 0) + int(count)
        total += int(count)
    return {"total": total, "by_state": by_state, "by_action_type": by_action}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_intent(
    project_id: uuid.UUID,
    payload: IntentCreate,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Create an intent together with its ``E_f→i`` predecessor edges.

    Invariant I3 is enforced by the repository/validator: every predecessor
    fact must exist in the same project and be ``verified``.  We perform a
    cheap pre-check here so the API returns a typed ``422`` rather than a
    database integrity error.
    """
    pid = _coerce_uuid(project_id, field="project_id")
    await _load_project(session, pid)

    action_type = payload.action_type.upper()
    if action_type not in _VALID_ACTION_TYPES:
        raise ValidationAppError(
            f"unknown action_type {payload.action_type!r}",
            details={"allowed": list(_VALID_ACTION_TYPES)},
        )
    for source in payload.sources:
        if source.role.upper() not in _VALID_SOURCE_ROLES:
            raise ValidationAppError(
                f"unknown intent_source role {source.role!r}",
                details={"allowed": list(_VALID_SOURCE_ROLES)},
            )

    precedent_ids = [
        _coerce_uuid(s.fact_id, field="sources[].fact_id") for s in payload.sources
    ]
    if precedent_ids:
        from sqlalchemy import select

        from app.db.models.fact import Fact

        rows = (
            await session.execute(
                select(Fact).where(
                    Fact.project_id == pid, Fact.id.in_(precedent_ids)
                )
            )
        ).scalars().all()
        found = {r.id for r in rows}
        missing = [str(f) for f in precedent_ids if f not in found]
        if missing:
            raise ValidationAppError(
                "predecessor facts do not exist in this project",
                details={"missing_fact_ids": missing},
            )
        unverified = [str(r.id) for r in rows if not bool(_attr(r, "verified"))]
        if unverified:
            from app.api.errors import PreconditionFailedError

            raise PreconditionFailedError(
                "every predecessor fact must be verified (invariant I3)",
                details={"unverified_fact_ids": unverified},
            )

    pattern_id = None
    if payload.pattern_id:
        try:
            pattern_id = _coerce_uuid(payload.pattern_id, field="pattern_id")
        except ValidationAppError:
            pattern_id = None

    values: dict[str, Any] = {
        "project_id": pid,
        "action_type": action_type,
        "tactic": payload.tactic,
        "objective": payload.objective,
        "target_entity": payload.target_entity,
        "action_spec": payload.action_spec or {},
        "origin": payload.origin,
        "priority": payload.priority,
        "depth": payload.depth,
        "pattern_id": pattern_id,
        "injected_context": payload.injected_context,
        "injection_meta": payload.injection_meta,
    }

    created: Any = None
    try:  # preferred path: repository owns dedupe_key + intent_source write-once
        from app.db.repositories.intent_repo import IntentRepository

        repo = IntentRepository(session)
        created = await _call_first(
            repo,
            (
                "create_with_sources",
                "create_intent_with_sources",
                "create_intent",
                "create",
            ),
            values,
            [s.model_dump() for s in payload.sources],
        )
        if created is None:
            created = await _call_first(
                repo, ("create_intent", "create", "add", "insert"), values
            )
    except ImportError:  # pragma: no cover - partial build
        created = None

    if created is None:
        from app.db.models.intent import Intent

        created = Intent(**values)
        session.add(created)
        await session.flush()

        if payload.sources:
            from app.db.models.intent_source import IntentSource

            for source, fact_id in zip(payload.sources, precedent_ids):
                session.add(
                    IntentSource(
                        intent_id=created.id,
                        fact_id=fact_id,
                        role=source.role.upper(),
                        weight=source.weight,
                    )
                )
            await session.flush()

    log.info(
        "intent.created_via_api",
        extra={
            "project_id": str(pid),
            "intent_id": str(_attr(created, "id")),
            "action_type": action_type,
            "sources": len(payload.sources),
        },
    )
    return intent_to_dict(created)


@router.get("/{intent_id}")
async def get_intent(
    project_id: uuid.UUID,
    intent_id: uuid.UUID,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Fetch one intent."""
    pid = _coerce_uuid(project_id, field="project_id")
    iid = _coerce_uuid(intent_id, field="intent_id")
    return intent_to_dict(await _get_intent_or_404(session, pid, iid))


@router.get("/{intent_id}/sources")
async def list_intent_sources(
    project_id: uuid.UUID,
    intent_id: uuid.UUID,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Return the ``E_f→i`` hyperedges (predecessor facts) of an intent."""
    from sqlalchemy import select

    from app.db.models.fact import Fact
    from app.db.models.intent_source import IntentSource

    pid = _coerce_uuid(project_id, field="project_id")
    iid = _coerce_uuid(intent_id, field="intent_id")
    await _get_intent_or_404(session, pid, iid)

    rows = (
        await session.execute(
            select(IntentSource)
            .where(IntentSource.intent_id == iid)
            .order_by(IntentSource.created_at.asc())
        )
    ).scalars().all()

    fact_ids = [r.fact_id for r in rows]
    facts: dict[Any, Any] = {}
    if fact_ids:
        fact_rows = (
            await session.execute(select(Fact).where(Fact.id.in_(fact_ids)))
        ).scalars().all()
        facts = {f.id: f for f in fact_rows}

    items = []
    for row in rows:
        payload = source_to_dict(row)
        fact = facts.get(row.fact_id)
        if fact is not None:
            payload["fact"] = {
                "id": str(fact.id),
                "kind": _normalise(_attr(fact, "kind")),
                "fact_key": _attr(fact, "fact_key"),
                "value": _jsonable(_attr(fact, "value")),
                "attributes": _jsonable(_attr(fact, "attributes", default={})),
                "verified": bool(_attr(fact, "verified")),
                "confidence": _attr(fact, "confidence"),
            }
        items.append(payload)

    return {"intent_id": str(iid), "items": items, "total": len(items)}


@router.get("/{intent_id}/facts")
async def list_intent_produced_facts(
    project_id: uuid.UUID,
    intent_id: uuid.UUID,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Return the facts produced by this intent (``E_i→f``, invariant I4)."""
    from sqlalchemy import select

    from app.db.models.fact import Fact

    pid = _coerce_uuid(project_id, field="project_id")
    iid = _coerce_uuid(intent_id, field="intent_id")
    await _get_intent_or_404(session, pid, iid)

    rows = (
        await session.execute(
            select(Fact)
            .where(Fact.derived_from_intent_id == iid)
            .order_by(Fact.created_at.asc(), Fact.id.asc())
        )
    ).scalars().all()

    from app.api.routes.facts import fact_to_dict

    items = [fact_to_dict(r) for r in rows]
    return {
        "intent_id": str(iid),
        "items": items,
        "total": len(items),
        "edges": [
            {
                "id": f"eif:{iid}->{_attr(r, 'id')}",
                "relation": "PRODUCES",
                "source": f"i:{iid}",
                "target": f"f:{_attr(r, 'id')}",
            }
            for r in rows
        ],
    }


@router.post("/{intent_id}/cancel")
async def cancel_intent(
    project_id: uuid.UUID,
    intent_id: uuid.UUID,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Cancel a pending intent.

    Only intents still in ``CREATED`` may be cancelled — ``CLAIMED`` intents
    are owned by a worker lease and are reclaimed by the reaper (T4) instead;
    terminal intents are immutable (invariant I6).
    """
    pid = _coerce_uuid(project_id, field="project_id")
    iid = _coerce_uuid(intent_id, field="intent_id")
    intent = await _get_intent_or_404(session, pid, iid)

    current = _normalise(_attr(intent, "state"))
    if current == "COMPLETED":
        raise IllegalTransitionError(
            "intent is already terminal (COMPLETED)",
            details={"intent_id": str(iid), "state": current},
        )
    if current != "CREATED":
        raise ConflictError(
            f"only CREATED intents can be cancelled (state={current})",
            details={"intent_id": str(iid), "state": current},
        )

    intent.state = "COMPLETED"
    result = _attr(intent, "result", default=None)
    result = dict(result) if isinstance(result, dict) else {}
    result.update({"reason": "cancelled_via_api", "cancelled": True})
    intent.result = result
    intent.success = False
    await session.flush()

    return intent_to_dict(intent)
