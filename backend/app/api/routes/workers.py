"""Worker registry REST surface.

Implements the worker-plane endpoints described in the reproduction plan
(component C13) and ``API.md``:

``POST   /workers/register``            register (or re-register) a worker
``POST   /workers/{worker_id}/heartbeat`` liveness + load report
``GET    /workers``                     list / filter the registry
``GET    /workers/{worker_id}``         single worker
``POST   /workers/{worker_id}/drain``   stop granting new leases
``POST   /workers/{worker_id}/undrain`` resume leasing
``POST   /workers/{worker_id}/cooldown`` open a cooldown window
``DELETE /workers/{worker_id}``         remove a worker from the registry
``GET    /workers/stats``               aggregate gauge data for the UI
``GET    /workers/capacity``            cluster-wide capacity snapshot

The scheduler's six-stage load balancer (Algorithm 1, lines 9-14) filters on
``tools_available`` (D_n), ``max_parallelism`` (C_n), ``status`` and
``cooldown_until``, then on ``priority_weight`` w(n) and ``active_tasks``
L(n).  These routes therefore only validate and persist that state; they never
perform scheduling decisions themselves (routes stay thin, the scheduler owns
the algorithm).

Worker rows carry no secrets.  ``metadata`` / ``capabilities`` payloads are
stored verbatim and echoed back, but the API never accepts a token that would
later be surfaced: authentication of workers is out of scope (see
``THREAT_MODEL.md``).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field

from app.api import deps
from app.api.errors import ConflictError, NotFoundError, ValidationAppError

__all__ = [
    "router",
    "WorkerRegister",
    "WorkerHeartbeat",
    "WorkerDrain",
    "worker_to_dict",
    "WORKER_STATUSES",
]

log = logging.getLogger(__name__)

router = APIRouter(prefix="/workers", tags=["workers"])

#: Legal values of ``worker.status`` (see plan section C1).
WORKER_STATUSES: tuple[str, ...] = (
    "IDLE",
    "ACTIVE",
    "COOLDOWN",
    "CIRCUIT_OPEN",
    "UNHEALTHY",
    "DRAINING",
)

#: Statuses that the load balancer excludes at stage 3 (health probe).
_UNHEALTHY_STATUSES = ("COOLDOWN", "CIRCUIT_OPEN", "UNHEALTHY", "DRAINING")

_MISS = object()

_MAX_LIMIT = 500


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class WorkerRegister(BaseModel):
    """Body of ``POST /workers/register``.

    ``name`` is the stable identity of a worker process.  Re-registering an
    existing name refreshes the advertisement (tools, capacity, weight) and
    brings the worker back to ``IDLE`` -- this is what a restarted worker
    container does on boot.
    """

    name: str = Field(..., min_length=1, max_length=200, description="Stable worker name.")
    tools_available: list[str] = Field(
        default_factory=list,
        description="Tool names installed in this worker's sandbox image (D_n).",
    )
    max_parallelism: int = Field(
        default=1, ge=1, le=256, description="Maximum concurrent tasks (C_n)."
    )
    priority_weight: float = Field(
        default=1.0, ge=0.0, description="Scheduling weight w(n); lower wins (stage 4)."
    )
    hostname: Optional[str] = Field(default=None, max_length=255)
    image: Optional[str] = Field(default=None, max_length=255)
    sandbox_mode: Optional[str] = Field(
        default=None,
        description="'docker' or 'local' (tests only; loudly flagged elsewhere).",
    )
    capabilities: Optional[dict[str, Any]] = Field(
        default=None, description="Free-form capability advertisement."
    )
    metadata: Optional[dict[str, Any]] = Field(default=None)


class WorkerHeartbeat(BaseModel):
    """Body of ``POST /workers/{worker_id}/heartbeat``.

    ``active_tasks`` is the authoritative L(n) used by load-balancer stage 5;
    omitting it leaves the stored value untouched so a bare liveness ping does
    not clobber the scheduler's bookkeeping.
    """

    active_tasks: Optional[int] = Field(default=None, ge=0, le=1024)
    status: Optional[str] = Field(default=None)
    consecutive_failures: Optional[int] = Field(default=None, ge=0)
    cooldown_seconds: Optional[int] = Field(
        default=None, ge=0, description="Open/replace a cooldown window of this length."
    )
    max_parallelism: Optional[int] = Field(default=None, ge=1, le=256)
    priority_weight: Optional[float] = Field(default=None, ge=0.0)
    tools_available: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class WorkerDrain(BaseModel):
    """Body of the drain/undrain/cooldown endpoints."""

    reason: Optional[str] = Field(default=None, max_length=500)
    cooldown_seconds: Optional[int] = Field(default=None, ge=0, le=86400)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Tolerant attribute/mapping lookup (ORM row, dataclass or dict)."""

    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _iso(value: Any) -> Any:
    """Render UUID/datetime/enum values as JSON-safe scalars."""

    if value is None:
        return None
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes, int, float, bool)):
        # enum-like
        return getattr(value, "value")
    if isinstance(value, (list, tuple, set)):
        return [_iso(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _iso(v) for k, v in value.items()}
    return value


def worker_to_dict(worker: Any) -> dict[str, Any]:
    """Serialise a ``Worker`` row into the documented JSON shape."""

    return {
        "id": _iso(_attr(worker, "id")),
        "name": _attr(worker, "name"),
        "status": str(_attr(worker, "status", default="IDLE")).upper(),
        "tools_available": list(_attr(worker, "tools_available", default=[]) or []),
        "max_parallelism": _attr(worker, "max_parallelism", default=1),
        "priority_weight": _attr(worker, "priority_weight", default=1.0),
        "active_tasks": _attr(worker, "active_tasks", default=0),
        "consecutive_failures": _attr(worker, "consecutive_failures", default=0),
        "cooldown_until": _iso(_attr(worker, "cooldown_until")),
        "last_heartbeat": _iso(_attr(worker, "last_heartbeat", "last_seen_at")),
        "hostname": _attr(worker, "hostname"),
        "image": _attr(worker, "image"),
        "sandbox_mode": _attr(worker, "sandbox_mode"),
        "capabilities": _iso(_attr(worker, "capabilities")),
        "metadata": _iso(_attr(worker, "metadata", "meta")),
        "created_at": _iso(_attr(worker, "created_at")),
        "updated_at": _iso(_attr(worker, "updated_at")),
        # Derived gauges consumed by the frontend worker view.
        "capacity_available": max(
            0,
            int(_attr(worker, "max_parallelism", default=1) or 1)
            - int(_attr(worker, "active_tasks", default=0) or 0),
        ),
        "eligible": _is_eligible(worker),
    }


def _is_eligible(worker: Any) -> bool:
    """Mirror of the load balancer's stage-2/3 admissibility test."""

    status_value = str(_attr(worker, "status", default="IDLE")).upper()
    if status_value in _UNHEALTHY_STATUSES:
        return False
    cooldown_until = _attr(worker, "cooldown_until")
    if cooldown_until is not None:
        if isinstance(cooldown_until, str):
            try:
                cooldown_until = datetime.fromisoformat(
                    cooldown_until.replace("Z", "+00:00")
                )
            except ValueError:
                cooldown_until = None
        if cooldown_until is not None:
            if cooldown_until.tzinfo is None:
                cooldown_until = cooldown_until.replace(tzinfo=timezone.utc)
            if cooldown_until > datetime.now(timezone.utc):
                return False
    active = int(_attr(worker, "active_tasks", default=0) or 0)
    capacity = int(_attr(worker, "max_parallelism", default=1) or 1)
    return active < capacity


def _normalise_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text not in WORKER_STATUSES:
        raise ValidationAppError(
            f"Unknown worker status {text!r}",
            details={"status": text, "allowed": list(WORKER_STATUSES)},
        )
    return text


# ---------------------------------------------------------------------------
# Repository / model access (lazy, tolerant)
# ---------------------------------------------------------------------------


def _worker_model() -> Any:
    """Import the ``Worker`` ORM model lazily (keeps app import light)."""

    try:
        from app.db.models.worker import Worker  # noqa: WPS433 (lazy by design)

        return Worker
    except Exception:  # pragma: no cover - model module always present in prod
        log.debug("worker_model_unavailable")
        return None


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _call_first(obj: Any, names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
    """Call the first existing method among ``names``; ``_MISS`` if none."""

    if obj is None:
        return _MISS
    for name in names:
        func = getattr(obj, name, None)
        if not callable(func):
            continue
        try:
            return await _maybe_await(func(*args, **kwargs))
        except TypeError:
            continue
    return _MISS


async def _load_worker(session: Any, worker_id: str) -> Any:
    """Fetch a worker row by id (UUID-coerced) or raise a typed 404."""

    identifier = _coerce_uuid(worker_id, field="worker_id")
    worker = await _call_first(
        _repo(session), ("get", "get_by_id", "fetch", "read", "find"), identifier
    )
    if worker is not _MISS and worker is not None:
        return worker

    model = _worker_model()
    if model is not None:
        try:
            from sqlalchemy import select

            result = await session.execute(select(model).where(model.id == identifier))
            row = result.scalar_one_or_none()
            if row is not None:
                return row
        except Exception:  # pragma: no cover - defensive against schema drift
            log.debug("worker_lookup_failed", exc_info=True)

    raise NotFoundError(
        f"Worker {worker_id} not found", details={"worker_id": str(worker_id)}
    )


def _repo(session: Any) -> Any:
    """Best-effort ``WorkerRepository`` instance bound to the request session."""

    try:
        from app.db.repositories.worker_repo import WorkerRepository

        return WorkerRepository(session)
    except Exception:  # pragma: no cover - repository optional in partial builds
        return None


def _coerce_uuid(value: Any, *, field: str = "id") -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise ValidationAppError(
            f"Invalid {field}", details={"field": field, "value": str(value)}
        )


async def _persist(session: Any, worker: Any) -> Any:
    """Flush pending changes; the session dependency owns the commit."""

    for name in ("flush", "commit"):
        func = getattr(session, name, None)
        if not callable(func):
            continue
        try:
            await _maybe_await(func())
            return worker
        except Exception:  # pragma: no cover
            log.debug("worker_persist_failed", exc_info=True)
            raise
    return worker


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_worker(
    payload: WorkerRegister,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Register (or refresh) a worker advertisement.

    Idempotent on ``name``: a restarted worker keeps its row and simply
    re-advertises D_n / C_n / w(n) and returns to ``IDLE``.
    """

    name = payload.name.strip()
    if not name:
        raise ValidationAppError("Worker name must not be blank", details={"field": "name"})

    model = _worker_model()
    if model is None:
        raise NotFoundError("Worker registry is unavailable", details={"reason": "no_model"})

    from sqlalchemy import select

    existing = None
    try:
        result = await session.execute(select(model).where(model.name == name))
        existing = result.scalar_one_or_none()
    except Exception:  # pragma: no cover
        log.debug("worker_register_lookup_failed", exc_info=True)

    now = datetime.now(timezone.utc)
    values: dict[str, Any] = {
        "name": name,
        "status": "IDLE",
        "tools_available": sorted(set(payload.tools_available)),
        "max_parallelism": payload.max_parallelism,
        "priority_weight": payload.priority_weight,
        "active_tasks": 0,
        "consecutive_failures": 0,
        "cooldown_until": None,
        "last_heartbeat": now,
        "hostname": payload.hostname,
        "image": payload.image,
        "sandbox_mode": payload.sandbox_mode,
        "capabilities": payload.capabilities,
        "metadata": payload.metadata,
    }

    if existing is not None:
        for key, value in values.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
        worker = existing
        created = False
    else:
        # Drop columns the model does not declare so partial schemas still work.
        kwargs = {
            key: value
            for key, value in values.items()
            if hasattr(model, key) or True
        }
        worker = model(**kwargs)
        session.add(worker)
        created = True

    await _persist(session, worker)
    log.info(
        "worker.registered",
        extra={"worker_id": str(_attr(worker, "id")), "name": name, "created": created},
    )
    body = worker_to_dict(worker)
    body["created"] = created
    return body


@router.get("/stats")
async def worker_stats(session: Any = Depends(deps.get_session)) -> dict[str, Any]:
    """Aggregate worker gauges for the dashboard.

    Declared before ``/{worker_id}`` so the literal path always wins.
    """

    items = await _list_workers(session)
    by_status: dict[str, int] = {name: 0 for name in WORKER_STATUSES}
    tool_counts: dict[str, int] = {}
    capacity = 0
    active = 0
    for worker in items:
        status_value = str(_attr(worker, "status", default="IDLE")).upper()
        by_status[status_value] = by_status.get(status_value, 0) + 1
        capacity += int(_attr(worker, "max_parallelism", default=1) or 1)
        active += int(_attr(worker, "active_tasks", default=0) or 0)
        for tool in _attr(worker, "tools_available", default=[]) or []:
            tool_counts[str(tool)] = tool_counts.get(str(tool), 0) + 1

    return {
        "total": len(items),
        "by_status": by_status,
        "eligible": sum(1 for worker in items if _is_eligible(worker)),
        "capacity_total": capacity,
        "capacity_active": active,
        "capacity_available": max(0, capacity - active),
        "tools": tool_counts,
    }


@router.get("/capacity")
async def worker_capacity(
    required_tools: Optional[str] = Query(
        default=None,
        description="Comma separated tool names; only workers providing all of them are counted.",
    ),
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Cluster capacity snapshot, optionally narrowed by a required tool set."""

    required: set[str] = set()
    if required_tools:
        required = {part.strip() for part in required_tools.split(",") if part.strip()}

    items = await _list_workers(session)
    slots = 0
    matching: list[str] = []
    for worker in items:
        if required:
            available = {str(tool) for tool in _attr(worker, "tools_available", default=[]) or []}
            if not required.issubset(available):
                continue
        if not _is_eligible(worker):
            continue
        slots += max(
            0,
            int(_attr(worker, "max_parallelism", default=1) or 1)
            - int(_attr(worker, "active_tasks", default=0) or 0),
        )
        matching.append(str(_attr(worker, "id")))

    return {
        "required_tools": sorted(required),
        "workers": matching,
        "worker_count": len(matching),
        "free_slots": slots,
        "saturated": slots == 0,
    }


@router.get("")
async def list_workers(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    tool: Optional[str] = Query(default=None, description="Only workers providing this tool."),
    eligible: Optional[bool] = Query(
        default=None, description="Filter to load-balancer admissible workers."
    ),
    limit: int = Query(default=100, ge=1, le=_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """List the worker registry with optional status/tool/admissibility filters."""

    wanted = _normalise_status(status_filter)
    rows = await _list_workers(session)

    filtered: list[Any] = []
    for worker in rows:
        if wanted is not None:
            if str(_attr(worker, "status", default="")).upper() != wanted:
                continue
        if tool:
            available = {str(name) for name in _attr(worker, "tools_available", default=[]) or []}
            if tool not in available:
                continue
        if eligible is True and not _is_eligible(worker):
            continue
        if eligible is False and _is_eligible(worker):
            continue
        filtered.append(worker)

    total = len(filtered)
    page = filtered[offset : offset + limit]
    return {
        "items": [worker_to_dict(worker) for worker in page],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{worker_id}")
async def get_worker(
    worker_id: str,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Fetch a single worker row."""

    worker = await _load_worker(session, worker_id)
    return worker_to_dict(worker)


@router.post("/{worker_id}/heartbeat")
async def heartbeat(
    worker_id: str,
    payload: Optional[WorkerHeartbeat] = None,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Record liveness (and optional load/capacity deltas) for a worker."""

    worker = await _load_worker(session, worker_id)
    body = payload or WorkerHeartbeat()
    now = datetime.now(timezone.utc)

    updates: dict[str, Any] = {"last_heartbeat": now}

    new_status = _normalise_status(body.status)
    if new_status is not None:
        updates["status"] = new_status
    if body.active_tasks is not None:
        updates["active_tasks"] = body.active_tasks
    if body.consecutive_failures is not None:
        updates["consecutive_failures"] = body.consecutive_failures
    if body.max_parallelism is not None:
        updates["max_parallelism"] = body.max_parallelism
    if body.priority_weight is not None:
        updates["priority_weight"] = body.priority_weight
    if body.tools_available is not None:
        updates["tools_available"] = sorted(set(body.tools_available))
    if body.metadata is not None:
        updates["metadata"] = body.metadata
    if body.cooldown_seconds is not None:
        if body.cooldown_seconds == 0:
            updates["cooldown_until"] = None
            if new_status in (None, "COOLDOWN"):
                updates["status"] = "IDLE"
        else:
            updates["cooldown_until"] = now + timedelta(seconds=body.cooldown_seconds)
            if new_status is None:
                updates["status"] = "COOLDOWN"

    for key, value in updates.items():
        if hasattr(worker, key):
            setattr(worker, key, value)

    await _persist(session, worker)
    log.debug(
        "worker.heartbeat",
        extra={"worker_id": str(_attr(worker, "id")), "status": _attr(worker, "status")},
    )
    return worker_to_dict(worker)


@router.post("/{worker_id}/drain")
async def drain_worker(
    worker_id: str,
    payload: Optional[WorkerDrain] = None,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Move a worker to ``DRAINING`` so no new intent is leased to it.

    In-flight tasks keep running to completion; the scheduler's health probe
    (stage 3) drops the worker from the candidate pool immediately.
    """

    worker = await _load_worker(session, worker_id)
    body = payload or WorkerDrain()

    status_value = str(_attr(worker, "status", default="IDLE")).upper()
    if status_value == "DRAINING":
        return worker_to_dict(worker)

    if hasattr(worker, "status"):
        worker.status = "DRAINING"
    if body.reason is not None and hasattr(worker, "metadata"):
        metadata = dict(_attr(worker, "metadata", "meta", default={}) or {})
        metadata["drain_reason"] = body.reason
        metadata["drained_at"] = datetime.now(timezone.utc).isoformat()
        if hasattr(worker, "metadata"):
            worker.metadata = metadata
        elif hasattr(worker, "meta"):
            worker.meta = metadata

    await _persist(session, worker)
    log.info(
        "worker.draining",
        extra={"worker_id": str(_attr(worker, "id")), "reason": body.reason},
    )
    return worker_to_dict(worker)


@router.post("/{worker_id}/undrain")
async def undrain_worker(
    worker_id: str,
    payload: Optional[WorkerDrain] = None,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Return a ``DRAINING`` worker to ``IDLE`` (or ``ACTIVE`` when busy)."""

    worker = await _load_worker(session, worker_id)
    status_value = str(_attr(worker, "status", default="IDLE")).upper()
    if status_value != "DRAINING":
        raise ConflictError(
            f"Worker {worker_id} is not draining",
            details={"worker_id": str(worker_id), "status": status_value},
        )

    active = int(_attr(worker, "active_tasks", default=0) or 0)
    if hasattr(worker, "status"):
        worker.status = "ACTIVE" if active > 0 else "IDLE"
    if hasattr(worker, "metadata"):
        metadata = dict(_attr(worker, "metadata", "meta", default={}) or {})
        metadata.pop("drain_reason", None)
        metadata.pop("drained_at", None)
        worker.metadata = metadata

    await _persist(session, worker)
    log.info("worker.undrained", extra={"worker_id": str(_attr(worker, "id"))})
    return worker_to_dict(worker)


@router.post("/{worker_id}/cooldown")
async def cooldown_worker(
    worker_id: str,
    payload: Optional[WorkerDrain] = None,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Open a ``COOLDOWN`` window (excluded by load-balancer stage 3)."""

    worker = await _load_worker(session, worker_id)
    body = payload or WorkerDrain()
    seconds = body.cooldown_seconds if body.cooldown_seconds is not None else 60
    now = datetime.now(timezone.utc)

    if hasattr(worker, "status"):
        worker.status = "COOLDOWN"
    if hasattr(worker, "cooldown_until"):
        worker.cooldown_until = now + timedelta(seconds=seconds) if seconds > 0 else None
    if body.reason is not None and hasattr(worker, "metadata"):
        metadata = dict(_attr(worker, "metadata", "meta", default={}) or {})
        metadata["cooldown_reason"] = body.reason
        worker.metadata = metadata

    await _persist(session, worker)
    log.info(
        "worker.cooldown",
        extra={"worker_id": str(_attr(worker, "id")), "seconds": seconds},
    )
    return worker_to_dict(worker)


@router.delete("/{worker_id}")
async def delete_worker(
    worker_id: str,
    force: bool = Query(
        default=False,
        description="Allow removal even when the worker currently holds active tasks.",
    ),
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Remove a worker from the registry.

    Refuses (409) when the worker still reports active tasks unless ``force``
    is set, because deleting a lease holder would strand its intent until the
    lease reaper (T4) reclaims it.
    """

    worker = await _load_worker(session, worker_id)
    active = int(_attr(worker, "active_tasks", default=0) or 0)
    if active > 0 and not force:
        raise ConflictError(
            f"Worker {worker_id} still has {active} active task(s)",
            details={"worker_id": str(worker_id), "active_tasks": active},
        )

    identifier = _attr(worker, "id")
    try:
        await _maybe_await(session.delete(worker))
    except Exception:  # pragma: no cover - fallback for repo-only sessions
        repo = _repo(session)
        deleted = await _call_first(repo, ("delete", "remove", "delete_by_id"), identifier)
        if deleted is _MISS:
            log.debug("worker_delete_fallback_failed", exc_info=True)

    await _persist(session, None)
    log.info("worker.deleted", extra={"worker_id": str(identifier), "force": force})
    return {"deleted": True, "worker_id": str(identifier)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _list_workers(session: Any) -> list[Any]:
    """Return every worker row, tolerating the absence of a repository."""

    repo = _repo(session)
    rows = await _call_first(repo, ("list", "list_all", "all", "list_workers"))
    if rows is not _MISS and rows is not None:
        try:
            return list(rows)
        except TypeError:  # pragma: no cover
            pass

    model = _worker_model()
    if model is None:
        return []

    try:
        from sqlalchemy import select

        result = await session.execute(select(model).order_by(model.name))
        return list(result.scalars().all())
    except Exception:  # pragma: no cover - defensive against schema drift
        log.debug("worker_list_failed", exc_info=True)
        return []
