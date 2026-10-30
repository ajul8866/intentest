"""Project lifecycle and scope REST routes.

Mounted at ``/projects`` (the router is included by :mod:`app.main` under the
``/api/v1`` prefix).  This module implements the project-facing half of the
API surface described in ``API.md``:

* CRUD on ``project`` rows (``POST/GET/PATCH/DELETE /projects``);
* lifecycle transitions ``start`` / ``pause`` / ``resume`` / ``cancel`` and a
  compact ``status`` view that the dashboard polls when the WebSocket is not
  available;
* nested scope management (``/projects/{id}/scopes``) -- scopes are the
  isolation namespace that the scope guard consults before every tool call.

Routes stay deliberately thin: payload validation happens here (Pydantic),
state changes are delegated to :mod:`app.services.project_service` when it is
importable and otherwise fall back to a tolerant repository/ORM path so the
API keeps working while the service layer is being built out.

Invariants enforced at this boundary
------------------------------------
* ``DELETE`` is only allowed when the project is **not** ``RUNNING``
  (otherwise 409 ``CONFLICT``).
* ``start`` seeds the ``BOOTSTRAP`` intent via ``bootstrap_service`` and moves
  ``INITIALIZING -> RUNNING``; ``pause`` moves ``RUNNING -> STALLED`` and
  releases the project lease; ``resume`` moves ``STALLED -> RUNNING``;
  ``cancel`` moves any non-terminal state to ``CANCELLED`` and bulk-completes
  unfinished intents with ``result.reason = "CANCELLED"`` (mirrors the goal
  freeze path, invariant I6).
* Scopes are unique per ``(project_id, name)``; deletion is refused with 409
  when facts still reference the scope.
"""

from __future__ import annotations

import importlib
import logging
import uuid
from typing import Any, Iterable, Mapping, Optional, Sequence

from fastapi import APIRouter, Body, Depends, Query, Response, status
from pydantic import BaseModel, Field

from app.api import deps
from app.api.errors import (
    ConflictError,
    NotFoundError,
    ValidationAppError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["projects"])

__all__ = [
    "router",
    "ProjectCreate",
    "ProjectUpdate",
    "ScopeCreate",
    "project_to_dict",
    "scope_to_dict",
]

_MISS = object()

_TERMINAL_STATUSES = ("COMPLETED", "FAILED", "CANCELLED")
_NON_DELETABLE_STATUSES = ("RUNNING",)

_DEFAULT_FLAG_REGEX = r"flag\{[^}]+\}"


# ---------------------------------------------------------------------------
# Pydantic payload models
# ---------------------------------------------------------------------------
class ScopeCreate(BaseModel):
    """Body for ``POST /projects/{id}/scopes``."""

    name: str = Field(..., min_length=1, max_length=128)
    allowed_hosts: list[str] = Field(default_factory=list)
    allowed_cidrs: list[str] = Field(default_factory=list)
    allowed_ports: list[int] = Field(default_factory=list)
    denied_actions: list[str] = Field(default_factory=list)


class ProjectCreate(BaseModel):
    """Body for ``POST /projects`` (see ``API.md``)."""

    name: str = Field(..., min_length=1, max_length=200)
    goal_text: str = Field(
        default="Obtain the flag from the target application.",
        description="Natural-language goal statement stored on the project.",
    )
    goal_anchor: dict[str, Any] = Field(
        default_factory=lambda: {"flag_regex": _DEFAULT_FLAG_REGEX},
        description="Structured goal anchor, normally {'flag_regex': ...}.",
    )
    scope: Optional[ScopeCreate] = None
    max_rounds: int = Field(default=40, ge=1, le=1000)
    config: dict[str, Any] = Field(default_factory=dict)


class ProjectUpdate(BaseModel):
    """Body for ``PATCH /projects/{id}`` -- every field optional."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    goal_text: Optional[str] = None
    goal_anchor: Optional[dict[str, Any]] = None
    max_rounds: Optional[int] = Field(default=None, ge=1, le=1000)
    config: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Tolerant attribute access across ORM rows, dataclasses and mappings."""

    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        try:
            value = getattr(obj, name)
        except (AttributeError, TypeError):
            continue
        if value is not None:
            return value
    for name in names:
        try:
            return getattr(obj, name)
        except (AttributeError, TypeError):
            continue
    return default


def _iso(value: Any) -> Any:
    """Stringify UUIDs/enums/datetimes, pass everything else through."""

    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes, int, float)):
        try:
            return _iso(value.value)
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_iso(item) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return _iso(value)


def _status_of(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value") and not isinstance(value, str):
        value = value.value
    return str(value).upper()


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__") or hasattr(value, "__aiter__"):
        return await value
    return value


async def _try_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``func`` and normalise the (possibly async) result.

    ``TypeError`` signals a signature mismatch during partial builds and is
    reported to the caller as the module sentinel ``_MISS``.
    """

    if func is None:
        return _MISS
    try:
        return await _maybe_await(func(*args, **kwargs))
    except TypeError:
        return _MISS


def _coerce_uuid(value: Any, *, field: str) -> Any:
    if isinstance(value, uuid.UUID):
        return value
    if value is None:
        raise ValidationAppError(
            f"Invalid {field}", details={"field": field, "value": None}
        )
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValidationAppError(
            f"Invalid {field}", details={"field": field, "value": str(value)}
        ) from exc


def project_to_dict(project: Any) -> dict[str, Any]:
    """Serialize a ``project`` row (ORM, dataclass or mapping)."""

    return {
        "id": _iso(_attr(project, "id")),
        "name": _attr(project, "name"),
        "goal_text": _attr(project, "goal_text", "goal"),
        "goal_anchor_fact": _iso(_attr(project, "goal_anchor_fact", "goal_anchor")),
        "goal_anchor": _jsonable(_attr(project, "goal_anchor")),
        "status": _status_of(_attr(project, "status")),
        "phase": _iso(_attr(project, "phase")),
        "max_rounds": _attr(project, "max_rounds", default=40),
        "rounds_used": _attr(project, "rounds_used", default=0),
        "facts_since_reason": _attr(project, "facts_since_reason", default=0),
        "last_fact_at": _iso(_attr(project, "last_fact_at")),
        "lease_owner": _iso(_attr(project, "lease_owner")),
        "lease_acquired_at": _iso(
            _attr(project, "lease_acquired_at", "lease_acquired")
        ),
        "lease_expires_at": _iso(_attr(project, "lease_expires_at", "lease_expires")),
        "lease_heartbeat": _iso(_attr(project, "lease_heartbeat")),
        "termination_reason": _iso(_attr(project, "termination_reason")),
        "config": _jsonable(_attr(project, "config", default={}) or {}),
        "created_at": _iso(_attr(project, "created_at")),
        "updated_at": _iso(_attr(project, "updated_at")),
        "started_at": _iso(_attr(project, "started_at")),
        "completed_at": _iso(_attr(project, "completed_at", "finished_at")),
    }


def scope_to_dict(scope: Any) -> dict[str, Any]:
    """Serialize a ``scope`` row."""

    return {
        "id": _iso(_attr(scope, "id")),
        "project_id": _iso(_attr(scope, "project_id")),
        "name": _attr(scope, "name"),
        "allowed_hosts": list(_attr(scope, "allowed_hosts", default=[]) or []),
        "allowed_cidrs": list(_attr(scope, "allowed_cidrs", default=[]) or []),
        "allowed_ports": list(_attr(scope, "allowed_ports", default=[]) or []),
        "denied_actions": list(_attr(scope, "denied_actions", default=[]) or []),
        "created_at": _iso(_attr(scope, "created_at")),
        "updated_at": _iso(_attr(scope, "updated_at")),
    }


# ---------------------------------------------------------------------------
# Model / service resolution (lazy so `import app.main` stays light)
# ---------------------------------------------------------------------------
def _model(module_name: str, class_name: str) -> Any:
    module = importlib.import_module(f"app.db.models.{module_name}")
    return getattr(module, class_name)


def _repo() -> Any:
    """Return a ``ProjectRepository`` class, or ``None`` when unavailable."""

    for module_name in ("app.db.repositories.project_repo", "app.db.repositories.projects"):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - import guards
            continue
        for attr in ("ProjectRepository", "ProjectRepo", "ProjectsRepository"):
            cls = getattr(module, attr, None)
            if cls is not None:
                return cls
    return None


def _scope_repo() -> Any:
    for module_name in (
        "app.db.repositories.project_repo",
        "app.db.repositories.scope_repo",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - import guards
            continue
        for attr in ("ScopeRepository", "ScopeRepo"):
            cls = getattr(module, attr, None)
            if cls is not None:
                return cls
    return None


def _service() -> Any:
    try:
        return importlib.import_module("app.services.project_service")
    except Exception:  # pragma: no cover - optional service
        return None


async def _service_call(names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
    module = _service()
    if module is None:
        return _MISS
    for name in names:
        func = getattr(module, name, None)
        if func is None:
            continue
        result = await _try_call(func, *args, **kwargs)
        if result is not _MISS:
            return result
    return _MISS


async def _bootstrap_call(project: Any, session: Any) -> None:
    """Best-effort BOOTSTRAP seeding when a project is started."""

    for module_name in ("app.services.bootstrap_service", "app.services.project_service"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name in ("seed_bootstrap", "bootstrap_project", "create_bootstrap_intent", "ensure_bootstrap"):
            func = getattr(module, name, None)
            if func is None:
                continue
            for call_args in ((project, session), (session, project), (project,)):
                result = await _try_call(func, *call_args)
                if result is not _MISS:
                    return


# ---------------------------------------------------------------------------
# Low-level DB helpers (fallback path)
# ---------------------------------------------------------------------------
async def _fetch_project(session: Any, project_id: Any) -> Any:
    Project = _model("project", "Project")
    from sqlalchemy import select

    result = await session.execute(
        select(Project).where(Project.id == _coerce_uuid(project_id, field="project_id"))
    )
    project = result.scalars().first()
    if project is None:
        raise NotFoundError(
            f"Project {project_id} not found", details={"project_id": str(project_id)}
        )
    return project


async def _get_project_via_repo(session: Any, project_id: Any) -> Any:
    repo_cls = _repo()
    if repo_cls is not None:
        try:
            repo = repo_cls(session)
        except TypeError:
            repo = repo_cls(session=session)
        for name in ("get", "get_by_id", "fetch", "read", "find_by_id"):
            func = getattr(repo, name, None)
            result = await _try_call(func, project_id)
            if result is not _MISS and result is not None:
                return result
    return await _fetch_project(session, project_id)


async def _list_projects(session: Any) -> list[Any]:
    Project = _model("project", "Project")
    from sqlalchemy import select

    result = await session.execute(select(Project).order_by(Project.created_at.desc()))
    return list(result.scalars().all())


async def _list_scopes(session: Any, project_id: Any) -> list[Any]:
    Scope = _model("scope", "Scope")
    from sqlalchemy import select

    result = await session.execute(
        select(Scope).where(Scope.project_id == project_id).order_by(Scope.created_at.asc())
    )
    return list(result.scalars().all())


async def _flush(session: Any) -> None:
    flush = getattr(session, "flush", None)
    if flush is not None:
        await _maybe_await(flush())


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------
@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project(
    payload: ProjectCreate,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Create a project together with its first scope."""

    values = {
        "name": payload.name,
        "goal_text": payload.goal_text,
        "goal_anchor": dict(payload.goal_anchor or {}),
        "max_rounds": payload.max_rounds,
        "config": dict(payload.config or {}),
        "status": "INITIALIZING",
        "rounds_used": 0,
        "facts_since_reason": 0,
    }

    service_result = await _service_call(
        ("create_project", "create"),
        payload.model_dump(),
        session,
    )
    if service_result is not _MISS and service_result is not None:
        project = service_result
    else:
        Project = _model("project", "Project")
        project = Project(**values)
        session.add(project)
        await _flush(session)

    scope_value = payload.scope
    if scope_value is not None:
        await _create_scope_row(session, _attr(project, "id"), scope_value)

    logger.info(
        "project.created",
        extra={
            "project_id": _iso(_attr(project, "id")),
            "name": payload.name,
            "has_scope": scope_value is not None,
        },
    )
    return project_to_dict(project)


@router.get("")
async def list_projects(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    page: Any = Depends(deps.pagination),
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """List projects, newest first, with optional status filtering."""

    rows = await _list_projects(session)
    if status_filter:
        wanted = status_filter.upper()
        rows = [row for row in rows if _status_of(_attr(row, "status")) == wanted]
    total = len(rows)
    items = rows
    try:
        start, stop = page.slice_bounds()
        items = rows[start:stop]
    except Exception:  # pragma: no cover - pagination shape tolerance
        pass
    return {"items": [project_to_dict(row) for row in items], "total": total}


@router.get("/{project_id}")
async def get_project(
    project_id: str,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    project = await _get_project_via_repo(session, project_id)
    return project_to_dict(project)


@router.patch("/{project_id}")
async def update_project(
    project_id: str,
    payload: ProjectUpdate,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    project = await _get_project_via_repo(session, project_id)
    values = {k: v for k, v in payload.model_dump(exclude_unset=True).items() if v is not None}

    service_result = await _service_call(
        ("update_project", "update"), project, values, session
    )
    if service_result is not _MISS and service_result is not None:
        project = service_result
    else:
        for key, value in values.items():
            setattr(project, key, value)
        await _flush(session)

    return project_to_dict(project)


@router.delete("/{project_id}", status_code=status.HTTP_200_OK)
async def delete_project(
    project_id: str,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    project = await _get_project_via_repo(session, project_id)
    current = _status_of(_attr(project, "status"))
    if current in _NON_DELETABLE_STATUSES:
        raise ConflictError(
            "Cannot delete a running project; cancel it first",
            details={"project_id": str(project_id), "status": current},
        )

    service_result = await _service_call(
        ("delete_project", "delete", "remove"), project, session
    )
    if service_result is _MISS:
        await _maybe_await(session.delete(project))
        await _flush(session)

    logger.info("project.deleted", extra={"project_id": str(project_id)})
    return {"deleted": True, "project_id": str(project_id)}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
async def _set_status(project: Any, value: str, *, session: Any) -> None:
    try:
        setattr(project, "status", value)
    except Exception:  # pragma: no cover - mapping-backed projects
        if isinstance(project, Mapping):
            project["status"] = value  # type: ignore[index]
    await _flush(session)


async def _release_lease(project: Any) -> None:
    for attr in ("lease_owner", "lease_expires_at", "lease_acquired_at", "lease_heartbeat"):
        try:
            setattr(project, attr, None)
        except Exception:  # pragma: no cover - defensive
            pass


@router.post("/{project_id}/start")
async def start_project(
    project_id: str,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """``INITIALIZING|STALLED -> RUNNING`` and seed the BOOTSTRAP intent."""

    project = await _get_project_via_repo(session, project_id)
    current = _status_of(_attr(project, "status"))
    if current == "RUNNING":
        raise ConflictError(
            "Project is already running",
            details={"project_id": str(project_id), "status": current},
        )
    if current in _TERMINAL_STATUSES:
        raise ConflictError(
            "Project has terminated and cannot be restarted",
            details={"project_id": str(project_id), "status": current},
        )

    await _set_status(project, "RUNNING", session=session)
    await _bootstrap_call(project, session)
    await _flush(session)

    logger.info("project.started", extra={"project_id": str(project_id)})
    return project_to_dict(project)


@router.post("/{project_id}/pause")
async def pause_project(
    project_id: str,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """``RUNNING -> STALLED`` and release the scheduler project lease."""

    project = await _get_project_via_repo(session, project_id)
    current = _status_of(_attr(project, "status"))
    if current != "RUNNING":
        raise ConflictError(
            "Only a running project can be paused",
            details={"project_id": str(project_id), "status": current},
        )
    await _set_status(project, "STALLED", session=session)
    await _release_lease(project)
    await _flush(session)
    logger.info("project.paused", extra={"project_id": str(project_id)})
    return project_to_dict(project)


@router.post("/{project_id}/resume")
async def resume_project(
    project_id: str,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """``STALLED -> RUNNING`` (manual override of the stall detector)."""

    project = await _get_project_via_repo(session, project_id)
    current = _status_of(_attr(project, "status"))
    if current == "RUNNING":
        raise ConflictError(
            "Project is already running",
            details={"project_id": str(project_id), "status": current},
        )
    if current in _TERMINAL_STATUSES:
        raise ConflictError(
            "Project has terminated and cannot be resumed",
            details={"project_id": str(project_id), "status": current},
        )
    await _set_status(project, "RUNNING", session=session)
    logger.info("project.resumed", extra={"project_id": str(project_id)})
    return project_to_dict(project)


@router.post("/{project_id}/cancel")
async def cancel_project(
    project_id: str,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Terminate a project and freeze every unfinished intent."""

    project = await _get_project_via_repo(session, project_id)
    current = _status_of(_attr(project, "status"))
    if current == "CANCELLED":
        return project_to_dict(project)

    service_result = await _service_call(
        ("cancel_project", "cancel"), project, session
    )
    if service_result is _MISS:
        await _set_status(project, "CANCELLED", session=session)
        try:
            setattr(project, "termination_reason", "CANCELLED")
        except Exception:  # pragma: no cover - defensive
            pass
        await _release_lease(project)
        await _freeze_unfinished_intents(session, _attr(project, "id"), "CANCELLED")
        await _flush(session)
    elif service_result is not None:
        project = service_result

    logger.info("project.cancelled", extra={"project_id": str(project_id)})
    return project_to_dict(project)


async def _freeze_unfinished_intents(session: Any, project_id: Any, reason: str) -> None:
    """Bulk-complete intents still in an open state (mirrors the goal freeze)."""

    try:
        Intent = _model("intent", "Intent")
    except Exception:  # pragma: no cover - partial builds
        return
    from sqlalchemy import select

    closed = ("COMPLETED",)
    result = await session.execute(
        select(Intent).where(Intent.project_id == project_id)
    )
    for intent in result.scalars().all():
        state = _status_of(_attr(intent, "state"))
        if state in closed:
            continue
        try:
            intent.state = "COMPLETED"
            intent.success = False
            existing = _attr(intent, "result", default=None) or {}
            merged = dict(existing) if isinstance(existing, Mapping) else {}
            merged.update({"reason": reason})
            intent.result = merged
        except Exception:  # pragma: no cover - defensive
            continue
    await _flush(session)


@router.get("/{project_id}/status")
async def project_status(
    project_id: str,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Compact lifecycle view used by the dashboard polling fallback."""

    project = await _get_project_via_repo(session, project_id)
    data = project_to_dict(project)
    return {
        "project_id": data["id"],
        "status": data["status"],
        "phase": data["phase"],
        "rounds_used": data["rounds_used"],
        "max_rounds": data["max_rounds"],
        "facts_since_reason": data["facts_since_reason"],
        "termination_reason": data["termination_reason"],
        "lease_owner": data["lease_owner"],
        "lease_expires_at": data["lease_expires_at"],
        "last_fact_at": data["last_fact_at"],
    }


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------
async def _create_scope_row(
    session: Any, project_id: Any, payload: ScopeCreate
) -> Any:
    Scope = _model("scope", "Scope")
    existing = await _list_scopes(session, project_id)
    for row in existing:
        if _attr(row, "name") == payload.name:
            raise ConflictError(
                "A scope with this name already exists for the project",
                details={"project_id": str(project_id), "name": payload.name},
            )
    scope = Scope(
        project_id=project_id,
        name=payload.name,
        allowed_hosts=list(payload.allowed_hosts),
        allowed_cidrs=list(payload.allowed_cidrs),
        allowed_ports=list(payload.allowed_ports),
        denied_actions=list(payload.denied_actions),
    )
    session.add(scope)
    await _flush(session)
    return scope


@router.get("/{project_id}/scopes")
async def list_scopes(
    project_id: str,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    project = await _get_project_via_repo(session, project_id)
    rows = await _list_scopes(session, _attr(project, "id"))
    return {"items": [scope_to_dict(row) for row in rows], "total": len(rows)}


@router.post("/{project_id}/scopes", status_code=status.HTTP_201_CREATED)
async def create_scope(
    project_id: str,
    payload: ScopeCreate,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    project = await _get_project_via_repo(session, project_id)
    scope = await _create_scope_row(session, _attr(project, "id"), payload)
    logger.info(
        "scope.created",
        extra={"project_id": str(project_id), "scope_id": _iso(_attr(scope, "id"))},
    )
    return scope_to_dict(scope)


@router.delete("/{project_id}/scopes/{scope_id}")
async def delete_scope(
    project_id: str,
    scope_id: str,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    project = await _get_project_via_repo(session, project_id)
    Scope = _model("scope", "Scope")
    from sqlalchemy import select

    pid = _attr(project, "id")
    sid = _coerce_uuid(scope_id, field="scope_id")
    result = await session.execute(
        select(Scope).where(Scope.id == sid, Scope.project_id == pid)
    )
    scope = result.scalars().first()
    if scope is None:
        raise NotFoundError(
            f"Scope {scope_id} not found", details={"scope_id": str(scope_id)}
        )

    fact_count = await _count_facts_for_scope(session, sid)
    if fact_count:
        raise ConflictError(
            "Scope is referenced by facts and cannot be deleted",
            details={"scope_id": str(scope_id), "facts": fact_count},
        )

    await _maybe_await(session.delete(scope))
    await _flush(session)
    logger.info(
        "scope.deleted", extra={"project_id": str(project_id), "scope_id": str(scope_id)}
    )
    return {"deleted": True, "scope_id": str(scope_id)}


async def _count_facts_for_scope(session: Any, scope_id: Any) -> int:
    try:
        Fact = _model("fact", "Fact")
    except Exception:  # pragma: no cover - partial builds
        return 0
    from sqlalchemy import func, select

    try:
        result = await session.execute(
            select(func.count()).select_from(Fact).where(Fact.scope_id == scope_id)
        )
        return int(result.scalar() or 0)
    except Exception:  # pragma: no cover - schema drift tolerance
        return 0
