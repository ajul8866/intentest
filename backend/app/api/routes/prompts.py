"""Expert prompt (hint) endpoints.

The paper's scheduler consumes ``prompt`` rows as the Cairn "Hint" primitive:
a human expert may asynchronously drop a hint into a running project.  The
hint does **not** modify the DAG; it only acts as one of the two wake
conditions for a ``REASON`` task (Algorithm 1, step 7)::

    reason_gate(p) = (facts_since_reason > 0 OR unconsumed_prompt(p))
                     AND no unclaimed EXPLORE globally
                     AND reason_mutex(p) free

Routes (mounted under ``/projects/{project_id}/prompts``):

* ``GET    ""``                 — list hints (consumed filter, pagination)
* ``POST   ""``                 — enqueue a new expert hint (201)
* ``GET    "/pending"``         — unconsumed hints ordered by priority
* ``GET    "/{prompt_id}"``     — fetch one hint
* ``POST   "/{prompt_id}/consume"`` — mark consumed by an intent (idempotent)
* ``DELETE "/{prompt_id}"``     — remove an *unconsumed* hint

Everything else (wake-up, gating, consumption at execution start) lives in
``app.core.scheduler.task_factory`` / ``app.core.scheduler.priority`` and in
``app.db.repositories.prompt_repo``.  This module is deliberately thin: it
validates the payload, resolves the project, and delegates.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable, Optional, Sequence

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field

from app.api import deps
from app.api.errors import ConflictError, NotFoundError, ValidationAppError

__all__ = [
    "router",
    "PromptCreate",
    "prompt_to_dict",
]

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/projects/{project_id}/prompts",
    tags=["prompts"],
)

_MAX_LIMIT = 500
_PRIORITY_RANK = {"critical": 0, "high": 1, "normal": 2, "low": 3}

_MISS = object()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PromptCreate(BaseModel):
    """Body for ``POST /projects/{id}/prompts``.

    ``content`` is copied verbatim into the belief projection handed to the
    next reasoning turn; it is therefore treated as *data* by the worker and
    wrapped in the untrusted-content block before reaching the LLM.
    """

    content: str = Field(..., min_length=1, description="Expert hint text.")
    priority: int = Field(
        0,
        ge=0,
        le=100,
        description="Lower value == more urgent (0 is highest priority).",
    )
    author: Optional[str] = Field(default=None, description="Optional expert id.")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Free-form metadata (never shown to the LLM verbatim)."
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _iso(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


def prompt_to_dict(prompt: Any) -> dict[str, Any]:
    """Tolerant serializer for ``prompt`` rows (ORM / dataclass / mapping)."""

    return {
        "id": _iso(_attr(prompt, "id")),
        "project_id": _iso(_attr(prompt, "project_id")),
        "content": _attr(prompt, "content", default=""),
        "priority": _attr(prompt, "priority", default=0),
        "consumed": bool(_attr(prompt, "consumed", default=False)),
        "consumed_by_intent": _iso(_attr(prompt, "consumed_by_intent", "consumed_by_intent_id")),
        "consumed_at": _iso(_attr(prompt, "consumed_at")),
        "author": _attr(prompt, "author", "created_by"),
        "metadata": _attr(prompt, "metadata", "meta", default=None),
        "created_at": _iso(_attr(prompt, "created_at")),
    }


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _call_first(obj: Any, names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
    for name in names:
        func = getattr(obj, name, None)
        if func is None:
            continue
        try:
            return await _maybe_await(func(*args, **kwargs))
        except TypeError:
            continue
    return _MISS


def _coerce_uuid(value: Any, *, field: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValidationAppError(
            f"{field} must be a UUID", details={field: str(value)}
        ) from exc


async def _load_project(session: Any, project_id: uuid.UUID) -> Any:
    from app.db.models.project import Project  # local import: keep startup light
    from sqlalchemy import select

    result = await session.execute(select(Project).where(Project.id == project_id))
    project = result.scalars().first()
    if project is None:
        raise NotFoundError(
            f"Project {project_id} not found", details={"project_id": str(project_id)}
        )
    return project


async def _get_prompt_or_404(session: Any, project_id: uuid.UUID, prompt_id: Any) -> Any:
    from app.db.models.prompt import Prompt  # local import
    from sqlalchemy import select

    result = await session.execute(
        select(Prompt).where(Prompt.id == prompt_id, Prompt.project_id == project_id)
    )
    prompt = result.scalars().first()
    if prompt is None:
        raise NotFoundError(
            f"Prompt {prompt_id} not found",
            details={"project_id": str(project_id), "prompt_id": str(prompt_id)},
        )
    return prompt


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("", summary="List expert hints for a project")
async def list_prompts(
    project_id: uuid.UUID,
    consumed: Optional[bool] = Query(
        default=None, description="Filter on the consumed flag."
    ),
    limit: int = Query(100, ge=1, le=_MAX_LIMIT),
    offset: int = Query(0, ge=0),
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    from app.db.models.prompt import Prompt  # local import
    from sqlalchemy import func, select

    await _load_project(session, project_id)

    stmt = select(Prompt).where(Prompt.project_id == project_id)
    count_stmt = select(func.count()).select_from(Prompt).where(Prompt.project_id == project_id)
    if consumed is not None:
        stmt = stmt.where(Prompt.consumed.is_(consumed))
        count_stmt = count_stmt.where(Prompt.consumed.is_(consumed))

    total = int((await session.execute(count_stmt)).scalar_one() or 0)
    stmt = (
        stmt.order_by(Prompt.consumed.asc(), Prompt.priority.asc(), Prompt.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return {
        "items": [prompt_to_dict(row) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/pending", summary="Unconsumed hints (priority order)")
async def list_pending_prompts(
    project_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=_MAX_LIMIT),
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    from app.db.models.prompt import Prompt  # local import
    from sqlalchemy import select

    await _load_project(session, project_id)
    stmt = (
        select(Prompt)
        .where(Prompt.project_id == project_id, Prompt.consumed.is_(False))
        .order_by(Prompt.priority.asc(), Prompt.created_at.asc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return {"items": [prompt_to_dict(row) for row in rows], "total": len(rows)}


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Enqueue an expert hint",
)
async def create_prompt(
    project_id: uuid.UUID,
    payload: PromptCreate,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    from app.db.models.prompt import Prompt  # local import

    project = await _load_project(session, project_id)
    project_status = str(_attr(project, "status", default="") or "").upper()
    if project_status in {"COMPLETED", "FAILED", "CANCELLED"}:
        raise ConflictError(
            "Cannot add hints to a finished project",
            details={"project_id": str(project_id), "status": project_status},
        )

    content = payload.content.strip()
    if not content:
        raise ValidationAppError("Prompt content must not be blank")

    row = Prompt(
        project_id=project_id,
        content=content,
        priority=payload.priority,
        consumed=False,
    )
    # Optional columns – set only when the model exposes them.
    for attr, value in (
        ("author", payload.author),
        ("created_by", payload.author),
        ("metadata", payload.metadata),
        ("meta", payload.metadata),
    ):
        if value is not None and hasattr(row, attr):
            try:
                setattr(row, attr, value)
            except Exception:  # pragma: no cover - read-only/hybrid props
                pass

    session.add(row)
    await session.flush()

    log.info(
        "prompt.created",
        extra={
            "project_id": str(project_id),
            "prompt_id": str(_attr(row, "id", default="")),
            "priority": payload.priority,
        },
    )
    return prompt_to_dict(row)


@router.get("/{prompt_id}", summary="Fetch one hint")
async def get_prompt(
    project_id: uuid.UUID,
    prompt_id: uuid.UUID,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    row = await _get_prompt_or_404(session, project_id, prompt_id)
    return prompt_to_dict(row)


@router.post("/{prompt_id}/consume", summary="Mark a hint as consumed")
async def consume_prompt(
    project_id: uuid.UUID,
    prompt_id: uuid.UUID,
    intent_id: Optional[uuid.UUID] = Query(
        default=None, description="Intent that consumed the hint."
    ),
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    row = await _get_prompt_or_404(session, project_id, prompt_id)

    if bool(_attr(row, "consumed", default=False)):
        return prompt_to_dict(row)

    row.consumed = True
    for attr in ("consumed_by_intent", "consumed_by_intent_id"):
        if hasattr(row, attr):
            try:
                setattr(row, attr, intent_id)
            except Exception:  # pragma: no cover
                pass
    if hasattr(row, "consumed_at"):
        try:
            from datetime import datetime, timezone

            row.consumed_at = datetime.now(timezone.utc)
        except Exception:  # pragma: no cover
            pass

    await session.flush()
    log.info(
        "prompt.consumed",
        extra={
            "project_id": str(project_id),
            "prompt_id": str(prompt_id),
            "intent_id": str(intent_id) if intent_id else None,
        },
    )
    return prompt_to_dict(row)


@router.delete("/{prompt_id}", status_code=status.HTTP_200_OK, summary="Delete an unconsumed hint")
async def delete_prompt(
    project_id: uuid.UUID,
    prompt_id: uuid.UUID,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    row = await _get_prompt_or_404(session, project_id, prompt_id)
    if bool(_attr(row, "consumed", default=False)):
        raise ConflictError(
            "Consumed hints are part of the audit trail and cannot be deleted",
            details={"prompt_id": str(prompt_id)},
        )
    await session.delete(row)
    await session.flush()
    log.info(
        "prompt.deleted",
        extra={"project_id": str(project_id), "prompt_id": str(prompt_id)},
    )
    return {"deleted": True, "prompt_id": str(prompt_id)}


@router.get("/stats/summary", summary="Hint counters for a project")
async def prompt_stats(
    project_id: uuid.UUID,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    from app.db.models.prompt import Prompt  # local import
    from sqlalchemy import func, select

    await _load_project(session, project_id)
    stmt = (
        select(Prompt.consumed, func.count())
        .where(Prompt.project_id == project_id)
        .group_by(Prompt.consumed)
    )
    rows = (await session.execute(stmt)).all()
    consumed = 0
    pending = 0
    for flag, count in rows:
        if flag:
            consumed += int(count)
        else:
            pending += int(count)
    return {"total": consumed + pending, "consumed": consumed, "pending": pending}
