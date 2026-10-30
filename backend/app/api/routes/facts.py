"""Fact routes.

Read/append access to the immutable fact store (paper Definition 1: ``F``).

Facts are append-only: this module deliberately exposes **no** update or delete
endpoint.  Corrections are expressed by appending a new fact that carries
``supersedes_fact_id`` -- invariant **I1** of ``ARCHITECTURE.md``.

Endpoints (see ``API.md``)
--------------------------
``GET    /projects/{project_id}/facts``              list + filter + paginate
``POST   /projects/{project_id}/facts``              append a fact (expert/API)
``GET    /projects/{project_id}/facts/{fact_id}``    single fact
``GET    /projects/{project_id}/facts/{fact_id}/evidence``
``GET    /projects/{project_id}/facts/{fact_id}/lineage``
``GET    /projects/{project_id}/facts/kinds``        counts per fact kind
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable, Mapping, Optional, Sequence

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field

from app.api import deps
from app.api.errors import NotFoundError, ValidationAppError
from app.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/projects/{project_id}/facts", tags=["facts"])

_MAX_LIMIT = 500


# --------------------------------------------------------------------------- #
# Schemas (local, tolerant: the canonical schemas live in app.schemas.fact but
# we only rely on loose dicts so that route wiring never breaks).
# --------------------------------------------------------------------------- #
class FactCreate(BaseModel):
    """Body for appending a fact directly (outside the worker pipeline)."""

    kind: str = Field(..., description="START|GOAL|HOST|PORT|SERVICE|ENDPOINT|PARAM|CREDENTIAL|VULNERABILITY|SOURCE_LEAK|FILE|SESSION|FLAG|NOTE")
    attributes: dict[str, Any] = Field(default_factory=dict)
    value: Any = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    verified: bool = False
    verification_method: Optional[str] = None
    evidence: Optional[str] = None
    mitre_technique: Optional[str] = None
    supersedes_fact_id: Optional[str] = None
    derived_from_intent_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first present attribute/key from ``obj``."""
    for name in names:
        if obj is None:
            continue
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _iso(value: Any) -> Any:
    """Best-effort JSON-safe conversion for datetimes/UUIDs/enums."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if hasattr(value, "value"):  # Enum
        return value.value
    return value


def fact_to_dict(fact: Any) -> dict[str, Any]:
    """Serialize a ``Fact`` ORM row (or mapping) into the API shape."""
    if fact is None:
        return {}
    get = lambda *names, default=None: _attr(fact, *names, default=default)  # noqa: E731
    return {
        "id": _iso(get("id")),
        "project_id": _iso(get("project_id")),
        "kind": _iso(get("kind")),
        "fact_key": get("fact_key"),
        "attributes": get("attributes") or {},
        "value": get("value"),
        "confidence": get("confidence"),
        "verified": bool(get("verified", default=False)),
        "verification_method": _iso(get("verification_method")),
        "evidence": get("evidence"),
        "evidence_hash": get("evidence_hash"),
        "derived_from_intent_id": _iso(get("derived_from_intent_id")),
        "supersedes_fact_id": _iso(get("supersedes_fact_id")),
        "content_hash": get("content_hash"),
        "mitre_technique": get("mitre_technique"),
        "scope_id": _iso(get("scope_id")),
        "created_at": _iso(get("created_at")),
    }


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _call_first(obj: Any, names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
    """Invoke the first existing coroutine/function on ``obj``."""
    for name in names:
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        try:
            return await _maybe_await(fn(*args, **kwargs))
        except TypeError:
            continue
    return None


async def _load_project(session: Any, project_id: uuid.UUID) -> Any:
    """Fetch the project row or raise a typed 404."""
    from app.db.models.project import Project
    from sqlalchemy import select

    result = await session.execute(select(Project).where(Project.id == project_id))
    project = result.scalars().first()
    if project is None:
        raise NotFoundError(f"Project {project_id} not found", details={"project_id": str(project_id)})
    return project


def _coerce_uuid(value: str, *, field: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValidationAppError(f"Invalid {field}", details={field: str(value)}) from exc


async def _query_facts(
    session: Any,
    project_id: uuid.UUID,
    *,
    kind: Optional[str] = None,
    verified: Optional[bool] = None,
    min_confidence: Optional[float] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[Any], int]:
    """Query facts directly through SQLAlchemy (works without a FactRepository)."""
    from sqlalchemy import func, or_, select

    from app.db.models.fact import Fact

    stmt = select(Fact).where(Fact.project_id == project_id)
    if kind:
        stmt = stmt.where(Fact.kind == kind.upper())
    if verified is not None:
        stmt = stmt.where(Fact.verified.is_(verified))
    if min_confidence is not None:
        stmt = stmt.where(Fact.confidence >= min_confidence)
    if search:
        pattern = f"%{search}%"
        stmt = stmt.where(
            or_(
                Fact.mitre_technique.ilike(pattern),
                Fact.evidence.ilike(pattern),
                func.cast(Fact.attributes, func.TEXT).ilike(pattern) if hasattr(func, "TEXT") else Fact.evidence.ilike(pattern),
            )
        )

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = int((await session.execute(count_stmt)).scalar_one() or 0)

    stmt = stmt.order_by(Fact.created_at.asc(), Fact.id.asc()).limit(limit).offset(offset)
    rows = (await session.execute(stmt)).scalars().all()
    return list(rows), total


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("", summary="List facts of a project")
async def list_facts(
    project_id: uuid.UUID,
    kind: Optional[str] = Query(None, description="Filter by fact kind (HOST, PORT, VULNERABILITY, ...)"),
    verified: Optional[bool] = Query(None, description="Filter by verification status"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    search: Optional[str] = Query(None, description="Substring match on evidence / attributes"),
    limit: int = Query(100, ge=1, le=_MAX_LIMIT),
    offset: int = Query(0, ge=0),
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    project_uuid = _coerce_uuid(project_id, field="project_id")
    await _load_project(session, project_uuid)

    rows, total = await _query_facts(
        session,
        project_uuid,
        kind=kind,
        verified=verified,
        min_confidence=min_confidence,
        search=search,
        limit=limit,
        offset=offset,
    )
    return {
        "items": [fact_to_dict(row) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/kinds", summary="Fact counts grouped by kind")
async def fact_kinds(
    project_id: uuid.UUID,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    project_uuid = _coerce_uuid(project_id, field="project_id")
    await _load_project(session, project_uuid)

    from sqlalchemy import func, select

    from app.db.models.fact import Fact

    stmt = select(Fact.kind, func.count()).where(Fact.project_id == project_uuid).group_by(Fact.kind)
    rows = (await session.execute(stmt)).all()
    counts = {_iso(getattr(kind, "value", kind)): int(count) for kind, count in rows}
    verified_total = int(
        (
            await session.execute(
                select(func.count()).select_from(Fact).where(Fact.project_id == project_uuid, Fact.verified.is_(True))
            )
        ).scalar_one()
        or 0
    )
    return {"counts": counts, "total": sum(counts.values()), "verified": verified_total}


@router.post("", status_code=status.HTTP_201_CREATED, summary="Append a fact")
async def create_fact(
    project_id: uuid.UUID,
    payload: FactCreate,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Append a fact to the immutable store.

    Uses the ``FactRepository`` when available (it owns the canonical fact key,
    the ``ON CONFLICT DO NOTHING RETURNING`` idempotency and the immutability
    guard) and falls back to raw SQLAlchemy otherwise.
    """
    project_uuid = _coerce_uuid(project_id, field="project_id")
    await _load_project(session, project_uuid)

    data = payload.model_dump()
    data["project_id"] = project_uuid
    for key in ("supersedes_fact_id", "derived_from_intent_id"):
        if data.get(key):
            data[key] = _coerce_uuid(data[key], field=key)

    try:
        from app.db.repositories.fact_repo import FactRepository

        repo = FactRepository(session)
        created = await _call_first(
            repo,
            ("create_if_absent", "create_fact", "insert_if_absent", "create", "insert"),
            **data,
        )
    except ImportError:  # pragma: no cover - repository is a planned file
        created = None

    if created is None:
        from app.db.models.fact import Fact

        fact = Fact(**{k: v for k, v in data.items() if hasattr(Fact, k)})
        session.add(fact)
        await session.flush()
        created = fact

    logger.info(
        "fact.created_via_api",
        extra={"project_id": str(project_uuid), "kind": data.get("kind")},
    )
    return fact_to_dict(created)


@router.get("/{fact_id}", summary="Get a single fact")
async def get_fact(
    project_id: uuid.UUID,
    fact_id: uuid.UUID,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    project_uuid = _coerce_uuid(project_id, field="project_id")
    fact_uuid = _coerce_uuid(fact_id, field="fact_id")

    from sqlalchemy import select

    from app.db.models.fact import Fact

    row = (
        await session.execute(select(Fact).where(Fact.id == fact_uuid, Fact.project_id == project_uuid))
    ).scalars().first()
    if row is None:
        raise NotFoundError(f"Fact {fact_uuid} not found", details={"fact_id": str(fact_uuid)})
    return fact_to_dict(row)


@router.get("/{fact_id}/evidence", summary="Evidence payload of a fact")
async def get_fact_evidence(
    project_id: uuid.UUID,
    fact_id: uuid.UUID,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    fact = await get_fact(project_id, fact_id, session)
    return {
        "fact_id": fact["id"],
        "evidence": fact.get("evidence"),
        "evidence_hash": fact.get("evidence_hash"),
        "content_hash": fact.get("content_hash"),
        "fact_key": fact.get("fact_key"),
        "verification_method": fact.get("verification_method"),
        "verified": fact.get("verified"),
    }


@router.get("/{fact_id}/lineage", summary="Predecessor chain of a fact (E_i->f + supersedes)")
async def get_fact_lineage(
    project_id: uuid.UUID,
    fact_id: uuid.UUID,
    session: Any = Depends(deps.get_session),
) -> dict[str, Any]:
    """Walk ``derived_from_intent_id`` / ``intent_source`` backwards.

    Produces the alternating fact -> intent -> fact path of Definition 1, which
    the Graph Explorer renders as ``path-to-goal``.
    """
    project_uuid = _coerce_uuid(project_id, field="project_id")
    fact_uuid = _coerce_uuid(fact_id, field="fact_id")

    from sqlalchemy import select

    from app.db.models.fact import Fact
    from app.db.models.intent import Intent
    from app.db.models.intent_source import IntentSource

    facts: dict[str, Any] = {}
    intents: dict[str, Any] = {}
    edges: list[dict[str, Any]] = []
    frontier: list[uuid.UUID] = [fact_uuid]
    seen: set[uuid.UUID] = set()

    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)

        fact = (
            await session.execute(select(Fact).where(Fact.id == current, Fact.project_id == project_uuid))
        ).scalars().first()
        if fact is None:
            continue
        facts[str(fact.id)] = fact_to_dict(fact)

        producer_id = _attr(fact, "derived_from_intent_id")
        if producer_id is None:
            continue
        intent = (
            await session.execute(select(Intent).where(Intent.id == producer_id))
        ).scalars().first()
        if intent is None:
            continue
        intents[str(intent.id)] = {
            "id": str(intent.id),
            "action_type": _iso(_attr(intent, "action_type")),
            "objective": _attr(intent, "objective"),
            "state": _iso(_attr(intent, "state")),
            "tactic": _attr(intent, "tactic"),
            "target_entity": _attr(intent, "target_entity"),
        }
        edges.append(
            {"id": f"eif:{intent.id}->{fact.id}", "source": f"i:{intent.id}", "target": f"f:{fact.id}", "relation": "PRODUCES"}
        )

        sources: Iterable[Any] = (
            await session.execute(select(IntentSource).where(IntentSource.intent_id == intent.id))
        ).scalars().all()
        for source in sources:
            predecessor_id = _attr(source, "fact_id")
            if predecessor_id is None:
                continue
            edges.append(
                {
                    "id": f"efi:{predecessor_id}->{intent.id}",
                    "source": f"f:{predecessor_id}",
                    "target": f"i:{intent.id}",
                    "relation": "TRIGGERS",
                    "role": _iso(_attr(source, "role")),
                    "weight": _attr(source, "weight"),
                }
            )
            frontier.append(predecessor_id)

    return {
        "fact_id": str(fact_uuid),
        "facts": list(facts.values()),
        "intents": list(intents.values()),
        "edges": edges,
    }
