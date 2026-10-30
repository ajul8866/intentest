"""Report endpoints.

Exposes the rendered episode report for a project in Markdown or JSON form
(``API.md``: ``GET /projects/{id}/report?format=md|json``), plus a small set
of focused sub-resources used by the frontend *Report* tab:

``GET /projects/{project_id}/report``              -> full report (md|json)
``GET /projects/{project_id}/report/markdown``     -> Markdown body
``GET /projects/{project_id}/report/json``         -> structured document
``GET /projects/{project_id}/report/summary``      -> small headline block
``GET /projects/{project_id}/report/findings``     -> findings table rows
``GET /projects/{project_id}/report/metrics``      -> drift / performance metrics

The heavy lifting (querying facts/intents, computing metrics, formatting the
Markdown) lives in :mod:`app.services.report_service`.  This module stays
thin: validate the requested format, delegate, and fall back to a minimal
read-only projection built directly from the session when the service is not
importable (partial builds / unit tests).

No state is mutated here.  Reports are a pure projection of the immutable
Fact-Intent DAG, so repeated calls are side-effect free.
"""

from __future__ import annotations

import importlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

from app.api import deps
from app.api.errors import NotFoundError, ValidationAppError
from app.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/projects/{project_id}/report", tags=["reports"])

__all__ = ["router", "REPORT_FORMATS", "DEFAULT_FORMAT"]

REPORT_FORMATS: tuple[str, ...] = ("md", "json")
DEFAULT_FORMAT = "md"

_MISS = object()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Tolerant attribute access across ORM rows, dataclasses and mappings."""
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


def _iso(value: Any) -> Any:
    """JSON-safe rendering of UUID / datetime / enum values."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int)):
        return enum_value
    return str(value)


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _coerce_uuid(value: Any, *, field: str = "project_id") -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise ValidationAppError(
            f"{field} must be a UUID", details={"field": field, "value": str(value)}
        ) from exc


def _normalise_format(fmt: str | None) -> str:
    value = (fmt or DEFAULT_FORMAT).strip().lower()
    aliases = {
        "markdown": "md",
        "text": "md",
        "txt": "md",
        "mdown": "md",
        "application/json": "json",
        "text/markdown": "md",
    }
    value = aliases.get(value, value)
    if value not in REPORT_FORMATS:
        raise ValidationAppError(
            f"Unsupported report format {fmt!r}",
            details={"allowed": list(REPORT_FORMATS)},
        )
    return value


def _service() -> Any | None:
    """Import :mod:`app.services.report_service` lazily (may be absent)."""
    try:
        return importlib.import_module("app.services.report_service")
    except Exception as exc:  # pragma: no cover - partial builds
        logger.debug("report_service_unavailable", extra={"error": str(exc)})
        return None


async def _service_call(names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
    """Invoke the first matching coroutine/function on the report service."""
    module = _service()
    if module is None:
        return _MISS
    for name in names:
        func = getattr(module, name, None)
        if not callable(func):
            continue
        try:
            return await _maybe_await(func(*args, **kwargs))
        except TypeError:
            continue
    # module-level ``ReportService`` class fallback
    cls = getattr(module, "ReportService", None)
    if callable(cls):
        for name in names:
            func = getattr(cls, name, None)
            if not callable(func):
                continue
            try:
                return await _maybe_await(func(*args, **kwargs))
            except TypeError:
                continue
    return _MISS


async def _load_project(session: Any, project_id: uuid.UUID) -> Any:
    from sqlalchemy import select

    from app.db.models.project import Project

    result = await session.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project is None:
        raise NotFoundError(
            f"Project {project_id} not found", details={"project_id": str(project_id)}
        )
    return project


# ---------------------------------------------------------------------------
# fallback projection (used only when report_service is unavailable)
# ---------------------------------------------------------------------------
_FACT_KINDS = (
    "START",
    "GOAL",
    "HOST",
    "PORT",
    "SERVICE",
    "ENDPOINT",
    "PARAM",
    "CREDENTIAL",
    "VULNERABILITY",
    "SOURCE_LEAK",
    "FILE",
    "SESSION",
    "FLAG",
    "NOTE",
)


async def _fallback_document(session: Any, project: Any) -> dict[str, Any]:
    """Minimal report built directly from the DAG when the service is absent."""
    from sqlalchemy import func, select

    from app.db.models.fact import Fact
    from app.db.models.intent import Intent

    project_id = _attr(project, "id")

    fact_rows = (
        await session.execute(
            select(Fact.kind, func.count()).where(Fact.project_id == project_id).group_by(Fact.kind)
        )
    ).all()
    facts_by_kind = {str(kind): int(count) for kind, count in fact_rows}
    facts_total = sum(facts_by_kind.values())

    verified = (
        await session.execute(
            select(func.count()).select_from(Fact).where(
                Fact.project_id == project_id, Fact.verified.is_(True)
            )
        )
    ).scalar_one()

    intent_rows = (
        await session.execute(
            select(Intent.state, func.count()).where(Intent.project_id == project_id).group_by(Intent.state)
        )
    ).all()
    intents_by_state = {str(state): int(count) for state, count in intent_rows}
    intents_total = sum(intents_by_state.values())

    flags = (
        await session.execute(
            select(Fact)
            .where(Fact.project_id == project_id, Fact.kind == "FLAG")
            .order_by(Fact.created_at.asc())
        )
    ).scalars().all()

    findings = [
        {
            "id": str(_attr(fact, "id")),
            "kind": str(_attr(fact, "kind") or ""),
            "value": _attr(fact, "value"),
            "attributes": _attr(fact, "attributes", default={}) or {},
            "confidence": float(_attr(fact, "confidence", default=0.0) or 0.0),
            "verified": bool(_attr(fact, "verified", default=False)),
            "mitre_technique": _attr(fact, "mitre_technique"),
            "created_at": _iso(_attr(fact, "created_at")),
        }
        for fact in flags
    ]

    status = _attr(project, "status")
    status = getattr(status, "value", status)
    termination_reason = _attr(project, "termination_reason")
    termination_reason = getattr(termination_reason, "value", termination_reason)

    return {
        "project": {
            "id": str(project_id),
            "name": _attr(project, "name"),
            "status": status,
            "phase": _iso(_attr(project, "phase")),
            "goal_text": _attr(project, "goal_text"),
            "rounds_used": int(_attr(project, "rounds_used", default=0) or 0),
            "max_rounds": int(_attr(project, "max_rounds", default=0) or 0),
            "termination_reason": termination_reason,
            "created_at": _iso(_attr(project, "created_at")),
            "updated_at": _iso(_attr(project, "updated_at")),
        },
        "summary": {
            "success": str(status or "").upper() == "COMPLETED"
            and str(termination_reason or "").upper() == "GOAL",
            "facts_total": facts_total,
            "facts_verified": int(verified or 0),
            "intents_total": intents_total,
            "rounds_used": int(_attr(project, "rounds_used", default=0) or 0),
            "termination_reason": termination_reason,
        },
        "facts_by_kind": {kind: facts_by_kind.get(kind, 0) for kind in _FACT_KINDS},
        "intents_by_state": intents_by_state,
        "findings": findings,
        "metrics": {},
        "generated_by": "fallback_projection",
    }


def _render_markdown(document: Mapping[str, Any]) -> str:
    project = document.get("project") or {}
    summary = document.get("summary") or {}
    lines: list[str] = []
    lines.append(f"# Report — {project.get('name') or project.get('id')}")
    lines.append("")
    lines.append(f"- **Project id:** `{project.get('id')}`")
    lines.append(f"- **Status:** {project.get('status')}")
    lines.append(f"- **Phase:** {project.get('phase')}")
    lines.append(f"- **Rounds:** {summary.get('rounds_used', project.get('rounds_used'))}"
                 f" / {project.get('max_rounds')}")
    lines.append(f"- **Termination:** {summary.get('termination_reason')}")
    lines.append(f"- **Goal:** {project.get('goal_text')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for key in ("success", "facts_total", "facts_verified", "intents_total", "rounds_used"):
        if key in summary:
            lines.append(f"| {key} | {summary[key]} |")
    lines.append("")

    facts_by_kind = document.get("facts_by_kind") or {}
    if facts_by_kind:
        lines.append("## Facts by kind")
        lines.append("")
        lines.append("| kind | count |")
        lines.append("| --- | --- |")
        for kind, count in sorted(facts_by_kind.items()):
            lines.append(f"| {kind} | {count} |")
        lines.append("")

    intents_by_state = document.get("intents_by_state") or {}
    if intents_by_state:
        lines.append("## Intents by state")
        lines.append("")
        lines.append("| state | count |")
        lines.append("| --- | --- |")
        for state, count in sorted(intents_by_state.items()):
            lines.append(f"| {state} | {count} |")
        lines.append("")

    findings = document.get("findings") or []
    lines.append("## Findings")
    lines.append("")
    if not findings:
        lines.append("_No findings recorded._")
    else:
        lines.append("| kind | value | confidence | verified | MITRE |")
        lines.append("| --- | --- | --- | --- | --- |")
        for finding in findings:
            value = str(finding.get("value"))
            if len(value) > 120:
                value = value[:117] + "..."
            lines.append(
                f"| {finding.get('kind')} | {value} | "
                f"{finding.get('confidence')} | {finding.get('verified')} | "
                f"{finding.get('mitre_technique') or ''} |"
            )
    lines.append("")

    metrics = document.get("metrics") or {}
    if metrics:
        lines.append("## Drift metrics")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for key, value in sorted(metrics.items()):
            lines.append(f"| {key} | {value} |")
        lines.append("")

    lines.append(
        "_Generated by Intentest. Numbers describe this locally executed episode only; "
        "they are not the paper's Table 1/2/3 results._"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
@router.get("", summary="Rendered project report")
async def get_report(
    project_id: str,
    format: str = Query(default=DEFAULT_FORMAT, description="md|json"),
    refresh: bool = Query(default=False, description="Bypass any cached rendering."),
    session: Any = Depends(deps.get_session),
) -> Any:
    """Return the episode report as Markdown (``text/markdown``) or JSON."""
    project_uuid = _coerce_uuid(project_id)
    fmt = _normalise_format(format)

    document = await _service_call(
        ("generate_report", "build_report", "report", "render_report", "generate"),
        session,
        project_uuid,
        format=fmt,
        refresh=refresh,
    )

    if document is _MISS or document is None:
        project = await _load_project(session, project_uuid)
        document = await _fallback_document(session, project)

    if fmt == "md":
        if isinstance(document, str):
            return PlainTextResponse(document, media_type="text/markdown; charset=utf-8")
        markdown = document.get("markdown") if isinstance(document, Mapping) else None
        if not isinstance(markdown, str):
            markdown = _render_markdown(document if isinstance(document, Mapping) else {})
        return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8")

    if isinstance(document, str):
        return {"project_id": str(project_uuid), "format": "md", "markdown": document}
    return {"project_id": str(project_uuid), "format": "json", "report": document}


@router.get("/markdown", summary="Report as Markdown")
async def get_report_markdown(
    project_id: str, session: Any = Depends(deps.get_session)
) -> Any:
    return await get_report(project_id, format="md", refresh=False, session=session)


@router.get("/json", summary="Report as JSON")
async def get_report_json(
    project_id: str, session: Any = Depends(deps.get_session)
) -> Any:
    return await get_report(project_id, format="json", refresh=False, session=session)


@router.get("/summary", summary="Headline report block")
async def get_report_summary(
    project_id: str, session: Any = Depends(deps.get_session)
) -> dict[str, Any]:
    project_uuid = _coerce_uuid(project_id)
    summary = await _service_call(
        ("summary", "get_summary", "build_summary"), session, project_uuid
    )
    if summary is not _MISS and summary is not None:
        return {"project_id": str(project_uuid), "summary": summary}

    project = await _load_project(session, project_uuid)
    document = await _fallback_document(session, project)
    return {
        "project_id": str(project_uuid),
        "project": document["project"],
        "summary": document["summary"],
    }


@router.get("/findings", summary="Findings rows")
async def get_report_findings(
    project_id: str, session: Any = Depends(deps.get_session)
) -> dict[str, Any]:
    project_uuid = _coerce_uuid(project_id)
    findings = await _service_call(
        ("findings", "get_findings", "collect_findings"), session, project_uuid
    )
    if findings is not _MISS and findings is not None:
        items = list(findings) if not isinstance(findings, Mapping) else list(
            findings.get("items", [])
        )
        return {"project_id": str(project_uuid), "items": items, "total": len(items)}

    project = await _load_project(session, project_uuid)
    document = await _fallback_document(session, project)
    items = document["findings"]
    return {"project_id": str(project_uuid), "items": items, "total": len(items)}


@router.get("/metrics", summary="Drift and performance metrics")
async def get_report_metrics(
    project_id: str, session: Any = Depends(deps.get_session)
) -> dict[str, Any]:
    project_uuid = _coerce_uuid(project_id)
    metrics = await _service_call(
        ("metrics", "get_metrics", "compute_metrics", "drift_metrics"),
        session,
        project_uuid,
    )
    if metrics is not _MISS and metrics is not None:
        return {"project_id": str(project_uuid), "metrics": metrics}

    project = await _load_project(session, project_uuid)
    document = await _fallback_document(session, project)
    return {
        "project_id": str(project_uuid),
        "metrics": document.get("metrics") or {},
        "summary": document.get("summary") or {},
    }


@router.get("/download", summary="Download the Markdown report")
async def download_report(
    project_id: str, session: Any = Depends(deps.get_session)
) -> Any:
    """Same bytes as ``/report?format=md`` but with a download disposition."""
    response = await get_report(project_id, format="md", refresh=False, session=session)
    filename = f"intentest-report-{project_id}.md"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


# JSON export helper kept importable for the frontend typed client generator.
def report_to_json(document: Mapping[str, Any]) -> str:
    """Stable, sorted JSON rendering of a report document."""
    return json.dumps(document, sort_keys=True, default=_iso, indent=2)
