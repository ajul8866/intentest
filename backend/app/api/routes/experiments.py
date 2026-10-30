"""Experiment orchestration endpoints (`/api/v1/experiments`).

Experiments group one project per ``(challenge, arm, repeat)`` so that the
evaluation harness can reproduce the paper's Tables 1/2/3 shaped report.  The
routes stay thin: payload validation happens here, all state changes are
delegated to ``app.services.experiment_service`` when available and otherwise
fall back to the request-scoped :class:`~app.api.deps.Repositories` bundle.

Contract (see ``API.md``)::

    POST   /experiments
    GET    /experiments?status=&limit=&offset=
    GET    /experiments/{experiment_id}
    POST   /experiments/{experiment_id}/start
    POST   /experiments/{experiment_id}/cancel
    GET    /experiments/{experiment_id}/results
    GET    /experiments/{experiment_id}/tasks

Every list response uses the canonical ``{"items": [...], "total": n}``
envelope and failures are raised as :mod:`app.api.errors` types so that the
error envelope (``{"error": {...}}``) is produced by the handlers installed in
``app.main``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Mapping, Sequence

from fastapi import APIRouter, Body, Depends, Query, status

from app.api.deps import (
    Pagination,
    Repositories,
    get_repositories,
    get_session,
    pagination,
)
from app.api.errors import (
    AppError,
    ConflictError,
    ErrorCode,
    NotFoundError,
    ValidationAppError,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])

EXPERIMENT_STATUSES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED")

_MISS = object()


# ---------------------------------------------------------------------------
# Small tolerant helpers (repositories/services may expose either naming style)
# ---------------------------------------------------------------------------


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _try_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``func`` and return ``_MISS`` when the signature does not match."""

    try:
        return await _maybe_await(func(*args, **kwargs))
    except TypeError:
        return _MISS


async def _get_entity(repo: Any, identifier: Any, *, resource: str = "experiment") -> Any:
    for name in ("get", "get_by_id", "fetch", "read", "find", "find_by_id", "find_by_pk"):
        fn = getattr(repo, name, None)
        if not callable(fn):
            continue
        result = await _try_call(fn, identifier)
        if result is not _MISS and result is not None:
            return result
    raise NotFoundError(
        f"{resource} {identifier!r} does not exist",
        details={"id": str(identifier)},
    )


async def _create_entity(repo: Any, payload: Mapping[str, Any]) -> Any:
    for name in ("create", "create_experiment", "add", "insert", "create_from_dict"):
        fn = getattr(repo, name, None)
        if not callable(fn):
            continue
        for attempt in (lambda fn=fn: fn(**dict(payload)), lambda fn=fn: fn(dict(payload))):
            result = await _try_call(attempt)
            if result is not _MISS and result is not None:
                return result
    raise AppError(
        "Experiment repository does not expose a compatible create method",
        code=ErrorCode.INTERNAL,
        details={"repository": type(repo).__name__},
    )


async def _list_entities(repo: Any) -> list[Any]:
    for name in ("list", "list_all", "list_experiments", "all", "find_all", "select_all"):
        fn = getattr(repo, name, None)
        if not callable(fn):
            continue
        result = await _try_call(fn)
        if result is not _MISS:
            if result is None:
                return []
            return list(result)
    return []


async def _update_entity(entity: Any, repo: Any, values: Mapping[str, Any]) -> Any:
    for name in ("update", "patch", "set_status", "save"):
        fn = getattr(repo, name, None)
        if not callable(fn):
            continue
        for attempt in (
            lambda fn=fn: fn(entity, **dict(values)),
            lambda fn=fn: fn(getattr(entity, "id", entity), **dict(values)),
            lambda fn=fn: fn(entity, dict(values)),
        ):
            result = await _try_call(attempt)
            if result is not _MISS:
                return result if result is not None else entity
    # Last resort: mutate the ORM object in place; ``get_session`` commits.
    for key, value in values.items():
        if hasattr(entity, key):
            setattr(entity, key, value)
    return entity


def _service() -> Any | None:
    try:
        return importlib.import_module("app.services.experiment_service")
    except Exception:  # pragma: no cover - service module is optional
        return None


async def _service_call(names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
    module = _service()
    if module is None:
        return _MISS
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            result = await _try_call(fn, *args, **kwargs)
            if result is not _MISS:
                return result
    return _MISS


def _serialise(entity: Any) -> Any:
    if isinstance(entity, Mapping):
        return dict(entity)
    for name in ("to_dict", "as_dict", "model_dump", "dict"):
        fn = getattr(entity, name, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # pragma: no cover - defensive
                continue
    data: dict[str, Any] = {}
    for attr in (
        "id",
        "name",
        "description",
        "status",
        "config",
        "created_at",
        "started_at",
        "finished_at",
        "completed_at",
        "total_tasks",
        "completed_tasks",
        "notes",
        "results",
    ):
        if hasattr(entity, attr):
            value = getattr(entity, attr)
            data[attr] = getattr(value, "value", value)
    return data


def _status_of(value: Any) -> str:
    return str(getattr(value, "value", value) or "").upper()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_experiment(
    payload: Mapping[str, Any] = Body(default_factory=dict, description="Experiment definition."),
    repos: Repositories = Depends(get_repositories),
    session: Any = Depends(get_session),
) -> Any:
    """Create an experiment (one project per challenge x arm x repeat)."""

    data = dict(payload or {})
    if not data.get("name") and not data.get("challenge_id"):
        raise ValidationAppError(
            "An experiment requires at least a name or a challenge_id",
            details={"required": ["name|challenge_id"]},
        )
    if not data.get("name") and data.get("challenge_id"):
        data["name"] = f"experiment-{data['challenge_id']}"

    result = await _service_call(("create_experiment", "create"), data)
    if result is _MISS:
        result = await _create_entity(repos.experiment, data)
    log.info("experiment.created", extra={"experiment_id": str(getattr(result, "id", ""))})
    return _serialise(result)


@router.get("")
async def list_experiments(
    status_filter: str | None = Query(default=None, alias="status", description="Filter by status."),
    page: Pagination = Depends(pagination),
    repos: Repositories = Depends(get_repositories),
    session: Any = Depends(get_session),
) -> dict[str, Any]:
    """List experiments with the canonical pagination envelope."""

    items = await _list_entities(repos.experiment)
    if status_filter:
        wanted = exp_status_placeholder if False else status_filter.upper()  # noqa: F821
        items = [item for item in items if _status_of(getattr(item, "status", "")) == wanted]
    start, stop = page.slice_bounds()
    page_items = items[start:stop]
    return {"items": [_serialise(item) for item in page_items], "total": len(items)}


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    repos: Repositories = Depends(get_repositories),
    session: Any = Depends(get_session),
) -> Any:
    """Fetch a single experiment."""

    experiment = await _get_entity(repos.experiment, experiment_id)
    return _serialise(experiment)


@router.post("/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    repos: Repositories = Depends(get_repositories),
    session: Any = Depends(get_session),
) -> Any:
    """Start an experiment (creates the per-challenge projects)."""

    experiment = await _get_entity(repos.experiment, experiment_id)
    current = _status_of(getattr(experiment, "status", ""))
    if current == "RUNNING":
        raise ConflictError(
            f"Experiment {experiment_id!r} is already running",
            details={"experiment_id": str(experiment_id), "status": current},
        )
    if current in {"COMPLETED", "CANCELLED"}:
        raise ConflictError(
            f"Experiment {experiment_id!r} is terminal",
            details={"experiment_id": str(experiment_id), "status": current},
        )

    result = await _service_call(("start_experiment", "start"), experiment)
    if result is _MISS:
        result = await _update_entity(experiment, repos.experiment, {"status": "RUNNING"})
    log.info("experiment.started", extra={"experiment_id": str(experiment_id)})
    return _serialise(result)


@router.post("/{experiment_id}/cancel")
async def cancel_experiment(
    experiment_id: str,
    repos: Repositories = Depends(get_repositories),
    session: Any = Depends(get_session),
) -> Any:
    """Cancel a running experiment."""

    experiment = await _get_entity(repos.experiment, experiment_id)
    result = await _service_call(("cancel_experiment", "cancel", "abort"), experiment)
    if result is _MISS:
        result = await _update_entity(experiment, repos.experiment, {"status": "CANCELLED"})
    log.info("experiment.cancelled", extra={"experiment_id": str(experiment_id)})
    return _serialise(result)


@router.get("/{experiment_id}/results")
async def experiment_results(
    experiment_id: str,
    repos: Repositories = Depends(get_repositories),
    session: Any = Depends(get_session),
) -> Any:
    """Aggregated per-difficulty results (Table 1/2/3 shaped)."""

    result = await _service_call(("results", "get_results", "aggregate"), experiment_id)
    if result is not _MISS:
        return result

    experiment = await _get_entity(repos.experiment, experiment_id)
    tasks: list[Any] = []
    for name in ("tasks", "list_tasks", "experiment_tasks"):
        attr = getattr(repos.experiment, name, None)
        if callable(attr):
            value = await _try_call(attr, experiment_id)
            if value is not _MISS and value is not None:
                tasks = list(value)
                break
    return {
        "experiment": _serialise(experiment),
        "tasks": [_serialise(task) for task in tasks],
        "total": len(tasks),
    }


@router.get("/{experiment_id}/tasks")
async def list_experiment_tasks(
    experiment_id: str,
    page: Pagination = Depends(pagination),
    repos: Repositories = Depends(get_repositories),
    session: Any = Depends(get_session),
) -> dict[str, Any]:
    """List the ``experiment_task`` rows (one per challenge x arm x repeat)."""

    tasks: list[Any] = []
    for name in ("list_tasks", "tasks", "experiment_tasks"):
        fn = getattr(repos.experiment, name, None)
        if callable(fn):
            value = await _try_call(fn, experiment_id)
            if value is not _MISS and value is not None:
                tasks = list(value)
                break
    start, stop = page.slice_bounds()
    return {"items": [_serialise(task) for task in tasks[start:stop]], "total": len(tasks)}


__all__ = ["router"]
