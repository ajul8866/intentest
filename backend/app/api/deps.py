"""FastAPI dependency providers for the Intentest HTTP/WebSocket API.

This module is the *only* place where route modules obtain infrastructure
objects.  Routes stay thin (validate → delegate to ``app.services`` → return a
schema); dependencies are intentionally small, overridable and import-light so
that unit tests can substitute fakes with ``app.dependency_overrides`` without
a running Postgres, Docker daemon or LLM endpoint.

Exports
-------
``Pagination`` / ``pagination``
    ``limit``/``offset`` query parameters with the bounds documented in
    ``API.md`` (``limit ∈ [1, 500]``, ``offset ≥ 0``).
``get_settings``
    Cached :class:`app.config.Settings` instance.
``get_session`` / ``get_db``
    Async SQLAlchemy ``AsyncSession`` dependency (one session per request,
    rolled back on error, always closed).
``SessionFactory`` / ``session_factory``
    A callable returning a *fresh* session for background work started from a
    request (WebSocket listeners, report generation).
``get_repositories``
    Bag of repositories bound to the request session.
``get_project_repo``, ``get_fact_repo``, ``get_intent_repo``,
``get_intent_source_repo``, ``get_task_run_repo``, ``get_worker_repo``,
``get_prompt_repo``, ``get_pattern_repo``, ``get_experiment_repo``
    Individual repository dependencies.
``get_llm_client`` / ``get_llm_capabilities``
    Shared OpenAI-compatible client plus its startup capability probe.
``get_embedding_provider``
    Embedding provider used by S1 and the intent bridge, with the documented
    fallback chain (API → sentence-transformers → deterministic hash).
``get_hub``
    The in-process WebSocket hub singleton.
``get_request_id``
    Correlation id (``X-Request-ID`` header or a generated UUID4).
``require_project`` / ``require_project_status``
    Convenience helpers that resolve a project by id and 404 otherwise.
``get_scope_guard``
    Scope guard bound to a project's scopes.
``get_services``
    Bag of service objects (project / fact / intent / retrieval / report).

Design notes
------------
* ``app.api`` deliberately performs no eager imports (see
  ``app/api/__init__.py``); this module therefore imports concrete
  implementations lazily inside the provider bodies where the import is heavy
  or optional (docker SDK, embedding backends, sentence-transformers).
* Every dependency that touches the database yields an ``AsyncSession`` and is
  the sole place ``commit``/``rollback`` is decided: services ``flush`` only.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Optional, Sequence

from fastapi import Depends, Header, Query, Request

from app.api.errors import NotFoundError, SandboxUnavailableError
from app.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "Pagination",
    "pagination",
    "get_settings",
    "get_session",
    "get_db",
    "SessionFactory",
    "session_factory",
    "Repositories",
    "get_repositories",
    "get_project_repo",
    "get_fact_repo",
    "get_intent_repo",
    "get_intent_source_repo",
    "get_task_run_repo",
    "get_worker_repo",
    "get_prompt_repo",
    "get_pattern_repo",
    "get_experiment_repo",
    "get_llm_client",
    "get_llm_capabilities",
    "get_embedding_provider",
    "get_hub",
    "get_request_id",
    "get_scope_guard",
    "require_project",
    "require_project_status",
    "get_services",
    "Services",
    "MAX_LIMIT",
    "DEFAULT_LIMIT",
]

# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

DEFAULT_LIMIT: int = 50
MAX_LIMIT: int = 500


@dataclass(frozen=True, slots=True)
class Pagination:
    """Resolved ``limit``/``offset`` pair for list endpoints."""

    limit: int = DEFAULT_LIMIT
    offset: int = 0

    @property
    def as_dict(self) -> dict[str, int]:
        return {"limit": self.limit, "offset": self.offset}

    def slice_bounds(self) -> tuple[int, int]:
        """Return ``(offset, offset + limit)`` for slicing in-memory lists."""
        return self.offset, self.offset + self.limit


def pagination(
    limit: int = Query(
        DEFAULT_LIMIT,
        ge=1,
        le=MAX_LIMIT,
        description=f"Maximum number of items to return (1..{MAX_LIMIT}).",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Number of items to skip before collecting the result set.",
    ),
) -> Pagination:
    """Dependency: validated pagination parameters.

    ``API.md`` requires every list endpoint to accept ``limit``/``offset`` and
    to answer with ``{"items": [...], "total": <int>}``.
    """
    return Pagination(limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

_SETTINGS: Any = None


def get_settings() -> Any:
    """Dependency: the process-wide :class:`app.config.Settings` singleton.

    ``app.config`` already caches via ``lru_cache``; the local memo avoids a
    repeated import cost on the hot path and lets tests override the dependency
    with a lightweight stub.
    """
    global _SETTINGS
    if _SETTINGS is None:
        from app.config import get_settings as _get_settings

        _SETTINGS = _get_settings()
    return _SETTINGS


# ---------------------------------------------------------------------------
# Database session
# ---------------------------------------------------------------------------

_ENGINE_SESSION_FACTORY: Any = None


def _engine() -> Any:
    """Return the lazily-created global async engine/session factory."""
    global _ENGINE_SESSION_FACTORY
    if _ENGINE_SESSION_FACTORY is None:
        from app.db import get_session_factory

        _ENGINE_SESSION_FACTORY = get_session_factory()
    return _ENGINE_SESSION_FACTORY


class SessionFactory:
    """Callable wrapper that yields ``AsyncSession`` objects on demand.

    Background tasks (WebSocket listeners, report rendering, the lab runner
    when it drives the API in-process) need sessions outside the request
    lifecycle; they receive this object instead of a live session.
    """

    def __init__(self, factory: Any | None = None) -> None:
        self._factory = factory

    def _resolve(self) -> Any:
        if self._factory is not None:
            return self._factory
        return _engine()

    def __call__(self) -> Any:
        factory = self._resolve()
        if callable(factory):
            return factory()
        raise RuntimeError("No async session factory configured (see app.db).")

    def session(self) -> Any:
        """Alias for ``__call__`` used by services that take a factory."""
        return self()


def session_factory() -> SessionFactory:
    """Dependency: a factory producing *new* async sessions."""
    return SessionFactory()


async def get_session(
    settings: Any = Depends(get_settings),
) -> AsyncIterator[Any]:
    """Dependency: request-scoped ``AsyncSession``.

    Commits when the handler returns normally, rolls back on exception and
    always closes the session.  Services and repositories only ``flush``, so
    a single ``commit`` per request keeps the DAG write set atomic from the
    client's point of view (the scheduler/worker use their own transactions).
    """
    factory = _engine()
    session = factory() if callable(factory) else factory
    try:
        yield session
        await session.commit()
    except Exception:
        try:
            await session.rollback()
        except Exception:  # pragma: no cover - defensive
            logger.warning("api.session_rollback_failed", exc_info=True)
        raise
    finally:
        await session.close()


# Backwards/forwards compatible alias (routes may import either name).
get_db = get_session


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Repositories:
    """Bag of repositories bound to the current request session."""

    project: Any
    fact: Any
    intent: Any
    intent_source: Any
    task_run: Any
    worker: Any
    prompt: Any
    pattern: Any
    experiment: Any

    def mapping(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "fact": self.fact,
            "intent": self.intent,
            "intent_source": self.intent_source,
            "task_run": self.task_run,
            "worker": self.worker,
            "prompt": self.prompt,
            "pattern": self.pattern,
            "experiment": self.experiment,
        }


def _repo(module: str, class_name: str, session: Any) -> Any:
    import importlib

    mod = importlib.import_module(module)
    cls = getattr(mod, class_name)
    try:
        return cls(session)
    except TypeError:  # pragma: no cover - repository without ctor arg
        return cls()


def get_project_repo(session: Any = Depends(get_session)) -> Any:
    from app.db.repositories.project_repo import ProjectRepository

    return ProjectRepository(session)


def get_fact_repo(session: Any = Depends(get_session)) -> Any:
    from app.db.repositories.fact_repo import FactRepository

    return FactRepository(session)


def get_intent_repo(session: Any = Depends(get_session)) -> Any:
    from app.db.repositories.intent_repo import IntentRepository

    return IntentRepository(session)


def get_intent_source_repo(session: Any = Depends(get_session)) -> Any:
    from app.db.repositories.intent_source_repo import IntentSourceRepository

    return IntentSourceRepository(session)


def get_task_run_repo(session: Any = Depends(get_session)) -> Any:
    from app.db.repositories.task_run_repo import TaskRunRepository

    return TaskRunRepository(session)


def get_worker_repo(session: Any = Depends(get_session)) -> Any:
    from app.db.repositories.worker_repo import WorkerRepository

    return WorkerRepository(session)


def get_prompt_repo(session: Any = Depends(get_session)) -> Any:
    from app.db.repositories.prompt_repo import PromptRepository

    return PromptRepository(session)


def get_pattern_repo(session: Any = Depends(get_session)) -> Any:
    from app.db.repositories.pattern_repo import PatternRepository

    return PatternRepository(session)


def get_experiment_repo(session: Any = Depends(get_session)) -> Any:
    from app.db.repositories.experiment_repo import ExperimentRepository

    return ExperimentRepository(session)


def get_repositories(
    session: Any = Depends(get_session),
) -> Repositories:
    """Dependency: every repository at once (cheaper than 9 sub-dependencies)."""
    return Repositories(
        project=_repo("app.db.repositories.project_repo", "ProjectRepository", session),
        fact=_repo("app.db.repositories.fact_repo", "FactRepository", session),
        intent=_repo("app.db.repositories.intent_repo", "IntentRepository", session),
        intent_source=_repo(
            "app.db.repositories.intent_source_repo", "IntentSourceRepository", session
        ),
        task_run=_repo(
            "app.db.repositories.task_run_repo", "TaskRunRepository", session
        ),
        worker=_repo("app.db.repositories.worker_repo", "WorkerRepository", session),
        prompt=_repo("app.db.repositories.prompt_repo", "PromptRepository", session),
        pattern=_repo("app.db.repositories.pattern_repo", "PatternRepository", session),
        experiment=_repo(
            "app.db.repositories.experiment_repo", "ExperimentRepository", session
        ),
    )


# ---------------------------------------------------------------------------
# LLM + embeddings
# ---------------------------------------------------------------------------

_LLM_CLIENT: Any = None
_LLM_CAPABILITIES: Any = None


def get_llm_client(settings: Any = Depends(get_settings)) -> Any:
    """Dependency: shared async OpenAI-compatible client (per-role models)."""
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        from app.core.llm.client import LLMClient

        _LLM_CLIENT = LLMClient.from_settings(settings)
    return _LLM_CLIENT


def get_llm_capabilities(
    client: Any = Depends(get_llm_client),
    settings: Any = Depends(get_settings),
) -> Any:
    """Dependency: startup capability matrix (``llm_capabilities`` table row)."""
    global _LLM_CAPABILITIES
    if _LLM_CAPABILITIES is None:
        cached = getattr(client, "capabilities", None)
        if cached is not None:
            _LLM_CAPABILITIES = cached
        else:
            from app.core.llm.capabilities import load_persisted_capabilities

            _LLM_CAPABILITIES = load_persisted_capabilities(settings)
    return _LLM_CAPABILITIES


def get_embedding_provider(settings: Any = Depends(get_settings)) -> Any:
    """Dependency: embedding provider with the logged fallback chain.

    ``api`` → ``sentence-transformers`` → deterministic ``hash`` provider; the
    selected backend is recorded on every retrieval trace
    (``LIMITATIONS.md`` §2).
    """
    from app.core.intent.embedding import get_embedding_provider as _provider

    return _provider(settings)


# ---------------------------------------------------------------------------
# WebSocket hub
# ---------------------------------------------------------------------------


def get_hub() -> Any:
    """Dependency: in-process WebSocket hub singleton."""
    from app.api.ws.hub import get_hub as _get_hub

    return _get_hub()


# ---------------------------------------------------------------------------
# Scope guard
# ---------------------------------------------------------------------------


def get_scope_guard(
    project_id: str,
    repos: Repositories = Depends(get_repositories),
) -> Any:
    """Dependency: scope guard loaded with the project's authorised scopes."""
    from app.core.security.scope_guard import ScopeGuard

    return ScopeGuard.from_project_id(project_id, repos)


# ---------------------------------------------------------------------------
# Request correlation id
# ---------------------------------------------------------------------------


async def get_request_id(
    request: Request,
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
) -> str:
    """Dependency: stable correlation id for the current request."""
    existing = getattr(request.state, "request_id", None)
    if existing:
        return str(existing)
    value = (x_request_id or "").strip() or uuid.uuid4().hex
    try:
        request.state.request_id = value
    except Exception:  # pragma: no cover - Request.state is always writable
        pass
    return value


# ---------------------------------------------------------------------------
# Project resolution helpers
# ---------------------------------------------------------------------------


async def require_project(
    project_id: str,
    repos: Repositories = Depends(get_repositories),
) -> Any:
    """Dependency: fetch a project or raise the canonical 404 envelope."""
    project = await repos.project.get(project_id)
    if project is None:
        raise NotFoundError(
            f"Project {project_id!r} does not exist", details={"project_id": project_id}
        )
    return project


def require_project_status(*allowed: str) -> Callable[..., Awaitable[Any]]:
    """Build a dependency asserting the project is in one of ``allowed`` states.

    Example: ``Depends(require_project_status("RUNNING"))`` for endpoints that
    may only act on a live episode.
    """

    async def _dependency(project: Any = Depends(require_project)) -> Any:
        status = getattr(project, "status", None)
        status_value = getattr(status, "value", status)
        if str(status_value) not in allowed:
            from app.api.errors import ConflictError

            raise ConflictError(
                f"Project is {status_value}; expected one of {sorted(allowed)}",
                details={
                    "project_id": str(getattr(project, "id", "")),
                    "status": status_value,
                    "allowed": list(allowed),
                },
            )
        return project

    return _dependency


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Services:
    """Bag of service objects used by the route layer."""

    project: Any
    fact: Any
    intent: Any
    retrieval: Any
    report: Any

    def get(self, name: str) -> Any:
        return getattr(self, name)


def get_services(
    session: Any = Depends(get_session),
    repos: Repositories = Depends(get_repositories),
    settings: Any = Depends(get_settings),
) -> Services:
    """Dependency: instantiate the services on top of the request repositories.

    Service constructors accept ``(session, repos, settings)``; a service that
    only needs a subset simply ignores the extra arguments (all in-repo
    services use the shared signature).
    """
    from app.services.fact_service import FactService
    from app.services.intent_service import IntentService
    from app.services.project_service import ProjectService
    from app.services.report_service import ReportService
    from app.services.retrieval_service import RetrievalService

    def build(cls: Any) -> Any:
        try:
            return cls(session=session, repos=repos, settings=settings)
        except TypeError:  # pragma: no cover - deliberately lenient
            try:
                return cls(session, repos, settings)
            except TypeError:
                return cls(session=session)

    return Services(
        project=build(ProjectService),
        fact=build(FactService),
        intent=build(IntentService),
        retrieval=build(RetrievalService),
        report=build(ReportService),
    )
