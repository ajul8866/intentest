"""HTTP/WebSocket API layer for the Intentest backend.

This package holds the FastAPI-facing surface split into three concerns:

``app.api.deps``
    Request-scoped dependency providers (DB session, settings, pagination,
    repository/service accessors) shared by every route module.

``app.api.errors``
    The uniform error envelope (``{"error": {...}}``) plus exception handlers
    and the typed exception hierarchy that services raise
    (``NotFoundError``, ``IllegalTransitionError``, ``PreconditionFailed``,
    ``ScopeViolation``, ``LeaseHeld``, ``IntegrityViolation``, ...). Route
    modules never build error payloads by hand — they raise and let the
    registered handlers serialize.

``app.api.routes``
    One module per resource (projects, graph, facts, intents, tasks, workers,
    prompts, retrieval, patterns, experiments, reports, settings, health).
    Routes stay thin: validate with Pydantic schemas from ``app.schemas``,
    delegate to ``app.services``, return the schema object.

``app.api.ws``
    The WebSocket hub (``hub``), channel naming/parsing (``channels``) and the
    Postgres ``LISTEN intentest_events`` listener that fans committed events
    out to connected clients (``notify_listener``).

Design rules enforced across this package
-----------------------------------------
* All REST routes live under ``/api/v1`` except ``/healthz``, ``/readyz`` and
  ``/ws`` (see ``API.md``).
* List endpoints accept ``limit``/``offset`` and answer
  ``{"items": [...], "total": <int>}``.
* Errors use the envelope defined in ``app.api.errors.ERROR_ENVELOPE``.
* No route imports the scheduler or the worker execution plane directly; the
  only coupling is through the database and ``services``.

This ``__init__`` stays import-light on purpose: importing ``app.api`` must not
pull in FastAPI routers, SQLAlchemy engines or the Docker SDK. Callers import
the submodules they need explicitly (``from app.api import errors``), which
keeps ``app.api.errors`` usable from unit tests and from CLI entrypoints that
have no web server running.
"""

from __future__ import annotations

__all__ = [
    "deps",
    "errors",
    "routes",
    "ws",
]

# NOTE: intentionally no eager ``from . import routes`` here. ``app.main``
# imports each router module directly so that a broken optional dependency in a
# single route module produces a precise error instead of a package-level one.
API_PREFIX = "/api/v1"
"""Prefix every versioned REST route is mounted under (see ``API.md``)."""

WS_PATH = "/ws"
"""WebSocket endpoint path; clients pass ``?channel=project:{id}``."""

EVENT_CHANNEL = "intentest_events"
"""Postgres NOTIFY channel used as the event bus (no Redis in this system)."""


def route_modules() -> list[str]:
    """Return the canonical list of route module names.

    ``app.main`` iterates this list to mount routers, and tests use it to
    assert that every documented resource has a module and that importing each
    one succeeds.
    """

    return [
        "projects",
        "graph",
        "facts",
        "intents",
        "tasks",
        "workers",
        "prompts",
        "retrieval",
        "patterns",
        "experiments",
        "reports",
        "settings",
        "health",
    ]
