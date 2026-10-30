"""Intentest realtime WebSocket plane.

The WebSocket layer is a *fan-out*, not a source of truth. Every event that
reaches a browser originates in the immutable ``event`` table:

    domain code (scheduler / worker / services)
        └── INSERT INTO event (...)                # append-only audit row
              └── AFTER INSERT trigger
                    └── pg_notify('intentest_events', json payload)
                          └── notify_listener  (LISTEN intentest_events)
                                └── hub.broadcast()   (in-process)
                                      └── WS /ws?channel=project:{id}

This file is deliberately an *export-only* package initializer: it binds a few
constants and re-exports nothing heavy.  Importing ``app.api.ws`` therefore does
**not** import FastAPI's WebSocket classes, asyncpg, or the hub singleton.  That
keeps ``app.main`` import-light and lets tests import individual submodules
(e.g. ``app.api.ws.channels``) without a running event loop.

Submodules
----------
``channels``
    Pure channel/event-name parsing and validation helpers
    (``parse_channel``, ``project_channel``, ``channel_matches``).  No I/O.

``hub``
    ``WebSocketHub`` — the in-process registry of connected clients, with
    per-channel subscriber sets, bounded per-client queues, heartbeat/ping, and
    a ``get_hub()`` process-wide singleton.

``notify_listener``
    ``NotifyListener`` — a dedicated asyncpg connection issuing
    ``LISTEN intentest_events`` that forwards notifications into the hub, plus
    replay of missed events from the ``event`` table via ``?since=<event_id>``.

Wire format
-----------
Every frame sent to a browser is a JSON object::

    {"type": "<event_type>", "channel": "<channel>", "data": {...},
     "event_id": <int|null>, "ts": "<ISO-8601>"}

``type`` values are the ``event.event_type`` strings emitted by the backend, e.g.

``fact.created``, ``intent.state_changed``, ``task.started``, ``task.log``,
``task.finished``, ``worker.status``, ``project.status``,
``retrieval.result``, ``bridge.injected``, ``degradation.triggered``,
``scheduler.backpressure``.

Channel grammar
---------------
``global``                — administrative stream (every project; ops only)
``project:<uuid>``        — everything belonging to one project
``task:<uuid>``           — one task_run's log/phase stream
``worker:<id>``           — one worker's status/load stream

The single client-facing entry point is ``WS /ws?channel=<channel>`` (the path
constant lives in ``app.api.WS_PATH``); ``project_id=<uuid>`` is accepted as a
convenience alias.  Frames are only ever pushed for channels the client is
subscribed to, and the hub never invents payloads — it forwards what the
database already committed, so a client can always re-read the same state over
REST and observe an identical result.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Constants (kept here so submodules/tests share one definition)
# ---------------------------------------------------------------------------

#: Postgres NOTIFY/LISTEN channel carrying committed event rows.
WS_EVENT_CHANNEL: Final[str] = "intentest_events"

#: Channel used by the hub for administrative/anonymous subscribers.
GLOBAL_CHANNEL: Final[str] = "global"

#: Prefix for per-project channels: ``project:<uuid>``.
PROJECT_CHANNEL_PREFIX: Final[str] = "project:"

#: Prefix for per-task channels: ``task:<uuid>``.
TASK_CHANNEL_PREFIX: Final[str] = "task:"

#: Prefix for per-worker channels: ``worker:<id>``.
WORKER_CHANNEL_PREFIX: Final[str] = "worker:"

#: Event types the backend is documented to emit (mirrors ``app/api/ws/channels``
#: and ``API.md``).  Kept as a tuple so callers can validate without importing
#: the ``channels`` module.
KNOWN_EVENT_TYPES: Final[tuple[str, ...]] = (
    "fact.created",
    "intent.created",
    "intent.state_changed",
    "task.started",
    "task.log",
    "task.finished",
    "task.degraded",
    "worker.registered",
    "worker.status",
    "worker.heartbeat",
    "project.status",
    "project.round",
    "prompt.created",
    "prompt.consumed",
    "retrieval.result",
    "bridge.injected",
    "bridge.dropped",
    "degradation.triggered",
    "degradation.recovered",
    "scheduler.backpressure",
    "scheduler.tick",
    "llm.call",
    "experiment.status",
    "report.ready",
    "heartbeat",
    "error",
)

#: Default WebSocket ping interval (seconds) used by the hub keep-alive task.
WS_PING_INTERVAL_S: Final[float] = 25.0

#: Maximum number of buffered frames per client before the socket is dropped.
WS_MAX_QUEUE: Final[int] = 512

#: Maximum number of ``?since=`` events replayed on connect.
WS_MAX_REPLAY: Final[int] = 500

#: Submodules exposed as the package's public surface (not imported here).
__all__ = [
    "channels",
    "hub",
    "notify_listener",
    "WS_EVENT_CHANNEL",
    "GLOBAL_CHANNEL",
    "PROJECT_CHANNEL_PREFIX",
    "TASK_CHANNEL_PREFIX",
    "WORKER_CHANNEL_PREFIX",
    "KNOWN_EVENT_TYPES",
    "WS_PING_INTERVAL_S",
    "WS_MAX_QUEUE",
    "WS_MAX_REPLAY",
    "event_channel",
    "is_known_event_type",
]


def event_channel() -> str:
    """Return the Postgres NOTIFY channel name used as the event bus.

    Prefers :data:`app.api.EVENT_CHANNEL` (single source of truth) and falls
    back to :data:`WS_EVENT_CHANNEL` so this module stays importable on its own.
    """

    try:  # pragma: no cover - trivial import guard
        from app.api import EVENT_CHANNEL  # type: ignore[attr-defined]

        if isinstance(EVENT_CHANNEL, str) and EVENT_CHANNEL:
            return EVENT_CHANNEL
    except Exception:  # pragma: no cover - defensive
        pass
    return WS_EVENT_CHANNEL


def is_known_event_type(event_type: str | None) -> bool:
    """Return ``True`` when *event_type* is one of :data:`KNOWN_EVENT_TYPES`.

    Unknown types are still forwarded (the hub is schema-agnostic); this helper
    exists for the settings/health endpoints and for tests that want to pin the
    documented vocabulary.
    """

    return isinstance(event_type, str) and event_type in KNOWN_EVENT_TYPES
