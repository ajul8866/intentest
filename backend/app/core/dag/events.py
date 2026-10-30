"""Append-only event emission for the Fact--Intent DAG (audit log + WS replay bus).

The ``event`` table is the *single* source of truth for everything that happens
inside an Intentest project.  Two things consume it:

1.  ``app.api.ws.notify_listener`` -- a Postgres ``LISTEN intentest_events``
    connection that re-reads committed rows and fans them out to browser
    WebSocket clients (channel ``project:<uuid>`` etc., ``?since=<event_id>``
    replay).
2.  The reporting / metrics plane, which reconstructs a timeline from the
    append-only rows.

Delivery contract
-----------------
*   An event is **only** visible to subscribers after the surrounding
    transaction commits.  We therefore write the row with ``INSERT`` (flush,
    no commit of our own -- the caller owns the transaction, exactly like the
    repositories: ``app.api.deps.get_session`` commits once per request / the
    worker commits its unit of work) and rely on a database trigger to call
    ``pg_notify('intentest_events', <event id>)``.  ``pg_notify`` is
    transaction-scoped, so a rolled-back write never notifies.

*   Emission is **best effort and never fatal**.  Every public helper swallows
    persistence failures and logs ``event.emit_failed``; losing an audit row
    must never abort a scheduler tick or an agent turn (the DAG remains
    consistent because the DAG write itself already happened).

*   Payloads are JSON-safe by construction: :func:`jsonable` coerces UUIDs,
    datetimes, enums, ``Decimal``, mapping/list/tuple/set/deque and falls back
    to ``str()`` for anything else, so callers may dump ORM rows into
    ``data`` without a serialisation pass.

Channel naming is *not* redefined here -- ``app.api.ws.channels`` owns the
grammar (``project:``, ``worker:``, ``experiment:``, ``global``).  This module
only needs the NOTIFY channel literal, which it resolves from
``app.api.EVENT_CHANNEL`` with a defensive fallback so ``app.core`` stays
importable without the API layer (tests, alembic, eval harness).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "EVENT_CHANNEL",
    "KNOWN_EVENT_TYPES",
    "jsonable",
    "EventEmitter",
    "build_event",
    "emit_event",
    "emit",
    "emit_many",
    "event_channel",
    "make_payload",
    "EventPublisher",
]

#: Fallback NOTIFY channel name.  Kept in sync with ``app.api.EVENT_CHANNEL``
#: and the ``0001_initial.py`` trigger ``event_notify``.
EVENT_CHANNEL = "intentest_events"

#: Canonical vocabulary of persisted ``event_type`` values.  Mirrors
#: ``app.api.ws.channels.EVENT_TYPES`` (defined there for routing) so that the
#: writer-side and the reader-side cannot drift silently.  Unknown types are
#: still persisted -- the listener logs them at debug level and routes them to
#: the owning project channel -- this tuple exists for validation and tests.
KNOWN_EVENT_TYPES: tuple[str, ...] = (
    # --- facts -----------------------------------------------------------
    "fact.created",
    "fact.verified",
    "fact.superseded",
    # --- intents ---------------------------------------------------------
    "intent.created",
    "intent.state_changed",
    "intent.injected",
    # --- graph -----------------------------------------------------------
    "graph.updated",
    "integrity.violation",
    # --- scheduler / tasks ----------------------------------------------
    "scheduler.tick",
    "scheduler.backpressure",
    "scheduler.lease",
    "task.started",
    "task.progress",
    "task.log",
    "task.finished",
    "task.degraded",
    # --- workers ---------------------------------------------------------
    "worker.registered",
    "worker.status",
    "worker.heartbeat",
    "worker.drained",
    # --- projects --------------------------------------------------------
    "project.created",
    "project.status",
    "project.phase",
    "project.round",
    "project.terminated",
    # --- knowledge plane (Algorithms 2 & 3) ------------------------------
    "retrieval.result",
    "bridge.injected",
    "bridge.dropped",
    "degradation.triggered",
    "degradation.recovered",
    # --- hints -----------------------------------------------------------
    "prompt.created",
    "prompt.consumed",
    # --- experiments -----------------------------------------------------
    "experiment.created",
    "experiment.status",
    "experiment.task_finished",
    "experiment.completed",
)

#: Worker-scoped event types (routed to ``worker:<id>`` in addition to the
#: owning project channel).
_WORKER_SCOPED: frozenset[str] = frozenset(
    {
        "worker.registered",
        "worker.status",
        "worker.heartbeat",
        "worker.drained",
        "scheduler.backpressure",
        "scheduler.lease",
    }
)


# ---------------------------------------------------------------------------
# JSON coercion
# ---------------------------------------------------------------------------
def jsonable(value: Any) -> Any:
    """Recursively coerce ``value`` into something ``json.dumps`` accepts.

    Handles the types that actually show up in our payloads: ``UUID``,
    ``datetime``/``date``/``time``, ``Enum``, ``Decimal``, ``bytes``,
    ``Mapping``/``list``/``tuple``/``set``/``frozenset``/``deque``.  Anything
    else is stringified so a rogue ORM attribute can never break the event bus.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        # ``bool`` before ``int`` is irrelevant for correctness here but keeps
        # the isinstance chain readable.
        return value
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, Decimal):
        try:
            return float(value)
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    # ``collections.deque`` and other sequences we do not want to hard-import.
    if hasattr(value, "__iter__") and not isinstance(value, str):
        try:
            return [jsonable(item) for item in value]  # type: ignore[union-attr]
        except TypeError:  # pragma: no cover - defensive
            pass
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)


def make_payload(data: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Return a JSON-safe copy of ``data`` (``None`` -> ``{}``)."""

    if not data:
        return {}
    return {str(key): jsonable(value) for key, value in dict(data).items()}


# ---------------------------------------------------------------------------
# Channel resolution
# ---------------------------------------------------------------------------
def event_channel() -> str:
    """Return the Postgres NOTIFY channel name.

    Prefers ``app.api.EVENT_CHANNEL`` (single source of truth, shared with the
    WS listener and the health probe) and falls back to :data:`EVENT_CHANNEL`
    when the API package is unavailable.
    """

    try:  # pragma: no cover - trivial import dance
        from app.api import EVENT_CHANNEL as _channel  # type: ignore

        if isinstance(_channel, str) and _channel.strip():
            return _channel.strip()
    except Exception:
        pass
    return EVENT_CHANNEL


# ---------------------------------------------------------------------------
# Event construction
# ---------------------------------------------------------------------------
def build_event(
    event_type: str,
    *,
    project_id: Any = None,
    payload: Optional[Mapping[str, Any]] = None,
    worker_id: Any = None,
    experiment_id: Any = None,
    intent_id: Any = None,
    fact_id: Any = None,
    task_run_id: Any = None,
    message: Optional[str] = None,
    scope: Optional[str] = None,
) -> dict[str, Any]:
    """Build the *column mapping* for one ``event`` row.

    The returned dict is what :meth:`EventEmitter.emit` passes to the ORM /
    repository.  Keys are exactly the ``event`` table columns used elsewhere in
    the codebase (``event_type``, ``project_id``, ``worker_id``,
    ``experiment_id``, ``payload``, ``scope``, ``created_at``); the convenience
    ids (``intent_id``/``fact_id``/``task_run_id``) are folded into ``payload``
    because the table keeps them there (the listener's routing helpers read
    them back out of ``payload``).
    """

    body: dict[str, Any] = make_payload(payload)
    for key, value in (
        ("intent_id", intent_id),
        ("fact_id", fact_id),
        ("task_run_id", task_run_id),
        ("worker_id", worker_id),
        ("experiment_id", experiment_id),
    ):
        if value is not None and key not in body:
            body[key] = jsonable(value)
    if message is not None and "message" not in body:
        body["message"] = str(message)

    return {
        "event_type": str(event_type),
        "project_id": _as_uuid(project_id),
        "worker_id": (str(worker_id)[:200] if worker_id is not None else None),
        "experiment_id": _as_uuid(experiment_id),
        "payload": body,
        "scope": (str(scope) if scope is not None else None),
        "created_at": datetime.now(timezone.utc),
    }


def _as_uuid(value: Any) -> Any:
    """Coerce ``value`` to ``uuid.UUID`` when possible, else ``None``."""

    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------
class EventEmitter:
    """Writes ``event`` rows and (best effort) notifies the WS bus.

    Constructor:: 

        EventEmitter(session=None, *, project_id=None, service="intentest")

    ``session`` is an ``AsyncSession`` (duck-typed: we only need ``add`` plus
    ``flush``/``commit``), a repository exposing ``add_event()`` /
    ``create()``, or ``None`` for a *no-op* emitter (unit tests, dry runs).
    ``project_id`` acts as the default subject for every event, so call sites
    do not have to repeat it.

    The emitter never commits: transaction ownership stays with the caller
    (``app.api.deps.get_session`` for HTTP, the worker's unit of work for agent
    turns).  When the caller commits, the ``event_notify`` trigger publishes
    ``pg_notify(channel, id)`` and the listener reloads the committed row.
    """

    def __init__(
        self,
        session: Any = None,
        *,
        project_id: Any = None,
        service: str = "intentest",
        channel: Optional[str] = None,
        hub: Any = None,
    ) -> None:
        self.session = session
        self.project_id = project_id
        self.service = service
        self._channel = channel
        self._hub = hub
        self._emitted = 0

    # -- introspection ----------------------------------------------------
    @property
    def channel(self) -> str:
        return self._channel or event_channel()

    @property
    def emitted(self) -> int:
        """Number of events this emitter has successfully persisted."""

        return self._emitted

    # -- emission ---------------------------------------------------------
    async def emit(
        self,
        event_type: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        project_id: Any = None,
        worker_id: Any = None,
        experiment_id: Any = None,
        intent_id: Any = None,
        fact_id: Any = None,
        task_run_id: Any = None,
        message: Optional[str] = None,
        scope: Optional[str] = None,
        flush: bool = True,
    ) -> Any:
        """Persist one event row and return it (``None`` on failure)."""

        values = build_event(
            event_type,
            project_id=project_id if project_id is not None else self.project_id,
            payload=payload,
            worker_id=worker_id,
            experiment_id=experiment_id,
            intent_id=intent_id,
            fact_id=fact_id,
            task_run_id=task_run_id,
            message=message,
            scope=scope,
        )
        row = await self._persist(values, flush=flush)
        if row is not None:
            self._emitted += 1
        return row

    # Convenience wrappers used across the codebase -----------------------
    async def fact_created(self, fact: Any, **kwargs: Any) -> Any:
        return await self.emit(
            "fact.created",
            _entity_payload(fact, "fact"),
            project_id=kwargs.pop("project_id", None) or _attr(fact, "project_id"),
            fact_id=_attr(fact, "id"),
            **kwargs,
        )

    async def intent_state_changed(
        self,
        intent: Any,
        *,
        from_state: Any = None,
        to_state: Any = None,
        **kwargs: Any,
    ) -> Any:
        extra = {"from_state": _name(from_state), "to_state": _name(to_state)}
        return await self.emit(
            "intent.state_changed",
            {**_entity_payload(intent, "intent"), **extra},
            project_id=kwargs.pop("project_id", None) or _attr(intent, "project_id"),
            intent_id=_attr(intent, "id"),
            worker_id=_attr(intent, "worker_id"),
            **kwargs,
        )

    async def task_event(self, event_type: str, task_run: Any, **kwargs: Any) -> Any:
        return await self.emit(
            event_type,
            _entity_payload(task_run, "task_run"),
            project_id=kwargs.pop("project_id", None) or _attr(task_run, "project_id"),
            task_run_id=_attr(task_run, "id"),
            worker_id=kwargs.pop("worker_id", None) or _attr(task_run, "worker_id"),
            **kwargs,
        )

    async def project_status(
        self,
        project: Any,
        *,
        status: Any = None,
        reason: Any = None,
        **kwargs: Any,
    ) -> Any:
        return await self.emit(
            "project.status",
            {
                "status": _name(status if status is not None else _attr(project, "status")),
                "phase": _name(_attr(project, "phase")),
                "termination_reason": _name(
                    reason if reason is not None else _attr(project, "termination_reason")
                ),
                "rounds_used": _attr(project, "rounds_used"),
                "max_rounds": _attr(project, "max_rounds"),
            },
            project_id=kwargs.pop("project_id", None) or _attr(project, "id"),
            **kwargs,
        )

    # -- internals --------------------------------------------------------
    async def _persist(self, values: Mapping[str, Any], *, flush: bool = True) -> Any:
        if self.session is None:
            # No-op emitter: still log so trace files/tests can assert intent.
            logger.debug("event.no_session", extra={"event_type": values.get("event_type")})
            return dict(values)

        # 1. repository path (preferred: it knows about idempotency/seq cols)
        for name in ("add_event", "emit_event", "create_event", "create", "add"):
            method = getattr(self.session, name, None)
            if method is None:
                continue
            try:
                result = method(values)
                if hasattr(result, "__await__"):
                    result = await result
                if result is not None:
                    await self._flush(flush)
                    await self._fan_out(result)
                    return result
            except TypeError:
                continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "event.emit_failed",
                    extra={
                        "event_type": values.get("event_type"),
                        "method": name,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                return None

        # 2. ORM path: instantiate the model named in the mapping.
        row = await self._persist_orm(values, flush=flush)
        if row is not None:
            await self._fan_out(row)
        return row

    async def _persist_orm(self, values: Mapping[str, Any], *, flush: bool) -> Any:
        model = _event_model()
        if model is None:
            logger.debug(
                "event.model_unavailable", extra={"event_type": values.get("event_type")}
            )
            return None
        try:
            row = _construct(model, values)
            add = getattr(self.session, "add", None)
            if callable(add):
                add(row)
            await self._flush(flush)
            return row
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "event.emit_failed",
                extra={
                    "event_type": values.get("event_type"),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return None

    async def _flush(self, flush: bool) -> None:
        if not flush:
            return
        flush_fn = getattr(self.session, "flush", None)
        if flush_fn is None:
            return
        try:
            result = flush_fn()
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("event.flush_failed", extra={"error": str(exc)})

    async def _fan_out(self, row: Any) -> None:
        """Optionally push the event to the in-process hub (same process only).

        Cross-process delivery always goes through ``pg_notify``; this hop only
        saves latency when the emitter and the API happen to share a process
        (single-container dev, tests).
        """

        hub = self._hub
        if hub is None:
            return
        publish = getattr(hub, "publish_event", None)
        if publish is None:
            return
        try:
            result = publish(row)
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("event.fan_out_failed", extra={"error": str(exc)})


# Backwards/forwards compatible alias: some call sites say "publisher".
EventPublisher = EventEmitter


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
async def emit_event(
    session: Any,
    event_type: str,
    payload: Optional[Mapping[str, Any]] = None,
    *,
    project_id: Any = None,
    **kwargs: Any,
) -> Any:
    """One-shot emission helper: ``await emit_event(session, "task.log", {...})``."""

    emitter = EventEmitter(session, project_id=project_id)
    return await emitter.emit(event_type, payload, project_id=project_id, **kwargs)


async def emit(
    session: Any,
    project_id: Any,
    event_type: str,
    payload: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Positional-argument variant of :func:`emit_event` used by services."""

    return await emit_event(session, event_type, payload, project_id=project_id, **kwargs)


async def emit_many(
    session: Any,
    events: Iterable[Any],
    *,
    project_id: Any = None,
) -> list[Any]:
    """Persist several events in one flush; returns the written rows.

    ``events`` entries are either ``(event_type, payload)`` tuples, mappings
    containing ``event_type``/``payload`` keys, or pre-built column mappings.
    """

    emitter = EventEmitter(session, project_id=project_id)
    written: list[Any] = []
    for entry in events:
        if isinstance(entry, tuple):
            event_type, payload = (entry + (None,))[:2]
            row = await emitter.emit(event_type, payload, flush=False)
        elif isinstance(entry, Mapping) and "event_type" in entry:
            values = dict(entry)
            if values.get("project_id") is None and project_id is not None:
                values["project_id"] = project_id
            values.setdefault("created_at", datetime.now(timezone.utc))
            values["payload"] = make_payload(values.get("payload"))
            values["project_id"] = _as_uuid(values.get("project_id"))
            row = await emitter._persist(values, flush=False)
        else:  # pragma: no cover - defensive
            logger.debug("event.emit_many_skipped", extra={"entry_type": type(entry).__name__})
            continue
        if row is not None:
            written.append(row)
    # single flush for the whole batch
    await emitter._flush(True)
    return written


# ---------------------------------------------------------------------------
# Tolerant attribute access / model loading
# ---------------------------------------------------------------------------
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first present attribute/key from ORM rows, dicts or mocks."""

    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            value = obj[name]
        else:
            value = getattr(obj, name, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    return default


_MISSING = object()


def _name(value: Any) -> Optional[str]:
    """Normalise an enum/str state name to its upper-case string value."""

    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    try:
        return str(value).upper()
    except Exception:  # pragma: no cover - defensive
        return None


def _entity_payload(entity: Any, kind: str) -> dict[str, Any]:
    """Small, stable projection of a domain row for event payloads."""

    identifier = _attr(entity, "id")
    if identifier is None:
        return {"kind": kind}
    out: dict[str, Any] = {"kind": kind, "id": identifier}
    for key in (
        "state",
        "status",
        "phase",
        "success",
        "action_type",
        "tactic",
        "objective",
        "kind",
        "fact_key",
        "round_index",
        "worker_id",
        "priority",
        "confidence",
        "verified",
    ):
        value = _attr(entity, key)
        if value is not None:
            out[key] = value
    return out


def _construct(model: Any, values: Mapping[str, Any]) -> Any:
    """Instantiate ``model`` from ``values``, filtering to declared columns.

    SQLAlchemy models raise ``TypeError`` on unknown kwargs, so we prefer the
    declarative ``__table__.columns`` set when available and fall back to the
    full mapping for plain dataclasses / test doubles.
    """

    table = getattr(model, "__table__", None)
    columns = getattr(table, "columns", None)
    if columns is not None:
        try:
            allowed = {column.name for column in columns}
        except Exception:  # pragma: no cover - defensive
            allowed = set()
        if allowed:
            filtered = {k: v for k, v in values.items() if k in allowed}
            return model(**filtered)
    return model(**dict(values))


def _event_model() -> Any:
    """Lazily resolve the ``Event`` ORM model (``None`` when unavailable)."""

    try:
        from app.db.models.event import Event  # type: ignore

        return Event
    except Exception:
        pass
    try:  # tolerate a differently-named module
        from app.db.models.events import Event  # type: ignore

        return Event
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sync convenience (scheduler/tests that are not async)
# ---------------------------------------------------------------------------
def serialise(values: Mapping[str, Any]) -> str:
    """``json.dumps`` an event mapping with :func:`jsonable` coercion."""

    return json.dumps(make_payload(values), sort_keys=True, default=str)


def summarise(event: Any) -> dict[str, Any]:
    """Project any event row/mapping into the WS wire-frame ``data`` block."""

    event_type = _attr(event, "event_type", "type", "name", default="") or ""
    payload = _attr(event, "payload", "data", "body")
    data = make_payload(payload) if isinstance(payload, Mapping) else {}
    for key in (
        "project_id",
        "worker_id",
        "experiment_id",
        "intent_id",
        "fact_id",
        "task_run_id",
        "message",
        "detail",
    ):
        if key in data:
            continue
        value = _attr(event, key)
        if value is not None:
            data[key] = jsonable(value)
    return {
        "type": str(event_type),
        "event_id": _as_int(_attr(event, "id", "event_id")),
        "data": data,
        "ts": jsonable(_attr(event, "created_at", "ts")),
    }


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_known_event_type(event_type: Any) -> bool:
    """``True`` when ``event_type`` is in :data:`KNOWN_EVENT_TYPES`."""

    if not isinstance(event_type, str):
        return False
    return event_type.strip().lower() in KNOWN_EVENT_TYPES


def is_worker_scoped(event_type: Any) -> bool:
    """``True`` for event types that also target ``worker:<id>``."""

    if not isinstance(event_type, str):
        return False
    return event_type.strip().lower() in _WORKER_SCOPED
