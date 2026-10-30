"""Postgres ``LISTEN`` -> in-process hub fan-out.

The event bus is Postgres itself: every row inserted into the ``event`` table
fires a trigger that calls ``pg_notify('intentest_events', <event id>)``.  This
module owns the *only* LISTEN connection in the backend process: it drains the
notification stream, loads the committed ``event`` rows (the NOTIFY payload is
just an id, never the payload, so the DB remains the single source of truth),
routes each row to its channels via :mod:`app.api.ws.channels` and hands the
frames to :mod:`app.api.ws.hub`.

Design notes
------------
* **At-least-once** delivery: notifications may be coalesced or dropped by
  Postgres when a listener is not connected, therefore on (re)start the listener
  performs a bounded catch-up read from the ``event`` table using the last
  delivered id (``?since=`` semantics are shared with the WS endpoint).
* **No busy loop**: uses ``asyncpg``'s native ``add_listener`` callback with an
  ``asyncio.Queue`` wake-up; when asyncpg is unavailable we fall back to a
  polling loop over the ``event`` table (``poll_interval_s``), which keeps the
  reproduction functional on drivers without LISTEN support while logging the
  degraded mode loudly.
* **Never raises into the server**: transient DB errors are logged and the
  connection is retried with exponential backoff.  A failure to start the
  listener must not prevent the API from serving REST traffic.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from app.api.ws import channels as ws_channels

logger = logging.getLogger(__name__)

__all__ = [
    "NotifyListener",
    "Notifier",
    "get_listener",
    "start_listener",
    "stop_listener",
    "publish",
    "envelope",
]

#: Default channel name; kept in sync with ``app.api.EVENT_CHANNEL``.
DEFAULT_CHANNEL = "intentest_events"
#: How many events to replay on (re)connect / catch-up.
CATCH_UP_LIMIT = 500
#: Reconnect backoff bounds (seconds).
MIN_BACKOFF_S = 0.5
MAX_BACKOFF_S = 15.0
#: Default polling interval for the driver without LISTEN support.
DEFAULT_POLL_INTERVAL_S = 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Tolerant attribute/mapping access (ORM row, dict, namespace)."""
    for name in names:
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
            continue
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _iso(value: Any) -> Any:
    """JSON-safe coercion for ids / datetimes / enums."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    enum_value = getattr(value, "value", None)
    if enum_value is not None and not callable(enum_value):
        return _iso(enum_value)
    return str(value)


def _jsonable(value: Any) -> Any:
    """Recursively coerce ``value`` into something ``json.dumps`` accepts."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return _iso(value)


def _parse_payload(raw: Any) -> Any:
    """Decode a NOTIFY payload.  Accepts a bare id or a JSON object."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8", "replace")
        except Exception:  # pragma: no cover - defensive
            return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text[0] in "{[":
        try:
            return json.loads(text)
        except Exception:
            logger.debug("ws.notify_payload_unparsable", extra={"payload": text[:200]})
    return text


def envelope(event: Any, *, event_type: Optional[str] = None, channel: str = "") -> dict[str, Any]:
    """Build the documented WS wire frame for one event row.

    Frame shape (see ``app.api.ws``)::

        {"type": str, "channel": str, "data": {...}, "event_id": int|null,
         "ts": "<ISO-8601>"}
    """
    resolved_type = (
        event_type
        or _attr(event, "event_type", "type", "name", default="")
        or ""
    )
    payload = _attr(event, "payload", "data", "meta", default=None)
    if payload is None:
        payload = {}
        # Fall back to a flattened view so a row without a JSON payload still
        # carries useful information (e.g. hand-inserted audit rows).
        for key in ("project_id", "intent_id", "fact_id", "task_run_id", "worker_id",
                    "experiment_id", "message", "detail"):
            value = _attr(event, key)
            if value is not None:
                payload[key] = _iso(value)
    frame = {
        "type": str(resolved_type),
        "channel": channel or (ws_channels.channels_for_event(event) or [ws_channels.GLOBAL])[0],
        "data": _jsonable(payload),
        "event_id": _iso(_attr(event, "id", "event_id")),
        "ts": _iso(_attr(event, "created_at", "ts", "timestamp", default=None)),
    }
    return frame


# ---------------------------------------------------------------------------
# listener
# ---------------------------------------------------------------------------
class NotifyListener:
    """Owns the ``LISTEN intentest_events`` connection and fans out events.

    Parameters
    ----------
    hub:
        Object exposing ``await hub.broadcast(channel, frame)`` and optionally
        ``await hub.send_to_channel(channel, frame)``.  The instance from
        :func:`app.api.ws.hub.get_hub` is used by default.
    database_url:
        Explicit DSN; defaults to ``app.config.settings.database_url``.
    channel:
        Postgres channel; defaults to ``app.api.EVENT_CHANNEL``.
    """

    def __init__(
        self,
        hub: Any = None,
        *,
        database_url: Optional[str] = None,
        channel: Optional[str] = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        catch_up_limit: int = CATCH_UP_LIMIT,
    ) -> None:
        self._hub = hub
        self._database_url = database_url
        self.channel = channel or _default_channel()
        self.poll_interval_s = max(0.05, float(poll_interval_s))
        self.catch_up_limit = max(0, int(catch_up_limit))

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue(maxsize=4096)
        self._connection: Any = None
        self._pool: Any = None
        self._last_event_id: Optional[int] = None
        self._mode: str = "idle"  # idle | listen | poll
        self._delivered = 0
        self._errors = 0
        self._started_at: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def last_event_id(self) -> Optional[int]:
        return self._last_event_id

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "mode": self._mode,
            "channel": self.channel,
            "last_event_id": self._last_event_id,
            "delivered": self._delivered,
            "errors": self._errors,
            "started_at": self._started_at,
            "queue_size": self._queue.qsize(),
        }

    async def start(self) -> "NotifyListener":
        """Start the background task (idempotent)."""
        if self._task is not None and not self._task.done():
            return self
        self._running = True
        self._started_at = _iso(_utcnow())
        self._task = asyncio.ensure_future(self.run())
        logger.info(
            "ws.listener_started",
            extra={"channel": self.channel, "poll_interval_s": self.poll_interval_s},
        )
        return self

    async def stop(self) -> None:
        """Cancel the background task and release the LISTEN connection."""
        self._running = False
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self._close_connection()
        logger.info("ws.listener_stopped", extra={"channel": self.channel})

    async def __aenter__(self) -> "NotifyListener":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.stop()

    # -- main loop ---------------------------------------------------------
    async def run(self) -> None:
        """Reconnect forever until :meth:`stop` is called."""
        backoff = MIN_BACKOFF_S
        while self._running:
            try:
                await self._run_listen()
                backoff = MIN_BACKOFF_S
            except asyncio.CancelledError:  # pragma: no cover - cancellation
                raise
            except Exception as exc:  # noqa: BLE001 - resilience boundary
                self._errors += 1
                logger.warning(
                    "ws.listener_error",
                    extra={"error": f"{type(exc).__name__}: {exc}", "channel": self.channel},
                )
                if not self._running:
                    break
                delay = min(MAX_BACKOFF_S, backoff) * (0.75 + random.random() * 0.5)
                await asyncio.sleep(delay)
                backoff = min(MAX_BACKOFF_S, backoff * 2)

    async def _run_listen(self) -> None:
        """Try native LISTEN; fall back to polling the ``event`` table."""
        try:
            import asyncpg  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            self._mode = "poll"
            logger.warning(
                "ws.listener_poll_mode",
                extra={"reason": "asyncpg is not installed", "channel": self.channel},
            )
            await self._run_poll()
            return

        dsn = self._resolve_dsn()
        if not dsn:
            raise RuntimeError("DATABASE_URL is not configured; cannot LISTEN")

        self._connection = await asyncpg.connect(self._normalise_dsn(dsn))
        self._mode = "listen"
        try:
            await self.catch_up()
            await self._connection.add_listener(self.channel, self._on_notify)
            logger.info("ws.listener_subscribed", extra={"channel": self.channel})
            while self._running:
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Idle heartbeat keeps the connection warm and lets us
                    # notice a dropped socket.
                    await self._connection.execute("SELECT 1")
                    continue
                await self._handle_notification(item)
        finally:
            with contextlib.suppress(Exception):
                await self._connection.remove_listener(self.channel, self._on_notify)
            await self._close_connection()

    async def _run_poll(self) -> None:
        """Polling fallback: read ``event`` rows sequentially."""
        while self._running:
            try:
                await self.drain_pending()
            except asyncio.CancelledError:  # pragma: no cover
                raise
            except Exception as exc:  # noqa: BLE001
                self._errors += 1
                logger.warning(
                    "ws.listener_poll_error",
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                )
            await asyncio.sleep(self.poll_interval_s)

    # -- notifications -----------------------------------------------------
    def _on_notify(self, _conn: Any, _pid: int, _channel: str, payload: Any) -> None:
        """asyncpg callback: must stay synchronous, so it only enqueues."""
        try:
            self._queue.put_nowait(_parse_payload(payload))
        except asyncio.QueueFull:  # pragma: no cover - overload
            self._errors += 1
            logger.warning("ws.listener_queue_full", extra={"channel": self.channel})

    async def _handle_notification(self, item: Any) -> None:
        event_id: Optional[int] = None
        if isinstance(item, int):
            event_id = item
        elif isinstance(item, Mapping):
            raw_id = item.get("id") or item.get("event_id")
            event_id = int(raw_id) if isinstance(raw_id, (int, str)) and str(raw_id).isdigit() else None
            if event_id is None:
                # Payload carried a full frame; broadcast verbatim.
                await self._broadcast_frame(item)
                return
        elif isinstance(item, str) and item.isdigit():
            event_id = int(item)

        if event_id is None:
            await self.drain_pending()
            return

        row = await self._load_event(event_id)
        if row is None:
            # Nothing committed yet (NOTIFY fires inside the tx) - fall back to
            # a bounded catch-up scan which will pick it up once visible.
            await asyncio.sleep(0)
            row = await self._load_event(event_id)
        if row is None:
            return
        await self._publish_row(row)

    async def catch_up(self) -> int:
        """Replay up to ``catch_up_limit`` events after ``_last_event_id``."""
        rows = await self._fetch_events(after=self._last_event_id, limit=self.catch_up_limit)
        for row in rows:
            await self._publish_row(row)
        return len(rows)

    async def drain_pending(self) -> int:
        """Load and publish every committed event not yet delivered."""
        rows = await self._fetch_events(after=self._last_event_id, limit=self.catch_up_limit)
        for row in rows:
            await self._publish_row(row)
        return len(rows)

    async def _publish_row(self, row: Any) -> None:
        event_id = _attr(row, "id", "event_id")
        try:
            if event_id is not None:
                numeric = int(event_id)
                if self._last_event_id is not None and numeric <= self._last_event_id:
                    return
                self._last_event_id = numeric
        except (TypeError, ValueError):  # pragma: no cover - non-numeric ids
            pass

        targets = ws_channels.channels_for_event(row) or []
        event_type = _attr(row, "event_type", "type", "name", default="")
        if event_type and not ws_channels.is_known_event_type(str(event_type)):
            logger.debug("ws.unknown_event_type", extra={"event_type": str(event_type)})
        for channel in targets:
            await self._broadcast_frame(envelope(row, event_type=event_type, channel=channel), channel)
        self._delivered += 1

    async def _broadcast_frame(self, frame: Mapping[str, Any], channel: Optional[str] = None) -> None:
        hub = self._hub_instance()
        if hub is None:
            return
        target = channel or str(frame.get("channel") or ws_channels.GLOBAL)
        for name in ("broadcast", "publish", "send_to_channel", "emit"):
            method = getattr(hub, name, None)
            if not callable(method):
                continue
            try:
                result = method(target, dict(frame))
                if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                    await result
                return
            except TypeError:
                continue
            except Exception as exc:  # noqa: BLE001 - hub must not kill listener
                logger.warning(
                    "ws.hub_dispatch_failed",
                    extra={"error": f"{type(exc).__name__}: {exc}", "channel": target},
                )
                return
        logger.debug("ws.hub_missing_broadcast", extra={"channel": target})

    # -- database access ---------------------------------------------------
    def _hub_instance(self) -> Any:
        if self._hub is not None:
            return self._hub
        try:
            from app.api.ws.hub import get_hub

            self._hub = get_hub()
        except Exception:  # pragma: no cover - hub always importable
            logger.debug("ws.hub_unavailable")
        return self._hub

    def _resolve_dsn(self) -> Optional[str]:
        if self._database_url:
            return self._database_url
        try:
            from app.config import settings

            for name in ("database_url", "DATABASE_URL", "db_url"):
                value = getattr(settings, name, None)
                if value:
                    return str(value)
        except Exception:  # pragma: no cover - config always importable
            pass
        return os.environ.get("DATABASE_URL")

    @staticmethod
    def _normalise_dsn(dsn: str) -> str:
        """asyncpg wants a plain ``postgresql://`` DSN (no SQLAlchemy driver)."""
        for prefix in ("postgresql+asyncpg://", "postgres+asyncpg://"):
            if dsn.startswith(prefix):
                return "postgresql://" + dsn[len(prefix):]
        return dsn

    async def _fetch_events(self, *, after: Optional[int], limit: int) -> Sequence[Any]:
        """Read committed ``event`` rows ordered by id."""
        dsn = self._resolve_dsn()
        if not dsn:
            return ()
        limit = max(1, int(limit))
        sql = (
            "SELECT id, event_type, project_id, worker_id, payload, created_at "
            "FROM event WHERE ($1::bigint IS NULL OR id > $1) "
            "ORDER BY id ASC LIMIT $2"
        )
        try:
            import asyncpg  # type: ignore

            conn = self._connection
            owns = False
            if conn is None or getattr(conn, "is_closed", lambda: True)():
                conn = await asyncpg.connect(self._normalise_dsn(dsn))
                owns = True
            try:
                rows = await conn.fetch(sql, after, limit)
            finally:
                if owns:
                    with contextlib.suppress(Exception):
                        await conn.close()
            return [_RowProxy(row) for row in rows]
        except Exception as exc:  # noqa: BLE001 - fall through to SQLAlchemy
            logger.debug(
                "ws.event_fetch_asyncpg_failed",
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )

        return await self._fetch_events_sqlalchemy(after=after, limit=limit)

    async def _fetch_events_sqlalchemy(self, *, after: Optional[int], limit: int) -> Sequence[Any]:
        """Driver-independent fallback using the app's async session factory."""
        try:
            from sqlalchemy import text

            factory = _session_factory()
            if factory is None:
                return ()
            async with factory() as session:
                result = await session.execute(
                    text(
                        "SELECT id, event_type, project_id, worker_id, payload, created_at "
                        "FROM event WHERE (:after IS NULL OR id > :after) "
                        "ORDER BY id ASC LIMIT :limit"
                    ),
                    {"after": after, "limit": max(1, int(limit))},
                )
                rows = result.mappings().all()
                return [_RowProxy(dict(row)) for row in rows]
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "ws.event_fetch_sqlalchemy_failed",
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )
            return ()

    async def _load_event(self, event_id: int) -> Any:
        try:
            import asyncpg  # type: ignore

            dsn = self._resolve_dsn()
            if not dsn:
                return None
            conn = self._connection
            owns = False
            if conn is None or getattr(conn, "is_closed", lambda: True)():
                conn = await asyncpg.connect(self._normalise_dsn(dsn))
                owns = True
            try:
                row = await conn.fetchrow(
                    "SELECT id, event_type, project_id, worker_id, payload, created_at "
                    "FROM event WHERE id = $1",
                    event_id,
                )
            finally:
                if owns:
                    with contextlib.suppress(Exception):
                        await conn.close()
            return _RowProxy(row) if row is not None else None
        except Exception as exc:  # noqa: BLE001 - fall through
            logger.debug(
                "ws.event_load_asyncpg_failed",
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )

        try:
            from sqlalchemy import text

            factory = _session_factory()
            if factory is None:
                return None
            async with factory() as session:
                result = await session.execute(
                    text(
                        "SELECT id, event_type, project_id, worker_id, payload, created_at "
                        "FROM event WHERE id = :id"
                    ),
                    {"id": int(event_id)},
                )
                row = result.mappings().first()
                return _RowProxy(dict(row)) if row is not None else None
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "ws.event_load_sqlalchemy_failed",
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )
            return None

    async def _close_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            with contextlib.suppress(Exception):
                await connection.close()

    # -- manual injection (tests / non-Postgres drivers) -------------------
    async def publish_event(self, event: Any) -> int:
        """Publish an in-memory event row (no DB round-trip).

        Used by tests and by code paths that already hold the committed row.
        Returns the number of channels the frame was routed to.
        """
        targets = ws_channels.channels_for_event(event) or []
        event_type = _attr(event, "event_type", "type", "name", default="")
        for channel in targets:
            await self._broadcast_frame(envelope(event, event_type=event_type, channel=channel), channel)
        if targets:
            self._delivered += 1
        return len(targets)


class _RowProxy:
    """Attribute view over an asyncpg Record / mapping, without silent loss."""

    __slots__ = ("_data",)

    def __init__(self, data: Any) -> None:
        if data is None:
            self._data: Mapping[str, Any] = {}
        elif isinstance(data, Mapping):
            self._data = data
        else:  # asyncpg.Record supports dict()
            try:
                self._data = dict(data)
            except Exception:  # pragma: no cover - exotic driver types
                self._data = {}

    def __getattr__(self, item: str) -> Any:
        try:
            return self._data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def payload(self) -> Any:
        raw = self._data.get("payload")
        if isinstance(raw, str):
            return _parse_payload(raw) or {}
        return raw if raw is not None else {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_RowProxy({self._data!r})"


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------
def _utcnow() -> Any:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def _default_channel() -> str:
    try:
        from app.api import EVENT_CHANNEL

        if isinstance(EVENT_CHANNEL, str) and EVENT_CHANNEL:
            return EVENT_CHANNEL
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        return ws_channels.event_channel() if hasattr(ws_channels, "event_channel") else DEFAULT_CHANNEL
    except Exception:  # pragma: no cover
        return DEFAULT_CHANNEL


def _session_factory() -> Optional[Callable[..., Any]]:
    """Resolve an async session factory suitable for ``async with factory()``."""
    try:
        from app.db import get_session_factory

        factory = get_session_factory
        if not callable(factory):
            return None
        resolved = factory()
        if asyncio.iscoroutine(resolved):  # pragma: no cover - async factory
            raise RuntimeError("async session factory must be synchronous")
        return resolved
    except Exception:
        pass
    try:
        from app.db import async_session_factory as factory  # noqa: F401

        return factory
    except Exception:
        pass
    try:
        from app.db import SessionLocal as factory  # noqa: F401

        return factory
    except Exception:
        return None


# ---------------------------------------------------------------------------
# singleton accessors used by the API lifespan
# ---------------------------------------------------------------------------
_listener: Optional[NotifyListener] = None
_listener_lock: Optional[asyncio.Lock] = None


def _lock() -> asyncio.Lock:
    global _listener_lock
    if _listener_lock is None:
        _listener_lock = asyncio.Lock()
    return _listener_lock


def get_listener() -> NotifyListener:
    """Return the process-wide listener (constructing it lazily)."""
    global _listener
    if _listener is None:
        _listener = NotifyListener()
    return _listener


async def start_listener(**kwargs: Any) -> NotifyListener:
    """Start the singleton listener; safe to call from a lifespan handler."""
    async with _lock():
        listener = get_listener()
        if kwargs:
            listener = NotifyListener(**kwargs)
            globals()["_listener"] = listener
        await listener.start()
        return listener


async def stop_listener() -> None:
    """Stop the singleton listener if it was started."""
    global _listener
    listener = _listener
    if listener is None:
        return
    await listener.stop()


async def publish(event: Any) -> int:
    """Publish one already-committed event row through the singleton listener."""
    return await get_listener().publish_event(event)
