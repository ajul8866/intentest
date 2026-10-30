"""In-process WebSocket hub for the Intentest event fan-out plane.

Architecture
------------
::

    domain write ──> event row ──> pg_notify('intentest_events')
                                        │
                                 notify_listener
                                        │
                                  hub.publish(channel, frame)
                                        │
                          ┌─────────────┴─────────────┐
                     WebSocket A                WebSocket B
                   (?channel=project:x)      (?channel=global)

The hub is *schema agnostic*: it never invents payloads.  Every frame it
delivers originates from a committed ``event`` row (see ``notify_listener``),
so a browser that reloads and re-reads the REST API observes exactly the same
state that the WebSocket stream reported.  The hub is therefore a fan-out
mechanism, not a source of truth.

Wire format (contract shared with the frontend)::

    {
      "type": "fact.created",
      "channel": "project:6f1c...",
      "data": {...},
      "event_id": 1234,          # null when synthesised locally
      "ts": "2025-01-01T12:00:00+00:00"
    }

Design notes
------------
* **Backpressure** -- each client owns a bounded queue (``WS_MAX_QUEUE``).
  When the queue is full the client is dropped rather than blocking the
  publisher (a slow browser must never stall the scheduler).  Dropped slow
  consumers are counted in :attr:`HubStats.dropped_slow_clients`.
* **Isolation** -- a client only receives frames whose channel matches its
  subscription via :func:`app.api.ws.channels.topic_matches`.  There is no
  wildcard ``project:*`` (see ``channels.py``), so a typo cannot leak another
  project's facts.
* **No circular imports** -- ``channels.py`` is pure/IO-free and is imported
  at module load; FastAPI's ``WebSocket`` is imported lazily inside the
  connection helpers so importing the hub never requires a running event loop
  or the ASGI stack.
* **Testability** -- :meth:`WebSocketHub.publish` is synchronous (it only
  appends to in-memory queues plus schedules an ``await`` via
  ``loop.call_soon``), so unit tests can assert delivery without a socket.

Public interface
----------------
``get_hub()``                 process-wide singleton (used by ``api/deps.py``)
``WebSocketHub``              the registry + router
``WebSocketClient``           one connected socket + its queue
``HubStats``                  counters surfaced on ``/healthz/checks``
``publish_event(event)``      convenience helper used by services
``shutdown_hub()``            close everything (app lifespan)
``WS_CLOSE_*``                close codes shared with the /ws endpoint
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from app.api.ws import channels as ch

logger = logging.getLogger(__name__)

__all__ = [
    "WebSocketHub",
    "WebSocketClient",
    "HubStats",
    "get_hub",
    "set_hub",
    "reset_hub",
    "publish_event",
    "shutdown_hub",
    "make_frame",
    "WS_CLOSE_GOING_AWAY",
    "WS_CLOSE_INTERNAL",
    "WS_CLOSE_INVALID_CHANNEL",
    "WS_CLOSE_QUEUE_OVERFLOW",
    "DEFAULT_PING_INTERVAL_S",
    "DEFAULT_MAX_QUEUE",
    "DEFAULT_REPLAY_LIMIT",
]

# --------------------------------------------------------------------------
# Constants (kept in sync with app/api/ws/__init__.py)
# --------------------------------------------------------------------------
DEFAULT_PING_INTERVAL_S: float = 25.0
DEFAULT_MAX_QUEUE: int = 512
DEFAULT_REPLAY_LIMIT: int = 500

# Custom application close codes (4400-4599 is the private-use range).
WS_CLOSE_GOING_AWAY: int = 1001
WS_CLOSE_INTERNAL: int = 1011
WS_CLOSE_INVALID_CHANNEL: int = 4400
WS_CLOSE_QUEUE_OVERFLOW: int = 4408


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _iso(value: Any = None) -> str:
    """ISO-8601 UTC timestamp for the ``ts`` field."""
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe projection.

    Handles UUIDs, datetimes, enums and nested containers.  Anything that is
    still not serialisable is stringified by :func:`make_frame` as a last
    resort so one bad field can never kill a live stream.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, Deque)):
        return [_jsonable(v) for v in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    if hasattr(value, "isoformat"):
        with contextlib.suppress(Exception):
            return value.isoformat()
    return str(value)


def make_frame(
    event_type: str,
    channel: str,
    data: Optional[Mapping[str, Any]] = None,
    *,
    event_id: Optional[int] = None,
    ts: Any = None,
) -> Dict[str, Any]:
    """Build the canonical wire frame documented in ``app/api/ws/__init__.py``."""
    return {
        "type": str(event_type or "message"),
        "channel": str(channel or ch.GLOBAL),
        "data": _jsonable(dict(data or {})),
        "event_id": event_id,
        "ts": _iso(ts),
    }


# --------------------------------------------------------------------------
# Client + stats
# --------------------------------------------------------------------------
@dataclass
class HubStats:
    """Aggregate counters exposed on ``/healthz/checks`` and the dashboard."""

    connections: int = 0
    frames_sent: int = 0
    frames_dropped: int = 0
    events_received: int = 0
    connections_total: int = 0
    dropped_slow_clients: int = 0
    publish_errors: int = 0
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connections": self.connections,
            "connections_total": self.connections_total,
            "frames_sent": self.frames_sent,
            "frames_dropped": self.frames_dropped,
            "events_received": self.events_received,
            "dropped_slow_clients": self.dropped_slow_clients,
            "publish_errors": self.publish_errors,
            "uptime_s": round(time.time() - self.started_at, 3),
        }


class WebSocketClient:
    """One connected browser socket and its bounded outbound queue.

    Parameters
    ----------
    websocket:
        The Starlette/FastAPI ``WebSocket`` (duck-typed: ``accept``,
        ``send_text``, ``receive_text``, ``close``).  ``None`` is legal for
        unit tests that only exercise the hub.
    channel:
        Canonical channel string from :func:`channels.parse_channel`.
    max_queue:
        Queue depth before the client is dropped (backpressure guard).
    """

    __slots__ = (
        "websocket",
        "channel",
        "client_id",
        "topics",
        "queue",
        "max_queue",
        "connected_at",
        "last_sent_at",
        "closed",
        "_writer_task",
        "_send_lock",
    )

    def __init__(
        self,
        websocket: Any = None,
        channel: str = ch.GLOBAL,
        *,
        max_queue: int = DEFAULT_MAX_QUEUE,
        client_id: Optional[str] = None,
        topics: Optional[Iterable[str]] = None,
    ) -> None:
        self.websocket = websocket
        self.channel = channel
        self.client_id = client_id or uuid.uuid4().hex
        self.topics: Set[str] = set(topics or ())
        self.topics.add(channel)
        self.max_queue = max(1, int(max_queue))
        self.queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=self.max_queue)
        self.connected_at = time.time()
        self.last_sent_at: Optional[float] = None
        self.closed = False
        self._writer_task: Optional[asyncio.Task] = None
        # Serialises socket writes so a ping can never interleave with a frame.
        self._send_lock = asyncio.Lock()

    # -- introspection ----------------------------------------------------
    @property
    def kind(self) -> str:
        try:
            return ch.parse_channel(self.channel).kind
        except Exception:  # pragma: no cover - defensive
            return "unknown"

    def matches(self, topic: str) -> bool:
        """True when a frame published on ``topic`` belongs to this client."""
        if self.closed:
            return False
        for subscription in self.topics:
            if ch.topic_matches(subscription, topic):
                return True
        return False

    def qsize(self) -> int:
        return self.queue.qsize()

    # -- delivery ---------------------------------------------------------
    def offer(self, frame: Dict[str, Any]) -> bool:
        """Enqueue ``frame`` without blocking.  ``False`` == queue overflow."""
        if self.closed:
            return False
        try:
            self.queue.put_nowait(frame)
            return True
        except asyncio.QueueFull:
            return False

    async def send_now(self, frame: Dict[str, Any]) -> bool:
        """Direct write (used for replay frames before the writer task starts)."""
        if self.websocket is None:
            return False
        payload = json.dumps(_jsonable(frame), separators=(",", ":"), default=str)
        async with self._send_lock:
            try:
                await self.websocket.send_text(payload)
                self.last_sent_at = time.time()
                return True
            except Exception as exc:  # socket already gone
                logger.debug("ws.send_failed", extra={"client_id": self.client_id, "error": str(exc)})
                self.closed = True
                return False

    async def send_json(self, payload: Mapping[str, Any]) -> bool:
        return await self.send_now(dict(payload))

    async def writer_loop(self) -> None:
        """Drain the queue onto the socket until the client closes."""
        while not self.closed:
            try:
                frame = await self.queue.get()
            except asyncio.CancelledError:  # pragma: no cover - shutdown
                break
            if frame is None:  # sentinel
                break
            ok = await self.send_now(frame)
            if not ok:
                break
        self.closed = True

    def start_writer(self) -> Optional["asyncio.Task"]:
        if self._writer_task is None and self.websocket is not None:
            with contextlib.suppress(RuntimeError):
                self._writer_task = asyncio.create_task(
                    self.writer_loop(), name=f"ws-writer-{self.client_id}"
                )
        return self._writer_task

    async def stop(self, code: int = WS_CLOSE_GOING_AWAY, reason: str = "") -> None:
        self.closed = True
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(None)  # type: ignore[arg-type]
        if self._writer_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._writer_task, timeout=1.0)
            self._writer_task = None
        if self.websocket is not None:
            with contextlib.suppress(Exception):
                await self.websocket.close(code=code, reason=reason)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "channel": self.channel,
            "kind": self.kind,
            "topics": sorted(self.topics),
            "queued": self.qsize(),
            "max_queue": self.max_queue,
            "connected_at": _iso(datetime.fromtimestamp(self.connected_at, tz=timezone.utc)),
            "last_sent_at": (
                _iso(datetime.fromtimestamp(self.last_sent_at, tz=timezone.utc))
                if self.last_sent_at
                else None
            ),
            "closed": self.closed,
        }


# --------------------------------------------------------------------------
# Hub
# --------------------------------------------------------------------------
class WebSocketHub:
    """Registry of live WebSocket clients plus channel-based routing.

    The hub is intentionally synchronous on the publish path
    (:meth:`publish`) so callers in async or sync code (services, scheduler,
    tests) can fire events without awaiting.  Delivery happens in each
    client's writer task.
    """

    def __init__(
        self,
        *,
        max_queue: int = DEFAULT_MAX_QUEUE,
        ping_interval_s: float = DEFAULT_PING_INTERVAL_S,
        replay_limit: int = DEFAULT_REPLAY_LIMIT,
    ) -> None:
        self.max_queue = max_queue
        self.ping_interval_s = ping_interval_s
        self.replay_limit = replay_limit
        self._clients: Dict[str, WebSocketClient] = {}
        self._by_channel: Dict[str, Set[str]] = {}
        self.stats = HubStats()
        self._lock = asyncio.Lock()
        self._ping_task: Optional["asyncio.Task"] = None

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        """Start the keep-alive ping task (idempotent)."""
        if self._ping_task is not None and not self._ping_task.done():
            return
        with contextlib.suppress(RuntimeError):
            self._ping_task = asyncio.create_task(self._ping_loop(), name="ws-hub-ping")

    async def shutdown(self) -> None:
        """Close every socket and stop background tasks (app lifespan)."""
        if self._ping_task is not None:
            self._ping_task.cancel()
            with contextlib.suppress(Exception):
                await self._ping_task
            self._ping_task = None
        for client in list(self._clients.values()):
            with contextlib.suppress(Exception):
                await client.stop(code=WS_CLOSE_GOING_AWAY, reason="server shutdown")
        self._clients.clear()
        self._by_channel.clear()
        self.stats.connections = 0

    async def _ping_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.ping_interval_s)
                frame = make_frame("heartbeat", ch.GLOBAL, {"clients": len(self._clients)})
                for client in list(self._clients.values()):
                    # only global subscribers get the hub heartbeat; project
                    # subscribers receive per-task heartbeats via events.
                    if client.matches(ch.GLOBAL):
                        client.offer(frame)
            except asyncio.CancelledError:  # pragma: no cover - shutdown
                break
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("ws.ping_loop_error", extra={"error": str(exc)})

    # -- registration -----------------------------------------------------
    def _index(self, client: WebSocketClient) -> None:
        for topic in client.topics:
            self._by_channel.setdefault(topic, set()).add(client.client_id)

    def _unindex(self, client: WebSocketClient) -> None:
        for topic in list(client.topics):
            bucket = self._by_channel.get(topic)
            if bucket:
                bucket.discard(client.client_id)
                if not bucket:
                    self._by_channel.pop(topic, None)

    def register(self, client: WebSocketClient) -> WebSocketClient:
        self._clients[client.client_id] = client
        self._index(client)
        self.stats.connections = len(self._clients)
        self.stats.connections_total += 1
        logger.info(
            "ws.client_connected",
            extra={"client_id": client.client_id, "channel": client.channel, "total": len(self._clients)},
        )
        return client

    # alias used by some call sites
    add = register

    def unregister(self, client_or_id: Any) -> None:
        client_id = (
            client_or_id.client_id
            if isinstance(client_or_id, WebSocketClient)
            else str(client_or_id)
        )
        client = self._clients.pop(client_id, None)
        if client is not None:
            client.closed = True
            self._unindex(client)
            logger.info(
                "ws.client_disconnected",
                extra={"client_id": client_id, "channel": client.channel, "total": len(self._clients)},
            )
        self.stats.connections = len(self._clients)

    remove = unregister

    def get_client(self, client_id: str) -> Optional[WebSocketClient]:
        return self._clients.get(client_id)

    def clients(self) -> List[WebSocketClient]:
        return list(self._clients.values())

    def subscriber_count(self, channel: Optional[str] = None) -> int:
        if channel is None:
            return len(self._clients)
        return len(self._by_channel.get(channel, ()))

    # -- publishing -------------------------------------------------------
    def publish(self, channel: str, frame: Mapping[str, Any]) -> int:
        """Fan ``frame`` out to every client subscribed to ``channel``.

        Returns the number of clients the frame was queued for.  Never
        raises: a failing client is dropped, not propagated.
        """
        payload = dict(frame)
        payload.setdefault("channel", channel)
        payload.setdefault("ts", _iso())
        self.stats.events_received += 1

        delivered = 0
        targets: List[WebSocketClient] = []
        # Fast path: direct channel index, then verify with topic_matches so
        # wildcard subscriptions (worker:*) are honoured.
        seen: Set[str] = set()
        for client_id in list(self._by_channel.get(channel, ())):
            c = self._clients.get(client_id)
            if c is not None and client_id not in seen:
                seen.add(client_id)
                targets.append(c)
        for client in list(self._clients.values()):
            if client.client_id in seen:
                continue
            if client.matches(channel):
                targets.append(client)

        for client in targets:
            if client.offer(payload):
                delivered += 1
                self.stats.frames_sent += 1
            else:
                self.stats.frames_dropped += 1
                self.stats.dropped_slow_clients += 1
                logger.warning(
                    "ws.client_queue_overflow",
                    extra={"client_id": client.client_id, "channel": client.channel},
                )
                # Drop the slow consumer; the browser will reconnect with
                # ?since=<last_event_id> and replay from Postgres.
                self.unregister(client)
        return delivered

    # -- event convenience -------------------------------------------------
    def publish_event(self, event: Any, *, frame: Optional[Mapping[str, Any]] = None) -> int:
        """Route a persisted ``event`` row to all matching channels.

        Uses :func:`app.api.ws.channels.channels_for_event` so the routing
        rules live in exactly one place (shared with the NOTIFY listener).
        """
        try:
            targets = ch.channels_for_event(event)
        except Exception as exc:  # pragma: no cover - defensive
            self.stats.publish_errors += 1
            logger.warning("ws.routing_failed", extra={"error": str(exc)})
            return 0
        if not targets:
            return 0
        built = dict(frame) if frame is not None else _frame_from_event(event)
        delivered = 0
        for channel in targets:
            delivered += self.publish(channel, {**built, "channel": channel})
        return delivered

    def broadcast(self, frame: Mapping[str, Any], *, channel: str = ch.GLOBAL) -> int:
        return self.publish(channel, frame)

    # -- introspection -----------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "stats": self.stats.to_dict(),
            "channels": {name: len(ids) for name, ids in self._by_channel.items()},
            "clients": [c.to_dict() for c in self._clients.values()],
        }

    def stats_dict(self) -> Dict[str, Any]:
        return self.stats.to_dict()

    def __len__(self) -> int:  # convenience for tests
        return len(self._clients)

    def __contains__(self, client_id: object) -> bool:
        return str(client_id) in self._clients


# --------------------------------------------------------------------------
# Frame construction from an event row
# --------------------------------------------------------------------------
def _frame_from_event(event: Any) -> Dict[str, Any]:
    """Project an ``event`` ORM row / mapping into the wire frame."""

    def get(*names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(event, Mapping) and name in event:
                return event[name]
            value = getattr(event, name, None)
            if value is not None:
                return value
        return default

    event_type = get("event_type", "type", "name", default="message") or "message"
    event_id = get("id", "event_id")
    try:
        event_id_int = int(event_id) if event_id is not None else None
    except (TypeError, ValueError):
        event_id_int = None

    data = get("payload", "data", "body")
    if data is None:
        data = {}
        for key in (
            "project_id",
            "worker_id",
            "experiment_id",
            "intent_id",
            "fact_id",
            "task_run_id",
            "message",
        ):
            value = get(key)
            if value is not None:
                data[key] = value
    if not isinstance(data, Mapping):
        data = {"value": data}
    data = dict(data)
    # Never leak the routing key twice; keep it available in `data` only when
    # the emitter explicitly put it there.
    return make_frame(
        str(event_type),
        ch.GLOBAL,  # replaced per-channel by publish_event
        data,
        event_id=event_id_int,
        ts=get("created_at", "ts", "timestamp"),
    )


# --------------------------------------------------------------------------
# Process-wide singleton
# --------------------------------------------------------------------------
_hub: Optional[WebSocketHub] = None


def get_hub() -> WebSocketHub:
    """Return the process-wide hub, creating it on first use."""
    global _hub
    if _hub is None:
        # Honour the configured queue limit when config is importable; the
        # ws package constants are the documented source of truth.
        max_queue = DEFAULT_MAX_QUEUE
        ping_interval = DEFAULT_PING_INTERVAL_S
        replay_limit = DEFAULT_REPLAY_LIMIT
        try:  # pragma: no cover - config is optional for unit tests
            from app.api.ws import WS_MAX_QUEUE, WS_MAX_REPLAY, WS_PING_INTERVAL_S

            max_queue = int(WS_MAX_QUEUE)
            ping_interval = float(WS_PING_INTERVAL_S)
            replay_limit = int(WS_MAX_REPLAY)
        except Exception:
            pass
        _hub = WebSocketHub(
            max_queue=max_queue,
            ping_interval_s=ping_interval,
            replay_limit=replay_limit,
        )
    return _hub


def set_hub(hub: Optional[WebSocketHub]) -> Optional[WebSocketHub]:
    """Install a hub instance (tests / lifespan)."""
    global _hub
    _hub = hub
    return _hub


def reset_hub() -> None:
    """Drop the singleton so the next :func:`get_hub` creates a fresh one."""
    global _hub
    _hub = None


async def shutdown_hub() -> None:
    """Close the singleton hub, if any (called from app lifespan)."""
    global _hub
    if _hub is not None:
        with contextlib.suppress(Exception):
            await _hub.shutdown()
        _hub = None


def publish_event(event: Any, *, hub: Optional[WebSocketHub] = None) -> int:
    """Module-level convenience wrapper around ``hub.publish_event``."""
    return (hub or get_hub()).publish_event(event)


# --------------------------------------------------------------------------
# Connection helpers used by the /ws endpoint
# --------------------------------------------------------------------------
async def connect(
    websocket: Any,
    channel: str = ch.GLOBAL,
    *,
    hub: Optional[WebSocketHub] = None,
    subprotocols: Optional[Sequence[str]] = None,
) -> WebSocketClient:
    """Accept ``websocket``, register it on ``channel`` and start its writer."""
    hub = hub or get_hub()
    try:
        await websocket.accept(subprotocol=(subprotocols[0] if subprotocols else None))
    except TypeError:  # pragma: no cover - older Starlette signature
        await websocket.accept()
    client = WebSocketClient(websocket, channel, max_queue=hub.max_queue)
    hub.register(client)
    client.start_writer()
    return client


async def disconnect(client: WebSocketClient, *, hub: Optional[WebSocketHub] = None) -> None:
    hub = hub or get_hub()
    hub.unregister(client)
    with contextlib.suppress(Exception):
        await client.stop()


async def replay(
    client: WebSocketClient,
    events: Iterable[Any],
    *,
    limit: Optional[int] = None,
) -> int:
    """Send replayed events (``?since=<event_id>``) directly to one client."""
    sent = 0
    cap = limit if limit is not None else DEFAULT_REPLAY_LIMIT
    for event in events:
        if sent >= cap:
            break
        frame = _frame_from_event(event)
        frame["replay"] = True
        if await client.send_now(frame):
            sent += 1
    return sent
