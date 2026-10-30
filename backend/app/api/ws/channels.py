"""WebSocket channel vocabulary and routing rules for the Intentest event bus.

Clients subscribe to one *channel* at a time::

    /ws?channel=project:<uuid>      # everything that happened in one project
    /ws?channel=worker:<name|id>    # worker lifecycle / leases
    /ws?channel=global              # scheduler-wide backpressure, experiments
    /ws?channel=experiments         # Table 1/2/3 progress across all experiments

This module owns three things and nothing else:

1. **Naming** -- :func:`project_channel`, :func:`worker_channel`, ... produce the
   canonical strings. ``app.api.deps.get_hub`` and the FastAPI endpoint in
   ``app/main.py`` both come through here so the format is never duplicated as a
   literal (see ``app/api/__init__.py`` for the same rule about
   ``EVENT_CHANNEL``).

2. **Parsing/validation** -- :func:`parse_channel` and :func:`normalise_channel`
   turn user input into a :class:`Channel` (kind + subject) or raise
   :class:`ChannelError`. A channel whose subject is not a UUID is *rejected*
   rather than silently served the global firehose, because leaking another
   project's facts across a typo is a real (if minor) information-disclosure
   bug in this threat model.

3. **Fan-out routing** -- :func:`channels_for_event` maps one persisted ``event``
   row (``event_type`` + ``project_id`` / ``worker_id`` / ``experiment_id``
   payload fields) to the set of channels that must receive it. The listener
   (``notify_listener.py``) calls this for every NOTIFY payload, then hands the
   resulting pairs to the in-memory hub. Routing lives here (pure function, no
   I/O) so it is unit-testable without Postgres or a socket.

Event types are the ones fixed by the reproduction plan §C13::

    fact.created              fact.verified           intent.created
    intent.state_changed      task.started            task.log
    task.finished             task.progress           worker.status
    worker.registered         project.status       project.phase
    project.round            retrieval.result        bridge.injected
    bridge.dropped            degradation.triggered   degradation.recovered
    scheduler.backpressure    scheduler.tick          prompt.created
    prompt.consumed           graph.updated           experiment.status
    experiment.task_finished experiment.completed    integrity.violation

Every event additionally fans out to ``global`` *only* if it is listed in
:data:`GLOBAL_EVENT_TYPES`; everything else stays scoped to its project (or
worker) channel so a busy project cannot drown the dashboard of an idle one.
``scheduler.tick`` is deliberately **not** global: the tick emits it every 2 s
and it would otherwise dominate the socket.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    # channel kinds / constants
    "GLOBAL",
    "EXPERIMENTS",
    "PROJECT_PREFIX",
    "WORKER_PREFIX",
    "EXPERIMENT_PREFIX",
    "CHANNEL_KINDS",
    "MAX_CHANNEL_LENGTH",
    # event vocabulary
    "EVENT_TYPES",
    "PROJECT_EVENT_TYPES",
    "WORKER_EVENT_TYPES",
    "EXPERIMENT_EVENT_TYPES",
    "GLOBAL_EVENT_TYPES",
    "TERMINAL_EVENT_TYPES",
    # naming
    "project_channel",
    "worker_channel",
    "experiment_channel",
    "Channel",
    "ChannelError",
    "parse_channel",
    "normalise_channel",
    "is_project_channel",
    "channel_subject",
    # routing
    "channels_for_event",
    "event_project_id",
    "event_worker_id",
    "event_experiment_id",
    "should_replay",
]

# --------------------------------------------------------------------------- #
# Naming constants
# --------------------------------------------------------------------------- #

GLOBAL = "global"
EXPERIMENTS = "experiments"
PROJECT_PREFIX = "project:"
WORKER_PREFIX = "worker:"
EXPERIMENT_PREFIX = "experiment:"

CHANNEL_KINDS: tuple[str, ...] = ("project", "worker", "experiment", "global", "experiments")

#: Hard cap so a hostile/typo'd ``?channel=`` cannot blow up memory keys.
MAX_CHANNEL_LENGTH = 200


# --------------------------------------------------------------------------- #
# Event vocabulary
# --------------------------------------------------------------------------- #

#: Every ``event_type`` the backend is allowed to publish. The frontend's typed
#: client is generated from this list; publishing an unlisted type is a bug and
#: is logged as ``ws.unknown_event_type`` by the listener (it is still delivered
#: to the owning project channel so nothing is silently dropped).
EVENT_TYPES: tuple[str, ...] = (
    # fact / intent graph
    "fact.created",
    "fact.verified",
    "fact.superseded",
    "intent.created",
    "intent.state_changed",
    "intent.injected",
    "graph.updated",
    "integrity.violation",
    # tasks
    "task.started",
    "task.progress",
    "task.log",
    "task.finished",
    # workers
    "worker.registered",
    "worker.status",
    "worker.heartbeat",
    "worker.drained",
    # project lifecycle
    "project.created",
    "project.status",
    "project.phase",
    "project.round",
    "project.terminated",
    # knowledge plane (Algorithm 2 + Algorithm 3)
    "retrieval.result",
    "bridge.injected",
    "bridge.dropped",
    # two-phase degradation
    "degradation.triggered",
    "degradation.recovered",
    # scheduler
    "scheduler.backpressure",
    "scheduler.tick",
    "scheduler.lease",
    # expert hints (Cairn ``Hint``)
    "prompt.created",
    "prompt.consumed",
    # experiments / evaluation harness
    "experiment.created",
    "experiment.status",
    "experiment.task_finished",
    "experiment.completed",
)

#: Events that belong to a single project. These are the types that require a
#: ``project:<uuid>`` channel to be delivered at all.
PROJECT_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "fact.created",
        "fact.verified",
        "fact.superseded",
        "intent.created",
        "intent.state_changed",
        "intent.injected",
        "graph.updated",
        "integrity.violation",
        "task.started",
        "task.progress",
        "task.log",
        "task.finished",
        "project.created",
        "project.status",
        "project.phase",
        "project.round",
        "project.terminated",
        "retrieval.result",
        "bridge.injected",
        "bridge.dropped",
        "degradation.triggered",
        "degradation.recovered",
        "prompt.created",
        "prompt.consumed",
    }
)

#: Worker-scoped events (fleet view). They carry ``worker_id`` and are delivered
#: to ``worker:<id>`` and to the shared ``worker:*`` topic, never to a project.
WORKER_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "worker.registered",
        "worker.status",
        "worker.heartbeat",
        "worker.drained",
        "scheduler.backpressure",
        "scheduler.lease",
    }
)

#: Experiment-scoped events (evaluation harness progress).
EXPERIMENT_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "experiment.created",
        "experiment.status",
        "experiment.task_finished",
        "experiment.completed",
    }
)

#: Types mirrored to ``global`` in addition to their own channel. Kept tiny on
#: purpose: these are the "something changed for everyone" signals a dashboard
#: needs, not per-fact chatter.
GLOBAL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "scheduler.backpressure",
        "worker.registered",
        "worker.status",
        "worker.drained",
        "experiment.status",
        "experiment.completed",
        "project.status",
        "project.terminated",
        "integrity.violation",
    }
)

#: Event types after which a client may close its socket / stop replaying.
TERMINAL_EVENT_TYPES: frozenset[str] = frozenset(
    {"project.terminated", "experiment.completed"}
)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ChannelError(ValueError):
    """Raised when a channel string cannot be parsed.

    The WS endpoint catches this and closes the socket with code ``4400`` and a
    human-readable reason (browsers cannot read a 4xx body after the upgrade, so
    the reason *is* the message).
    """

    def __init__(self, message: str, *, channel: Optional[str] = None, code: str = "INVALID_CHANNEL") -> None:
        super().__init__(message)
        self.channel = channel
        self.code = code

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "channel": self.channel}


# --------------------------------------------------------------------------- #
# Channel value object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Channel:
    """A parsed subscription target.

    ``kind`` is one of :data:`CHANNEL_KINDS`; ``subject`` is the UUID/name that
    follows the prefix (empty for ``global`` / ``experiments``). ``raw`` keeps
    the exact string the client sent so logs and ``X-Channel`` echoes are
    faithful.
    """

    kind: str
    subject: str = ""
    raw: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.raw or self.canonical()

    @property
    def is_project(self) -> bool:
        return self.kind == "project"

    @property
    def is_worker(self) -> bool:
        return self.kind == "worker"

    @property
    def is_global(self) -> bool:
        return self.kind in ("global", "experiments")

    @property
    def project_id(self) -> Optional[str]:
        return self.subject if self.kind == "project" else None

    def canonical(self) -> str:
        """Return the canonical channel string (lower-cased subject)."""
        if self.kind == "project":
            return PROJECT_PREFIX + self.subject.lower()
        if self.kind == "worker":
            return WORKER_PREFIX + self.subject
        if self.kind == "experiment":
            return EXPERIMENT_PREFIX + self.subject.lower()
        if self.kind == "experiments":
            return EXPERIMENTS
        return GLOBAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.canonical(),
            "kind": self.kind,
            "subject": self.subject,
        }


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #


def _as_subject(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, uuid.UUID):
        return str(value)
    return str(value).strip()


def project_channel(project_id: Any) -> str:
    """Return ``project:<uuid>`` for a project id (str or UUID)."""
    subject = _as_subject(project_id)
    if not subject:
        raise ChannelError("project id is required to build a project channel", channel=PROJECT_PREFIX)
    return PROJECT_PREFIX + subject.lower()


def worker_channel(worker: Any) -> str:
    """Return ``worker:<name|id>`` for a worker row, id, or name."""
    subject = _as_subject(worker)
    if not subject:
        raise ChannelError("worker id/name is required to build a worker channel", channel=WORKER_PREFIX)
    return WORKER_PREFIX + subject


def experiment_channel(experiment_id: Any) -> str:
    subject = _as_subject(experiment_id)
    if not subject:
        raise ChannelError(
            "experiment id is required to build an experiment channel",
            channel=EXPERIMENT_PREFIX,
        )
    return EXPERIMENT_PREFIX + subject.lower()


# --------------------------------------------------------------------------- #
# Parsing / normalisation
# --------------------------------------------------------------------------- #


def _looks_like_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def parse_channel(channel: Optional[str] = None, *, project_id: Any = None) -> Channel:
    """Parse a client-supplied channel string.

    Accepted forms (case-insensitive prefix, subject case preserved except for
    UUIDs where lower-case is canonicalised):

    * ``global`` / ``all`` / ``*`` / ``""``      -> global firehose
    * ``experiments``                            -> evaluation harness
    * ``project:<uuid>``                         -> one project
    * ``worker:<name-or-uuid>``                  -> one worker
    * ``experiment:<uuid>``                      -> one experiment

    ``project_id`` is the convenience alias used by ``?project_id=`` on the WS
    query string; it is consulted only when ``channel`` is falsy.

    Raises :class:`ChannelError` for anything malformed. Notably ``project:abc``
    is rejected: it is far more likely a typo than an intent to subscribe to a
    non-existent project, and the hub keys genuinely are UUIDs.
    """

    raw = "" if channel is None else str(channel).strip()

    if not raw:
        if project_id:
            return parse_channel(project_channel(project_id))
        return Channel(kind="global", subject="", raw=GLOBAL)

    if len(raw) > MAX_CHANNEL_LENGTH:
        raise ChannelError(
            f"channel is too long ({len(raw)} > {MAX_CHANNEL_LENGTH})", channel=raw[:64]
        )

    lowered = raw.lower()

    if lowered in ("global", "all", "*", "broadcast"):
        return Channel(kind="global", subject="", raw=GLOBAL)
    if lowered in ("experiments", "experiment", "evals", "evaluation"):
        # ``experiment`` alone is ambiguous with the ``experiment:<id>`` prefix;
        # treat the bare word as the shared harness channel, which is what the
        # frontend's Experiments table subscribes to.
        return Channel(kind="experiments", subject="", raw=EXPERIMENTS)

    if ":" not in lowered:
        # Bare uuid -> project channel (handy for curl-based debugging).
        if _looks_like_uuid(raw):
            return Channel(kind="project", subject=raw.lower(), raw=PROJECT_PREFIX + raw.lower())
        raise ChannelError(
            "channel must be 'global', 'experiments', or '<kind>:<subject>' "
            "with kind in {project, worker, experiment}",
            channel=raw,
        )

    kind, _, subject = lowered.partition(":")
    subject = subject.strip()
    original_subject = raw.partition(":")[2].strip()

    if not subject:
        raise ChannelError(f"channel {raw!r} is missing its subject", channel=raw)

    if kind in ("project", "projects", "proj"):
        if not _looks_like_uuid(subject):
            raise ChannelError(
                f"project channel subject must be a UUID, got {subject!r}", channel=raw
            )
        return Channel(kind="project", subject=subject, raw=PROJECT_PREFIX + subject)

    if kind in ("worker", "workers"):
        return Channel(kind="worker", subject=original_subject, raw=WORKER_PREFIX + original_subject)

    if kind in ("experiment", "experiments"):
        if not _looks_like_uuid(subject):
            raise ChannelError(
                f"experiment channel subject must be a UUID, got {subject!r}", channel=raw
            )
        return Channel(kind="experiment", subject=subject, raw=EXPERIMENT_PREFIX + subject)

    raise ChannelError(
        f"unknown channel kind {kind!r}; expected one of {sorted(CHANNEL_KINDS)}", channel=raw
    )


def normalise_channel(channel: Optional[str] = None, *, project_id: Any = None) -> str:
    """Return the canonical channel string, or ``"global"`` on empty input.

    Unlike :func:`parse_channel` this never raises for the *empty* case (it is
    what ``app.api.deps.get_channel_name`` and the tests use); malformed
    non-empty input still raises :class:`ChannelError`.
    """

    return parse_channel(channel, project_id=project_id).canonical()


def is_project_channel(channel: str) -> bool:
    try:
        return parse_channel(channel).kind == "project"
    except ChannelError:
        return False


def channel_subject(channel: str) -> str:
    """Return the subject (uuid/name) of a channel, or ``""`` for global."""
    try:
        return parse_channel(channel).subject
    except ChannelError:
        return ""


# --------------------------------------------------------------------------- #
# Event routing
# --------------------------------------------------------------------------- #


def _get(event: Any, *names: str) -> Any:
    """Tolerant read of an event row / NOTIFY dict (ORM, mapping, namespace)."""
    for name in names:
        if isinstance(event, Mapping):
            if name in event and event[name] is not None:
                return event[name]
        else:
            value = getattr(event, name, None)
            if value is not None:
                return value
    return None


def event_project_id(event: Any) -> Optional[str]:
    """Extract the project id an event belongs to (``None`` if unscoped).

    Checks the dedicated column first, then the ``payload``/``data`` blob that
    the ``event`` table stores, because workers occasionally publish project
    facts with only ``{"project_id": ...}`` in the payload.
    """

    value = _get(event, "project_id")
    if value is None:
        payload = _get(event, "payload", "data", "meta")
        if isinstance(payload, Mapping):
            value = payload.get("project_id") or payload.get("projectId")
    if value is None:
        return None
    return _as_subject(value).lower() or None


def event_worker_id(event: Any) -> Optional[str]:
    value = _get(event, "worker_id")
    if value is None:
        payload = _get(event, "payload", "data", "meta")
        if isinstance(payload, Mapping):
            value = payload.get("worker_id") or payload.get("worker")
    if value is None:
        return None
    return _as_subject(value)


def event_experiment_id(event: Any) -> Optional[str]:
    value = _get(event, "experiment_id")
    if value is None:
        payload = _get(event, "payload", "data", "meta")
        if isinstance(payload, Mapping):
            value = payload.get("experiment_id") or payload.get("experiment")
    if value is None:
        return None
    return _as_subject(value).lower() or None


def channels_for_event(event: Any, *, event_type: Optional[str] = None) -> list[str]:
    """Return the ordered list of channels that must receive ``event``.

    Pure function -- no I/O -- so ``notify_listener`` can call it per NOTIFY and
    tests can assert routing without a hub. Ordering is "most specific first" so
    the hub can stop early if a subscriber is already satisfied (it does not, but
    the ordering keeps logs readable):

    1. ``project:<uuid>``   (if the event carries a project id)
    2. ``experiment:<uuid>``(if it carries an experiment id)
    3. ``worker:<id>``      (if it carries a worker id)
    4. ``experiments``      (for experiment-scoped types)
    5. ``global``           (only for the small allow-list)
    """

    etype = (event_type or str(_get(event, "event_type", "type", "name") or "")).strip().lower()

    channels: list[str] = []

    project_id = event_project_id(event)
    if project_id:
        channels.append(PROJECT_PREFIX + project_id)

    experiment_id = event_experiment_id(event)
    if experiment_id:
        channels.append(EXPERIMENT_PREFIX + experiment_id)

    worker_id = event_worker_id(event)
    if worker_id:
        channels.append(WORKER_PREFIX + worker_id)

    if etype in EXPERIMENT_EVENT_TYPES:
        channels.append(EXPERIMENTS)

    # A project event with no project id is unroutable except globally; that is
    # a publisher bug and the listener logs it (``ws.event_without_subject``).
    if etype in GLOBAL_EVENT_TYPES:
        channels.append(GLOBAL)

    # Deduplicate while preserving order (a single event may carry both a
    # project id and a worker id and match two rules).
    seen: set[str] = set()
    out: list[str] = []
    for channel in channels:
        key = channel.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(channel)
    return out


def should_replay(event: Any, *, since_event_id: Optional[int] = None) -> bool:
    """Whether an event should be replayed to a client asking for ``since``.

    ``since_event_id=None`` means "live only" -> never replay. Otherwise replay
    everything strictly after the cursor (``event.id`` is a BIGSERIAL, so this is
    a monotonic total order and exactly the resume semantics documented in
    ARCHITECTURE.md §Persistence & replay).
    """

    if since_event_id is None:
        return False
    event_id = _get(event, "id", "event_id")
    try:
        return int(event_id) > int(since_event_id)
    except (TypeError, ValueError):
        return False


def topic_matches(subscription: str, channel: str) -> bool:
    """Loose match used by the hub's shared-topic fan-out.

    A subscription ``worker:*`` receives every worker channel; ``project:*`` is
    intentionally *not* supported (it would expose every project's facts to any
    client, defeating the per-project isolation the channel design provides).
    ``global`` subscribers receive global plus ``experiments`` topics.
    """

    sub = (subscription or "").strip().lower()
    chan = (channel or "").strip().lower()
    if not sub:
        return False
    if sub == chan:
        return True
    if sub == "worker:*":
        return chan.startswith(WORKER_PREFIX)
    if sub == GLOBAL:
        return chan in (GLOBAL, EXPERIMENTS)
    return False


def iter_channels(channels: Iterable[Any]) -> Sequence[Channel]:
    """Parse a batch of channel strings, skipping (but not hiding) bad ones.

    Used by the hub for its admin/debug snapshot; invalid entries are kept out of
    the returned sequence and reported by the caller.
    """

    parsed: list[Channel] = []
    for channel in channels:
        try:
            parsed.append(parse_channel(channel))
        except ChannelError:
            continue
    return tuple(parsed)
