"""Intent finite-state machine (Definition 2 / Eq. 1).

Exactly four legal transitions exist:

    T1 (CREATED,  claim)    -> CLAIMED    CAS, exactly one winner, intent lease set
    T2 (CLAIMED,  conclude) -> CONCLUDED  guard: caller worker_id must match
    T3 (CONCLUDED, goal)    -> COMPLETED  terminal (I6)
    T4 (CLAIMED,  timeout)  -> CREATED    reaper only, now() > lease_expires_at

Everything else raises :class:`IllegalTransition`.

The module is usable in two ways:

* **Row-level (no DB)**: :func:`transition` / :func:`can` / :func:`apply_transition`
  operate on anything with ``state``, ``worker_id``, ``lease_expires_at``
  attributes or mapping keys (ORM rows, dataclasses, dicts, test stand-ins).
* **Database-level**: :class:`IntentFSM` performs the durable, race-free
  operations against Postgres (CAS ``UPDATE ... WHERE state='CREATED'`` with
  ``rowcount == 1``), plus the goal freeze and the lease reaper sweep.

The FSM never mutates ``intent_source`` rows (I2) and never removes facts (I1).
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from app.core.dag import ALLOWED_TRANSITIONS, INTENT_STATES, TERMINAL_INTENT_STATES

__all__ = [
    "IllegalTransition",
    "CLAIM",
    "CONCLUDE",
    "GOAL",
    "TIMEOUT",
    "LEGAL_TRANSITIONS",
    "TERMINAL_STATES",
    "ACTIVE_STATES",
    "state_of",
    "is_terminal",
    "is_active",
    "can",
    "transition",
    "legal_actions",
    "apply_transition",
    "IntentFSM",
    "goal_freeze",
    "reclaim_expired",
    "claim_cas_statement",
]

log = logging.getLogger(__name__)

# ── actions ────────────────────────────────────────────────────────────────
CLAIM = "claim"
CONCLUDE = "conclude"
GOAL = "goal"
TIMEOUT = "timeout"

#: Normative transition table copied from ``app.core.dag`` (Definition 2).
LEGAL_TRANSITIONS: dict[tuple[str, str], str] = {
    (str(frm).upper(), str(action).lower()): str(to).upper()
    for (frm, action), to in ALLOWED_TRANSITIONS.items()
}

STATES: tuple[str, ...] = tuple(str(s).upper() for s in INTENT_STATES)
TERMINAL_STATES: frozenset[str] = frozenset(str(s).upper() for s in TERMINAL_INTENT_STATES) or frozenset({"COMPLETED"})
#: States that may still produce facts (i.e. not terminal).
ACTIVE_STATES: frozenset[str] = frozenset({"CLAIMED"})


# ── errors ─────────────────────────────────────────────────────────────────
class IllegalTransition(Exception):
    """Raised when ``(from_state, action)`` is not one of the four legal moves."""

    def __init__(
        self,
        from_state: Any,
        action: Any,
        *,
        to_state: Any = None,
        intent_id: Any = None,
        reason: str = "",
    ) -> None:
        self.from_state = state_of(from_state) or _raw(from_state)
        self.action = str(_raw(action) or "").strip().lower()
        self.to_state = state_of(to_state) if to_state is not None else None
        self.intent_id = str(intent_id) if intent_id is not None else None
        self.reason = reason or (
            f"illegal transition ({self.from_state!r}, {self.action!r}); "
            f"legal actions from {self.from_state!r}: {legal_actions(self.from_state)!r}"
        )
        super().__init__(self.reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": "ILLEGAL_TRANSITION",
            "from_state": self.from_state,
            "action": self.action,
            "to_state": self.to_state,
            "intent_id": self.intent_id,
            "reason": self.reason,
            "invariant": "Definition 2",
        }


# ── tolerant accessors ─────────────────────────────────────────────────────
def _raw(value: Any) -> Any:
    """Unwrap ``Enum``-like values to their plain value."""
    return getattr(value, "value", value)


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except Exception:  # attribute raises (e.g. unloaded lazy loader)
                continue
    return default


def _set_attr(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, MutableMapping):
        obj[name] = value
        return
    try:
        setattr(obj, name, value)
    except Exception:  # pragma: no cover - read-only stand-ins
        log.debug("fsm.set_attr_failed", extra={"attr": name, "type": type(obj).__name__})


def state_of(intent_or_state: Any) -> Optional[str]:
    """Return the upper-case state string of an intent row *or* a state value."""
    if intent_or_state is None:
        return None
    value = _raw(intent_or_state)
    if isinstance(value, str):
        text = value.strip().upper()
        if text in STATES or text in TERMINAL_STATES:
            return text
        # Not a state string: maybe an object carrying ``state``.
    state = _raw(_attr(intent_or_state, "state", "intent_state", "status"))
    if state is None:
        return None
    text = str(state).strip().upper()
    return text or None


def is_terminal(intent_or_state: Any) -> bool:
    return state_of(intent_or_state) in TERMINAL_STATES


def is_active(intent_or_state: Any) -> bool:
    return state_of(intent_or_state) == "CLAIMED"


def legal_actions(intent_or_state: Any) -> tuple[str, ...]:
    state = state_of(intent_or_state)
    if state is None:
        return ()
    return tuple(action for (frm, action) in LEGAL_TRANSITIONS if frm == state)


def can(intent_or_state: Any, action: Any) -> bool:
    """``True`` iff ``(state, action)`` is legal (no side effects, no raise)."""
    state = state_of(intent_or_state)
    key = (state or "", str(_raw(action) or "").strip().lower())
    return key in LEGAL_TRANSITIONS


def transition(intent_or_state: Any, action: Any) -> str:
    """Validate the transition and return the target state, else raise."""
    state = state_of(intent_or_state)
    act = str(_raw(action) or "").strip().lower()
    key = (state or "", act)
    target = LEGAL_TRANSITIONS.get(key)
    if target is None:
        raise IllegalTransition(state or intent_or_state, act, intent_id=_attr(intent_or_state, "id"))
    return target


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _as_datetime(value: Any) -> Optional[_dt.datetime]:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day, tzinfo=_dt.timezone.utc)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = _dt.datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=_dt.timezone.utc)
    return None


def _lease_expired(intent: Any, now: Optional[_dt.datetime] = None) -> bool:
    expires = _as_datetime(
        _attr(intent, "lease_expires_at", "lease_expires", "lease_expiry", "lease_deadline")
    )
    if expires is None:
        return False
    return (now or _now()) > expires


def apply_transition(
    intent: Any,
    action: Any,
    *,
    worker_id: Any = None,
    lease_s: Optional[int] = None,
    now: Optional[_dt.datetime] = None,
    reason: str = "",
    strict_worker: bool = True,
) -> str:
    """Validate + mutate an in-memory intent object. Returns the new state.

    Guards:

    * ``claim``    — requires ``state == CREATED``; sets ``worker_id`` and
      ``lease_expires_at = now + lease_s`` when provided.
    * ``conclude`` — requires the caller's ``worker_id`` to match the claimed
      ``intent.worker_id`` (guard of T2) when ``strict_worker`` and both are set.
    * ``goal``     — only from ``CONCLUDED`` (T3).
    * ``timeout``  — only from ``CLAIMED`` **and** the lease must have expired
      (T4); clears ``worker_id``/``lease_expires_at`` so the intent is
      re-executable.
    """
    state = state_of(intent)
    act = str(_raw(action) or "").strip().lower()
    intent_id = _attr(intent, "id")

    target = transition(intent, act)

    if act == CONCLUDE and strict_worker:
        owner = _raw(_attr(intent, "worker_id"))
        caller = _raw(worker_id)
        if owner is not None and caller is not None and str(owner) != str(caller):
            raise IllegalTransition(
                state,
                act,
                intent_id=intent_id,
                reason=f"worker_id mismatch: intent owned by {owner!r}, caller {caller!r}",
            )

    if act == TIMEOUT and not _lease_expired(intent, now):
        raise IllegalTransition(
            state,
            act,
            intent_id=intent_id,
            reason="lease has not expired yet; timeout reclamation refused",
        )

    _set_attr(intent, "state", target)
    stamp = now or _now()

    if act == CLAIM:
        if worker_id is not None:
            _set_attr(intent, "worker_id", worker_id)
        if lease_s is not None:
            _set_attr(intent, "lease_expires_at", stamp + _dt.timedelta(seconds=int(lease_s)))
        _set_attr(intent, "lease_heartbeat", stamp)
        _set_attr(intent, "claimed_at", stamp)
    elif act == TIMEOUT:
        _set_attr(intent, "worker_id", None)
        _set_attr(intent, "lease_expires_at", None)
        _set_attr(intent, "lease_heartbeat", None)
        _set_attr(intent, "task_run_id", None)
        result = _attr(intent, "result")
        merged = dict(result) if isinstance(result, Mapping) else {}
        merged["timeout_reclaimed_at"] = stamp.isoformat()
        if reason:
            merged["timeout_reason"] = reason
        _set_attr(intent, "result", merged)
    elif act == CONCLUSION_PLACEHOLDER:  # pragma: no cover - replaced below
        pass
    return target


# ``CONCLUDE`` post-processing (kept separate for readability).
CONCLUDE = "conclude"
CONCLUSION_PLACEHOLDER = "\x00conclude"  # never matched; guards accidental reuse


def _post_conclude(intent: Any, *, success: Any = None, result: Any = None, now: Optional[_dt.datetime] = None) -> None:
    stamp = now or _now()
    _set_attr(intent, "concluded_at", stamp)
    _set_attr(intent, "lease_expires_at", None)
    _set_attr(intent, "lease_heartbeat", None)
    if success is not None:
        _set_attr(intent, "success", bool(success))
    if result is not None:
        _set_attr(intent, "result", dict(result) if isinstance(result, Mapping) else result)


def claim_cas_statement() -> str:
    """The exact CAS predicate used by durable claims (documented for tests)."""
    return (
        "UPDATE intent SET state = 'CLAIMED', worker_id = :worker_id, "
        "lease_expires_at = :lease_expires_at, lease_heartbeat = :now, claimed_at = :now, "
        "updated_at = :now "
        "WHERE id = :intent_id AND project_id = :project_id AND state = 'CREATED'"
    )


# ── durable FSM ────────────────────────────────────────────────────────────
class IntentFSM:
    """Database-backed FSM operations for ``intent`` rows.

    All write paths are single-statement CAS operations so a claim has exactly
    one winner even with 50 concurrent claimers.  The class accepts either a
    SQLAlchemy ``AsyncSession`` (preferred) or any object exposing
    ``execute``/``commit``/``flush``.
    """

    def __init__(
        self,
        session: Any = None,
        *,
        lease_s: int = 360,
        max_rounds: int = 40,
        now: Any = None,
    ) -> None:
        self.session = session
        self.lease_s = int(lease_s)
        self.max_rounds = int(max_rounds)
        self._now = now  # injectable clock (tests)

    # -- clock ------------------------------------------------------------
    def now(self) -> _dt.datetime:
        value = self._now() if callable(self._now) else _now()
        return value

    # -- helpers ----------------------------------------------------------
    async def _scalar(self, statement: Any) -> Any:
        result = await self.session.execute(statement)
        scalar = getattr(result, "scalar_one_or_none", None)
        if scalar is not None:
            return scalar()
        first = getattr(result, "first", None)
        if first is not None:
            row = first()
            return row[0] if row else None
        return result

    async def _rowcount(self, statement: Any) -> int:
        result = await self.session.execute(statement)
        return int(getattr(result, "rowcount", 0) or 0)

    async def _flush(self) -> None:
        flush = getattr(self.session, "flush", None)
        if flush is not None:
            value = flush()
            if hasattr(value, "__await__"):
                await value

    async def get(self, intent_id: Any) -> Any:
        from sqlalchemy import select  # local import: keeps module import-light

        from app.db.models.intent import Intent

        return await self._scalar(select(Intent).where(Intent.id == _coerce(intent_id)))

    # -- T1 claim ---------------------------------------------------------
    async def claim(
        self,
        intent_id: Any,
        worker_id: Any,
        *,
        task_run_id: Any = None,
        lease_s: Optional[int] = None,
        round_index: Optional[int] = None,
        project_id: Any = None,
    ) -> Any:
        """CAS claim: returns the claimed row, or ``None`` if another winner won.

        Uses ``UPDATE ... WHERE state = 'CREATED'`` and checks
        ``rowcount == 1`` — the textbook single-winner primitive.
        """
        from sqlalchemy import update

        from app.db.models.intent import Intent

        lease = int(lease_s if lease_s is not None else self.lease_s)
        stamp = self.now()
        values: dict[str, Any] = {
            "state": "CLAIMED",
            "worker_id": _coerce(worker_id, none_ok=True),
            "lease_expires_at": stamp + _dt.timedelta(seconds=lease),
            "lease_heartbeat": stamp,
            "claimed_at": stamp,
            "updated_at": stamp,
        }
        if task_run_id is not None:
            values["task_run_id"] = _coerce(task_run_id)
        if round_index is not None:
            values["round_index"] = int(round_index)

        predicate = [Intent.id == _coerce(intent_id), Intent.state == "CREATED"]
        if project_id is not None:
            predicate.append(Intent.project_id == _coerce(project_id))

        result = await self.session.execute(
            update(Intent).where(*predicate).values(**values)
        )
        if int(getattr(result, "rowcount", 0) or 0) != 1:
            log.debug(
                "fsm.claim_lost",
                extra={"intent_id": str(intent_id), "worker_id": str(worker_id)},
            )
            return None

        await self._flush()
        log.info(
            "fsm.claimed",
            extra={
                "intent_id": str(intent_id),
                "worker_id": str(worker_id),
                "lease_s": lease,
                "transition": "T1",
            },
        )
        return await self.get(intent_id)

    # -- T2 conclude ------------------------------------------------------
    async def conclude(
        self,
        intent_id: Any,
        *,
        worker_id: Any = None,
        success: Optional[bool] = None,
        result: Any = None,
        new_fact_ids: Optional[Sequence[Any]] = None,
        strict_worker: bool = True,
    ) -> Any:
        """CLAIMED -> CONCLUDED. Refuses when the caller is not the lease owner."""
        from sqlalchemy import update

        from app.db.models.intent import Intent

        intent = await self.get(intent_id)
        if intent is None:
            return None

        state = state_of(intent)
        owner = _raw(_attr(intent, "worker_id"))
        caller = _raw(worker_id)
        if state != "CLAIMED":
            raise IllegalTransition(state, CONCLUDE, intent_id=intent_id)
        if strict_worker and owner is not None and caller is not None and str(owner) != str(caller):
            raise IllegalTransition(
                state,
                CONCLUDE,
                intent_id=intent_id,
                reason=f"worker_id mismatch (owner={owner!r}, caller={caller!r})",
            )

        stamp = self.now()
        merged: dict[str, Any] = {}
        current = _attr(intent, "result")
        if isinstance(current, Mapping):
            merged.update(dict(current))
        if isinstance(result, Mapping):
            merged.update(dict(result))
        elif result is not None:
            merged["value"] = result
        if new_fact_ids is not None:
            merged["new_fact_ids"] = [str(f) for f in new_fact_ids]

        values: dict[str, Any] = {
            "state": "CONCLUDED",
            "concluded_at": stamp,
            "lease_expires_at": None,
            "lease_heartbeat": None,
            "updated_at": stamp,
        }
        if success is not None:
            values["success"] = bool(success)
        if merged:
            values["result"] = merged

        predicate = [
            Intent.id == _coerce(intent_id),
            Intent.state == "CLAIMED",
        ]
        db_worker = _coerce(worker_id, none_ok=True)
        if strict_worker and db_worker is not None:
            predicate.append(Intent.worker_id == db_worker)

        outcome = await self.session.execute(
            update(Intent).where(*predicate).values(**values)
        )
        if int(getattr(outcome, "rowcount", 0) or 0) != 1:
            raise IllegalTransition(
                state_of(intent),
                CONCLUDE,
                intent_id=intent_id,
                reason="conclude CAS lost (state changed or worker mismatch)",
            )
        await self._flush()
        log.info("fsm.concluded", extra={"intent_id": str(intent_id), "success": values.get("success")})
        return await self.get(intent_id)

    # -- T3 goal ----------------------------------------------------------
    async def mark_goal(self, intent_id: Any) -> Any:
        """CONCLUDED -> COMPLETED (invariant I6: no way back out)."""
        from sqlalchemy import update

        from app.db.models.intent import Intent

        intent = await self.get(intent_id)
        if intent is None:
            return None
        if state_of(intent) != "CONCLUDED":
            raise IllegalTransition(state_of(intent), GOAL, intent_id=intent_id)

        stamp = self.now()
        current = _attr(intent, "result")
        merged = dict(current) if isinstance(current, Mapping) else {}
        merged.setdefault("reason", "goal_reached")

        outcome = await self.session.execute(
            update(Intent)
            .where(Intent.id == _coerce(intent_id), Intent.state == "CONCLUDED")
            .values(state="COMPLETED", result=merged, updated_at=stamp)
        )
        if int(getattr(outcome, "rowcount", 0) or 0) != 1:
            raise IllegalTransition(state_of(intent), GOAL, intent_id=intent_id)
        await self._flush()
        return await self.get(intent_id)

    # -- T4 timeout -------------------------------------------------------
    async def reclaim(
        self,
        intent_id: Any,
        *,
        reason: str = "lease_expired",
        require_expired: bool = True,
    ) -> Any:
        """CLAIMED -> CREATED by the reaper (``now() > lease_expires_at``)."""
        from sqlalchemy import update

        from app.db.models.intent import Intent

        intent = await self.get(intent_id)
        if intent is None:
            return None
        if state_of(intent) != "CLAIMED":
            raise IllegalTransition(state_of(intent), TIMEOUT, intent_id=intent_id)
        stamp = self.now()
        if require_expired and not _lease_expired(intent, stamp):
            raise IllegalTransition(
                state_of(intent),
                TIMEOUT,
                intent_id=intent_id,
                reason="lease still valid; reaper refused the transition",
            )

        current = _attr(intent, "result")
        merged = dict(current) if isinstance(current, Mapping) else {}
        merged["timeout_reclaimed_at"] = stamp.isoformat()
        merged["timeout_reason"] = reason

        outcome = await self.session.execute(
            update(Intent)
            .where(
                Intent.id == _coerce(intent_id),
                Intent.state == "CLAIMED",
                Intent.lease_expires_at.is_not(None),
                Intent.lease_expires_at < stamp,
            )
            .values(
                state="CREATED",
                worker_id=None,
                lease_expires_at=None,
                lease_heartbeat=None,
                task_run_id=None,
                result=merged,
                updated_at=stamp,
            )
        )
        if int(getattr(outcome, "rowcount", 0) or 0) != 1:
            return None
        await self._flush()
        log.info("fsm.reclaimed", extra={"intent_id": str(intent_id), "reason": reason})
        return await self.get(intent_id)

    async def reclaim_expired(self, *, project_id: Any = None, limit: int = 200) -> list[Any]:
        """Sweep all expired CLAIMED intents back to CREATED (T4 batch)."""
        from sqlalchemy import select

        from app.db.models.intent import Intent

        stamp = self.now()
        query = (
            select(Intent)
            .where(
                Intent.state == "CLAIMED",
                Intent.lease_expires_at.is_not(None),
                Intent.lease_expires_at < stamp,
            )
            .limit(int(limit))
        )
        if project_id is not None:
            query = query.where(Intent.project_id == _coerce(project_id))
        result = await self.session.execute(query)
        rows = list(result.scalars().all()) if hasattr(result, "scalars") else []
        reclaimed: list[Any] = []
        for row in rows:
            try:
                updated = await self.reclaim(_attr(row, "id"), require_expired=False)
            except IllegalTransition:
                continue
            if updated is not None:
                reclaimed.append(updated)
        return reclaimed

    # -- goal freeze ------------------------------------------------------
    async def goal_freeze(
        self,
        project_id: Any,
        *,
        goal_fact_id: Any = None,
        reason: str = "goal_reached",
    ) -> int:
        """Bulk-migrate every unfinished intent to COMPLETED (goal freeze)."""
        return await goal_freeze(
            self.session,
            project_id,
            goal_fact_id=goal_fact_id,
            reason=reason,
            now=self.now(),
        )


# ── module-level DB helpers (usable without constructing an IntentFSM) ─────
def _coerce(value: Any, *, none_ok: bool = False) -> Any:
    if value is None:
        if none_ok:
            return None
        raise ValueError("value is required")
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return uuid.UUID(text)
        except (ValueError, AttributeError):
            return text
    return value


async def _flush_session(session: Any) -> None:
    flush = getattr(session, "flush", None)
    if flush is None:
        return
    value = flush()
    if hasattr(value, "__await__"):
        await value


async def goal_freeze(
    session: Any,
    project_id: Any,
    *,
    goal_fact_id: Any = None,
    reason: str = "goal_reached",
    now: Optional[_dt.datetime] = None,
) -> int:
    """T3 + project termination, in one transaction.

    * every intent not yet ``COMPLETED`` (``CREATED``/``CLAIMED``/``CONCLUDED``)
      is bulk-migrated to ``COMPLETED`` with ``result.reason = reason``;
    * the project becomes ``COMPLETED`` with ``termination_reason=GOAL`` and its
      lease is cleared.

    Returns the number of intents frozen.  Idempotent: a second call is a no-op.
    """
    from sqlalchemy import select, update

    from app.db.models.intent import Intent
    from app.db.models.project import Project

    stamp = now or _now()
    pid = _coerce(project_id)

    project = None
    try:
        result = await session.execute(select(Project).where(Project.id == pid))
        scalar = getattr(result, "scalar_one_or_none", None)
        project = scalar() if scalar is not None else None
    except Exception:  # pragma: no cover - project model may be absent in tests
        project = None

    freeze = await session.execute(
        update(Intent)
        .where(Intent.project_id == pid, Intent.state != "COMPLETED")
        .values(state="COMPLETED", success=False, updated_at=stamp)
    )
    frozen = int(getattr(freeze, "rowcount", 0) or 0)

    if project is not None:
        values: dict[str, Any] = {"updated_at": stamp}
        for field, value in (
            ("status", "COMPLETED"),
            ("termination_reason", "GOAL"),
            ("completed_at", stamp),
            ("finished_at", stamp),
            ("lease_owner", None),
            ("lease_expires_at", None),
            ("lease_acquired_at", None),
            ("lease_heartbeat", None),
        ):
            if hasattr(Project, field):
                values[field] = value
        if goal_fact_id is not None and hasattr(Project, "goal_anchor_fact"):
            values.setdefault("goal_anchor_fact", goal_fact_id)
        await session.execute(update(Project).where(Project.id == pid).values(**values))

    await _flush_session(session)
    log.info(
        "fsm.goal_freeze",
        extra={"project_id": str(pid), "frozen": frozen, "reason": reason},
    )
    return frozen


async def reclaim_expired(
    session: Any,
    *,
    project_id: Any = None,
    limit: int = 200,
    now: Optional[_dt.datetime] = None,
) -> list[Any]:
    """Module-level reaper sweep (thin wrapper over :class:`IntentFSM`)."""
    return await IntentFSM(session, now=(lambda: now) if now is not None else None).reclaim_expired(
        project_id=project_id, limit=limit
    )
