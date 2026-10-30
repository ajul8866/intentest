"""Fact–Intent DAG core (paper Definition 1, Definition 2, invariants I1–I6).

This package owns the *structured state* of an Intentest episode.  Everything the
LLM sees is a projection of the structures built here; nothing here ever depends on
an LLM being available.

Modules
-------
``graph``      :class:`FactIntentDAG` — reconstructs ``G = (F, I, E_f->i, E_i->f)``
               from Postgres as a bipartite, alternating, acyclic directed graph and
               exposes the DFS colour-sweep acyclicity audit (invariant I5).
``fsm``        :class:`IntentFSM` — the four-transition intent lifecycle
               (Definition 2 / Equation 1) with CAS claims, goal freeze and timeout
               reclamation.
``validator``  : precondition / scope / acyclicity validation for intents
               (invariant I3).  This is the IPIA defence: a candidate intent — even
               one produced by the S5 LLM heuristic — only enters the graph through
               :func:`validator.validate_preconditions`.
``belief``     : the POMDP belief projection ``b_t`` — a token-budgeted rendering of
               verified facts, frontier leaves, pending/active intents, the goal and
               the phase.  State lives in Postgres, never in the context window.
``events``     : append-only audit + ``pg_notify`` publishing on channel
               ``intest_events`` (the WebSocket event bus) for DAG mutations.

Invariants (enforced across this package, see ``ALGORITHMS.md`` §1)
------------------------------------------------------------------
I1  Facts are never UPDATEd; corrections create a new fact with ``supersedes_fact_id``.
I2  ``intent_source`` (``E_f->i``) rows are written exactly once, at intent creation.
I3  Every predecessor of a legal intent is verified and belongs to the same
    project/scope.
I4  ``E_i->f`` is exactly ``fact.derived_from_intent_id``; a fact has at most one
    producing intent.
I5  Acyclicity is structural (edges are frozen before an intent may produce facts)
    and auditable (``verify_acyclic()``).
I6  Terminal irreversibility — no intent leaves ``COMPLETED``.

Import policy
-------------
This ``__init__`` stays import-light: submodules are *not* imported eagerly and no
SQLAlchemy / networkx / asyncpg object is constructed at import time.  Callers must
import explicitly, e.g. ``from app.core.dag.fsm import IntentFSM``.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    # submodule names (import explicitly; not re-exported eagerly)
    "belief",
    "events",
    "fsm",
    "graph",
    "validator",
    # constants
    "ALLOWED_TRANSITIONS",
    "DAG_INVARIANTS",
    "INTENT_STATES",
    "INTENT_STATE_ACTIONS",
    "TERMINAL_INTENT_STATES",
]


#: Intent lifecycle states (Definition 2).  Exactly four states, four transitions.
INTENT_STATES: Final[tuple[str, ...]] = ("CREATED", "CLAIMED", "CONCLUDED", "COMPLETED")

#: The only legal ``(from_state, action) -> to_state`` transitions (Definition 2).
#: ``claim``   : CREATED  -> CLAIMED   (CAS UPDATE ... WHERE state='CREATED', one winner)
#: ``conclude``: CLAIMED  -> CONCLUDED (guarded by worker_id match)
#: ``goal``    : CONCLUDED-> COMPLETED (goal hit; also used by the bulk goal freeze)
#: ``timeout`` : CLAIMED  -> CREATED   (lease reaper, now() > lease_expires_at)
ALLOWED_TRANSITIONS: Final[dict[tuple[str, str], str]] = {
    ("CREATED", "claim"): "CLAIMED",
    ("CLAIMED", "conclude"): "CONCLUDED",
    ("CONCLUDED", "goal"): "COMPLETED",
    ("CLAIMED", "timeout"): "CREATED",
}

#: Actions valid for each state, derived from :data:`ALLOWED_TRANSITIONS`.
INTENT_STATE_ACTIONS: Final[dict[str, tuple[str, ...]]] = {
    state: tuple(action for (src, action) in ALLOWED_TRANSITIONS if src == state)
    for state in INTENT_STATES
}

#: States from which no transition may ever be taken again (invariant I6).
TERMINAL_INTENT_STATES: Final[frozenset[str]] = frozenset({"COMPLETED"})

#: Human-readable invariant identifiers; used in error details, tests and docs.
DAG_INVARIANTS: Final[dict[str, str]] = {
    "I1": "facts are immutable (corrections via supersedes_fact_id)",
    "I2": "intent_source (E_f->i) is write-once at intent creation",
    "I3": "every predecessor is verified and same project/scope",
    "I4": "E_i->f is exactly fact.derived_from_intent_id (single producer)",
    "I5": "the graph is acyclic (structural + verify_acyclic() audit)",
    "I6": "terminal irreversibility (no intent leaves COMPLETED)",
}


def invariant_ids() -> tuple[str, ...]:
    """Return the invariant identifiers in declaration order (``I1``..``I6``)."""
    return tuple(DAG_INVARIANTS)


def describe_invariant(identifier: str) -> str:
    """Return the human-readable statement of ``identifier`` (``""`` if unknown)."""
    return DAG_INVARIANTS.get(str(identifier).strip().upper(), "")


def is_terminal(state: object) -> bool:
    """``True`` when *state* is terminal (i.e. ``COMPLETED``), tolerant of enums."""
    return _state_name(state) in TERMINAL_INTENT_STATES


def legal_transitions(state: object) -> tuple[str, ...]:
    """Return the actions allowed from *state* (empty tuple for unknown states)."""
    return INTENT_STATE_ACTIONS.get(_state_name(state), ())


def next_state(state: object, action: object) -> str | None:
    """Return the target state of ``(state, action)`` or ``None`` if illegal."""
    src = _state_name(state)
    act = _action_name(action)
    return ALLOWED_TRANSITIONS.get((src, act))


def _state_name(value: object) -> str:
    """Normalise an intent state (enum / str) to its upper-case string form."""
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().upper()


def _action_name(value: object) -> str:
    """Normalise a transition action to its lower-case string form."""
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().lower()
