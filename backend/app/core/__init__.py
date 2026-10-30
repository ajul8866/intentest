"""Core plane of Intentest — algorithms, invariants and guards.

This package holds everything that is *pure domain logic* and therefore testable
without a running FastAPI app, Docker daemon or LLM endpoint:

``app.core.dag``
    Paper Definition 1 & 2. The persistent, immutable Fact-Intent DAG
    ``G = (F, I, E_f->i, E_i->f)`` reconstructed from Postgres, the four-state
    intent FSM (Definition 2 / Eq. 1), the precondition/scope validator that
    enforces invariants I1-I6, the POMDP belief projection ``b_t`` and the
    append-only event/audit helpers.

``app.core.scheduler``
    Paper Algorithm 1. The tick loop (harvest -> reclaim -> cleanup ->
    project lease -> task factory -> priority -> Reason gate -> six-stage
    load balancing -> CAS claim -> dispatch), the TaskFactory, priority ranks,
    leases and reapers, stall detection and two-phase degradation bookkeeping.

``app.core.intent``
    Paper Algorithms 2 & 3 and the knowledge plane: offline intent graph
    ``G_o`` construction (MITRE-mandatory extractor + validator), runtime
    attack graph ``G_a``, merge/cross-link, the five-stage retrieval pipeline
    (S1 embedding -> S2 bounded isomorphism -> S3 Jaccard -> S4 strategic
    alignment -> S5 LLM heuristic) and the asynchronous intent bridge
    (``theta = 0.3``, dedupe, ``strip_meta`` allow-list).

``app.core.llm``
    OpenAI-compatible async client, mandatory startup capability probe,
    strategy resolution ``TOOL_CALLING > JSON_SCHEMA > JSON_OBJECT >
    PROMPT_JSON``, fenced/balanced-JSON parser with one repair round-trip,
    retry policy and per-call usage accounting.

``app.core.security``
    Two-layer security: the scope guard (``allowed_hosts``/``allowed_cidrs``/
    ``allowed_ports``/``denied_actions``), sandbox isolation policy,
    prompt-injection detection for tool output and secret redaction.

``app.core.metrics``
    Drift / performance / graph statistics used by reports, the retrieval
    trace UI and the evaluation harness.

``app.core.goals`` / ``app.core.rounds``
    Goal-anchor matching (FLAG vs ``goal_anchor_fact.attributes["flag_regex"]``,
    ``re.fullmatch`` with ``IGNORECASE``) and the 40-round per-task budget
    accounting.

Design rule: importing ``app.core`` must stay cheap. No submodule is imported
eagerly here so that ``alembic``, the evaluation harness and unit tests can
import individual algorithms without pulling in SQLAlchemy engines, the Docker
SDK, networkx or HTTP clients. Import what you need explicitly, e.g.::

    from app.core.dag.fsm import IntentFSM
    from app.core.scheduler.tick import TickScheduler
    from app.core.intent.retrieval import RetrievalPipeline

Subpackages exported (names only, not imported): ``dag``, ``scheduler``,
``intent``, ``llm``, ``security``, ``metrics``.
"""

from __future__ import annotations

from typing import Final

#: Subpackages of the core plane. Exposed as names only; importing
#: ``app.core`` never imports these eagerly.
CORE_SUBPACKAGES: Final[tuple[str, ...]] = (
    "dag",
    "scheduler",
    "intent",
    "llm",
    "security",
    "metrics",
)

#: Modules that live directly under ``app.core`` (not in a subpackage).
CORE_MODULES: Final[tuple[str, ...]] = (
    "goals",
    "rounds",
)

#: Paper-stated constants that core code must never silently change. These
#: mirror ``app.config`` defaults; the authoritative runtime values come from
#: settings, this mapping exists so pure-logic modules (and tests) can assert
#: on the paper contract without importing the settings singleton.
PAPER_CONSTANTS: Final[dict[str, object]] = {
    "max_rounds": 40,
    "main_timeout_s": 180,
    "degradation_timeout_s": 90,
    "bridge_theta": 0.3,
    "embedding_dim": 2560,
    "extractor_temperature": 0.0,
    "extractor_max_retries": 3,
}

#: Documented (non-paper-stated) defaults, see LIMITATIONS.md section 2.
DEFAULT_PARAMETERS: Final[dict[str, object]] = {
    "project_lease_s": 600,
    "intent_lease_s": 360,
    "intent_heartbeat_s": 20,
    "tick_seconds": 2,
    "subgraph_top_k": 10,
    "jaccard_threshold": 0.5,
    "crosslink_cosine": 0.75,
    "confidence_threshold": 0.6,
    "align_alpha": 0.5,
    "align_beta": 0.3,
    "align_gamma": 0.2,
    "goal_align_threshold": 0.55,
    "subgraph_max_query_nodes": 12,
    "subgraph_max_pattern_nodes": 60,
    "subgraph_timeout_s": 5.0,
    "stall_ticks": 15,
}

#: Invariant identifiers enforced in code (see ARCHITECTURE.md section 3).
INVARIANTS: Final[dict[str, str]] = {
    "I1": "facts are immutable; corrections use supersedes_fact_id",
    "I2": "intent_source (E_f->i) is write-once at intent creation",
    "I3": "every predecessor fact must be verified and in the same project/scope",
    "I4": "E_i->f is exactly fact.derived_from_intent_id (one producing intent)",
    "I5": "acyclicity is structural; verify_acyclic() is an audit",
    "I6": "terminal irreversibility: no transition out of COMPLETED",
}

#: FSM states (Definition 2) and the four legal transitions (Eq. 1).
FSM_STATES: Final[tuple[str, ...]] = ("CREATED", "CLAIMED", "CONCLUDED", "COMPLETED")

FSM_TRANSITIONS: Final[dict[tuple[str, str], str]] = {
    ("CREATED", "claim"): "CLAIMED",
    ("CLAIMED", "conclude"): "CONCLUDED",
    ("CONCLUDED", "goal"): "COMPLETED",
    ("CLAIMED", "timeout"): "CREATED",
}

__all__ = [
    "CORE_SUBPACKAGES",
    "CORE_MODULES",
    "PAPER_CONSTANTS",
    "DEFAULT_PARAMETERS",
    "INVARIANTS",
    "FSM_STATES",
    "FSM_TRANSITIONS",
]
