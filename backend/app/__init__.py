"""Intentest backend application package.

Intentest is an intent-graph-guided automated penetration-testing agent.  The
backend package is organised into four concentric planes (see
``ARCHITECTURE.md``):

``db``
    SQLAlchemy models and repositories for the six core tables of the
    Fact--Intent DAG (``project``, ``scope``, ``fact``, ``intent``,
    ``intent_source``) plus support tables (``prompt``, ``worker``,
    ``task_run``, ``event``, ``llm_call``, ``intent_pattern``,
    ``intent_pattern_edge``, ``error_path``, ``challenge``, ``experiment``).
``core``
    Pure domain logic: the DAG/FSM/validator (paper Def. 1-2), the scheduler
    tick (Algorithm 1), the intent knowledge plane (Algorithms 2-3), the LLM
    client, security guards and metrics.
``worker``
    Execution plane: sandbox lifecycle, tool adapters, the agent loop, fact
    extraction/verification and two-phase degradation recovery.
``api`` / ``services`` / ``schemas``
    HTTP + WebSocket surface.  Routes stay thin; business logic lives in
    ``services``.

This module intentionally stays import-light: importing ``app`` must not pull in
SQLAlchemy, FastAPI, Docker or networkx, so that scripts (evaluation harness,
alembic env, tests) can import submodules cheaply and without side effects.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "__app_name__",
    "__description__",
    "__paper_title__",
]

#: Semantic version of the reproduction.  Bumped when the on-disk schema or the
#: paper-facing algorithms change in a way that invalidates stored experiments.
__version__ = "1.0.0"

#: Short application name used for container labels, the Postgres NOTIFY
#: channel prefix and structlog service names.
__app_name__ = "intentest"

#: Human readable description (also surfaced by ``GET /openapi.json``).
__description__ = (
    "Intentest - intent-graph-guided automated penetration testing with a "
    "persistent immutable Fact-Intent DAG, lease/FSM scheduler and a "
    "five-stage intent retrieval pipeline."
)

#: Title of the paper this repository reproduces.
__paper_title__ = (
    "Staying on the Attack Path: Structured State for Long-Horizon Automated "
    "Penetration Testing"
)
