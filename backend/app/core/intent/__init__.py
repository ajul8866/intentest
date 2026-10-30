"""Intent knowledge plane (paper Algorithms 2 & 3).

This package implements the "intent graph" side of Intentest:

* :mod:`app.core.intent.offline_graph` -- build the offline, MITRE-mapped
  intent graph ``G_o`` from a corpus (extract -> assemble -> validate ->
  embed -> persist).
* :mod:`app.core.intent.attack_graph` -- reconstruct the runtime attack
  graph ``G_a`` from verified facts and the persisted ``E_f->i`` /
  ``E_i->f`` edges of the Fact--Intent DAG.
* :mod:`app.core.intent.merge` -- merge ``G_o`` and ``G_a`` with
  method-tagged cross links (exact -> cosine >= 0.75 -> attribute Jaccard).
* :mod:`app.core.intent.retrieval` -- Algorithm 2: the five-stage intent
  retrieval & prediction pipeline (S1 embedding probing, S2 bounded
  subgraph isomorphism, S3 Jaccard fuzzy fallback, S4 strategic-intent
  alignment, S5 LLM heuristic reasoning) plus ``submit_for_validation``.
* :mod:`app.core.intent.stages` -- one module per retrieval stage.
* :mod:`app.core.intent.bridge` -- Algorithm 3: the asynchronous intent
  bridge (activation guard -> theta -> dedupe -> three blocks ->
  ``strip_meta`` -> store in ``intent.injected_context``).
* :mod:`app.core.intent.embedding` -- the embedding provider with the
  logged ``api -> sentence-transformers -> hash`` fallback chain.
* :mod:`app.core.intent.prompts` -- prompt templates used by the offline
  extractor and the S5 heuristic stage.

Design rule: this ``__init__`` is deliberately *import-light*.  It defines
the stage vocabulary and the paper-stated / default parameters only; heavy
dependencies (networkx, numpy, SQLAlchemy, HTTP clients) are imported by the
submodules that need them so ``import app.core.intent`` stays cheap for
alembic, the evaluation harness and unit tests.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "KNOWLEDGE_SUBMODULES",
    "RETRIEVAL_STAGE_NAMES",
    "RETRIEVAL_STAGES",
    "STAGE_NAMES",
    "EMBEDDING_FALLBACK_CHAIN",
    "CROSSLINK_METHODS",
    "CROSSLINK_PRECEDENCE",
    "BRIDGE_BLOCK_NAMES",
    "PAPER_CONSTANTS",
    "DEFAULT_PARAMETERS",
    "stage_names",
    "is_valid_stage",
    "stage_index",
    "normalise_stage",
    "default_parameters",
    "paper_constants",
    "embedding_fallback_chain",
]


# ---------------------------------------------------------------------------
# Submodules (names only -- never imported eagerly)
# ---------------------------------------------------------------------------
KNOWLEDGE_SUBMODULES: Final[tuple[str, ...]] = (
    "offline_graph",
    "attack_graph",
    "merge",
    "retrieval",
    "embedding",
    "bridge",
    "dedupe",
    "prompts",
    "stages",
)


# ---------------------------------------------------------------------------
# Algorithm 2 stage vocabulary
# ---------------------------------------------------------------------------
STAGE_NAMES: Final[tuple[str, ...]] = ("S1", "S2", "S3", "S4", "S5")

#: ``STAGE_NAMES`` under an alternative name used by the API/report layers.
RETRIEVAL_STAGE_NAMES: Final[tuple[str, ...]] = STAGE_NAMES

#: Human-readable description of every stage.  The API exposes this
#: verbatim from ``GET /projects/{id}/retrieval/stages``.
RETRIEVAL_STAGES: Final[tuple[dict[str, str], ...]] = (
    {
        "stage": "S1",
        "name": "semantic_probing",
        "description": (
            "Embed the predecessor attribute vector with the embedding "
            "provider (Qwen3-Embedding-4B, 2560-d) and keep the top-k "
            "offline intent patterns by cosine similarity (Eq. 2)."
        ),
    },
    {
        "stage": "S2",
        "name": "bounded_subgraph_isomorphism",
        "description": (
            "Match a depth-2 predecessor pattern against only the S1 Top-k "
            "candidates with a bounded subgraph isomorphism check "
            "(<=12 query nodes, <=60 pattern nodes, 5s)."
        ),
    },
    {
        "stage": "S3",
        "name": "degraded_fuzzy_inference",
        "description": (
            "Only when S2 is empty: rank candidates by attribute Jaccard "
            "similarity over labelled neighbourhood token sets."
        ),
    },
    {
        "stage": "S4",
        "name": "strategic_intent_alignment",
        "description": (
            "Prune out-of-scope / unauthorised lateral movement and "
            "goal-destructive candidates, then score the remainder with "
            "alpha*cos(obj,goal) + beta*cos(obj,primary_chain) + "
            "gamma*confidence >= GOAL_ALIGN_THRESHOLD."
        ),
    },
    {
        "stage": "S5",
        "name": "llm_heuristic_reasoning",
        "description": (
            "LLM heuristic reasoning over a reduced, belief-projected log "
            "when S1-S4 yield nothing.  Output is still validated through "
            "submit_for_validation()."
        ),
    },
)


def stage_names() -> tuple[str, ...]:
    """Return the canonical stage identifiers in pipeline order."""

    return STAGE_NAMES


def is_valid_stage(stage: object) -> bool:
    """True when *stage* names one of the five retrieval stages."""

    return normalise_stage(stage) in STAGE_NAMES


def stage_index(stage: object) -> int | None:
    """Return the 1-based position of a stage (``S1`` -> 1) or ``None``."""

    name = normalise_stage(stage)
    if name in STAGE_NAMES:
        return STAGE_NAMES.index(name) + 1
    return None


def normalise_stage(stage: object) -> str:
    """Normalise ``"s1"``/``1``/``"S1"`` to ``"S1"`` (``""`` when invalid)."""

    if stage is None:
        return ""
    if isinstance(stage, int):
        return f"S{stage}" if 1 <= stage <= len(STAGE_NAMES) else ""
    text = str(getattr(stage, "value", stage)).strip().upper()
    if not text:
        return ""
    if text.isdigit():
        return f"S{text}" if 1 <= int(text) <= len(STAGE_NAMES) else ""
    if text.startswith("STAGE"):
        text = text[5:].strip()
    if not text.startswith("S"):
        text = f"S{text}"
    return text if text in STAGE_NAMES else ""


# ---------------------------------------------------------------------------
# Merge (G_o x G_a) cross-link vocabulary
# ---------------------------------------------------------------------------
#: Matching methods in precedence order (paper section 4.4 / plan C8).
CROSSLINK_PRECEDENCE: Final[tuple[str, ...]] = ("exact", "cosine", "jaccard")

#: Alias kept for readability in call sites that only need the set.
CROSSLINK_METHODS: Final[tuple[str, ...]] = CROSSLINK_PRECEDENCE


# ---------------------------------------------------------------------------
# Algorithm 3 bridge vocabulary
# ---------------------------------------------------------------------------
BRIDGE_BLOCK_NAMES: Final[tuple[str, ...]] = ("B1", "B2", "B3")


# ---------------------------------------------------------------------------
# Embedding provider fallback chain
# ---------------------------------------------------------------------------
EMBEDDING_FALLBACK_CHAIN: Final[tuple[str, ...]] = (
    "api",
    "sentence-transformers",
    "hash",
)


def embedding_fallback_chain() -> tuple[str, ...]:
    """Return the embedding backend fallback chain (highest quality first)."""

    return EMBEDDING_FALLBACK_CHAIN


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
#: Values stated by the paper.  These are *not* the runtime source of truth
#: (``app.config`` is); they exist so tests and the report can assert the
#: paper contract without importing settings.
PAPER_CONSTANTS: Final[dict[str, object]] = {
    "max_rounds": 40,
    "main_timeout_s": 180,
    "degradation_timeout_s": 90,
    "bridge_theta": 0.3,
    "embedding_dim": 2560,
    "extractor_temperature": 0.0,
    "extractor_max_retries": 3,
}

#: Documented defaults for parameters the paper does not publish
#: (see ``LIMITATIONS.md`` section 2).
DEFAULT_PARAMETERS: Final[dict[str, object]] = {
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
    "bridge_theta": 0.3,
    "bridge_max_adjacent_patterns": 3,
    "bridge_recent_facts": 10,
    "bridge_theta_epsilon": 1e-9,
    "embedding_dim": 2560,
}


def default_parameters() -> dict[str, object]:
    """Return a mutable copy of the documented default parameter table."""

    return dict(DEFAULT_PARAMETERS)


def paper_constants() -> dict[str, object]:
    """Return a mutable copy of the paper-stated constant table."""

    return dict(PAPER_CONSTANTS)
