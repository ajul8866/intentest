"""Runtime attack graph ``G_a`` construction (paper Definition 1 / §4.4, plan C8).

The Fact-Intent DAG persisted in Postgres *is* the runtime attack graph::

    G_a = (F, I, E_f->i, E_i->f)

where

* ``F`` are the immutable facts (verified + candidate) of one project,
* ``I`` are the intents,
* ``E_f->i`` are the ``intent_source`` hyperedges (TRIGGERS), and
* ``E_i->f`` are the producing edges (PRODUCES, i.e. ``fact.derived_from_intent_id``).

This module rebuilds that graph from the database into a ``networkx.DiGraph``
whose node/edge attribute vocabulary matches ``app.core.dag.graph`` and the
``/projects/{id}/graph`` REST view so the retrieval pipeline (Algorithm 2) and
the merge stage can consume it without any translation layer.

Design notes
------------
* **Read-only.** Nothing here mutates the DAG; the only writer of facts/intents
  remains the worker/harvester plane.
* **Import-light at module load.** SQLAlchemy and networkx are imported inside
  the functions so ``import app.core.intent`` stays cheap (matching the package
  convention used across ``app/core``).
* **Tolerant input.** ``build_attack_graph`` accepts either an ``AsyncSession``
  or a pre-loaded row bundle (list of facts / list of intents / list of
  ``intent_source`` rows, or mappings keyed by those names). This keeps it
  usable from unit tests without a database.

Public surface
--------------
``ATTACK_NODE_PREFIX``, ``FACT_PREFIX``, ``INTENT_PREFIX``, ``EDGE_TRIGGERS``,
``EDGE_PRODUCES``, ``fact_node_id``, ``intent_node_id``, ``AttackGraph``,
``build_attack_graph``, ``load_attack_graph``, ``attack_graph_from_rows``,
``frontier_facts``, ``pending_intents``, ``primary_chain``, ``neighbourhood``,
``subgraph_for_facts``, ``graph_stats``, ``to_cytoscape``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

log = logging.getLogger(__name__)

__all__ = [
    "ATTACK_NODE_PREFIX",
    "FACT_PREFIX",
    "INTENT_PREFIX",
    "EDGE_TRIGGERS",
    "EDGE_PRODUCES",
    "EDGE_CORRESPONDS",
    "fact_node_id",
    "intent_node_id",
    "AttackGraph",
    "build_attack_graph",
    "load_attack_graph",
    "attack_graph_from_rows",
    "frontier_facts",
    "pending_intents",
    "primary_chain",
    "neighbourhood",
    "subgraph_for_facts",
    "graph_stats",
    "to_cytoscape",
]

# ---------------------------------------------------------------------------
# Node / edge vocabulary (mirrors app.core.dag.graph + API.md)
# ---------------------------------------------------------------------------

ATTACK_NODE_PREFIX = "attack"
"""Prefix used by the REST graph view for the runtime attack graph."""

FACT_PREFIX = "f:"
INTENT_PREFIX = "i:"

EDGE_TRIGGERS = "TRIGGERS"        # E_f->i  (intent_source)   solid
EDGE_PRODUCES = "PRODUCES"        # E_i->f  (derived_from)    dashed
EDGE_CORRESPONDS = "CORRESPONDS"  # cross-link to G_o         dotted

# Intent states (kept as literals so this module never imports the ORM enum).
_STATE_CREATED = "CREATED"
_STATE_CLAIMED = "CLAIMED"
_STATE_CONCLUDED = "CONCLUDED"
_STATE_COMPLETED = "COMPLETED"
_PENDING_STATES = frozenset({_STATE_CREATED, _STATE_CLAIMED})


def fact_node_id(fact_or_id: Any) -> str:
    """Return the canonical graph node id for a fact (``f:<uuid>``)."""

    return f"{FACT_PREFIX}{_id_of(fact_or_id)}"


def intent_node_id(intent_or_id: Any) -> str:
    """Return the canonical graph node id for an intent (``i:<uuid>``)."""

    return f"{INTENT_PREFIX}{_id_of(intent_or_id)}"


# ---------------------------------------------------------------------------
# Tolerant attribute access helpers
# ---------------------------------------------------------------------------


def _id_of(value: Any) -> str:
    """Stringify an id carried either directly or on an object/mapping."""

    if value is None:
        return ""
    if isinstance(value, (str, int)):
        return str(value)
    if isinstance(value, Mapping):
        for key in ("id", "fact_id", "intent_id", "uuid"):
            if value.get(key) is not None:
                return str(value[key])
        return ""
    for attr in ("id", "fact_id", "intent_id"):
        found = getattr(value, attr, None)
        if found is not None:
            return str(found)
    return str(value)


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first present attribute/mapping key among ``names``."""

    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


def _name(value: Any) -> Optional[str]:
    """Upper-case an enum/str state or kind name; ``None`` stays ``None``."""

    if value is None:
        return None
    raw = getattr(value, "value", value)
    if raw is None:
        return None
    return str(raw).strip().upper()


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _row_bundle(source: Any) -> Dict[str, List[Any]]:
    """Normalise ``source`` into ``{facts, intents, sources}`` row lists.

    Accepts a mapping with those keys (plus aliases), an object exposing them
    as attributes, or an object exposing a ``rows()``/``toString``-style bundle.
    """

    aliases = {
        "facts": ("facts", "fact_rows", "F"),
        "intents": ("intents", "intent_rows", "I"),
        "sources": ("sources", "intent_sources", "edges", "efi"),
    }
    bundle: Dict[str, List[Any]] = {}
    for canonical, names in aliases.items():
        value = _attr(source, *names, default=None)
        bundle[canonical] = list(value) if value else []
    return bundle


def _is_session_like(source: Any) -> bool:
    """Heuristic: does ``source`` look like an async SQLAlchemy session?"""

    if source is None:
        return False
    if isinstance(source, (list, tuple, set, Mapping)):
        return False
    return any(
        callable(getattr(source, name, None))
        for name in ("execute", "scalars", "stream", "get")
    )


# ---------------------------------------------------------------------------
# AttackGraph container
# ---------------------------------------------------------------------------


@dataclass
class AttackGraph:
    """An in-memory view of ``G_a`` for one project.

    ``graph`` is a :class:`networkx.DiGraph` with

    * fact nodes: ``{"node_type": "fact", "kind", "verified", "confidence",
      "label", "fact_key", "mitre_technique", "scope_id", "derived_from"}``
    * intent nodes: ``{"node_type": "intent", "state", "action_type", "tactic",
      "objective", "target_entity", "success", "depth", "label", "pattern_id"}``
    * edges: ``{"relation": TRIGGERS|PRODUCES, ...}``
    """

    project_id: Optional[str] = None
    graph: Any = None
    facts: Dict[str, Any] = field(default_factory=dict)
    intents: Dict[str, Any] = field(default_factory=dict)
    sources: List[Any] = field(default_factory=list)
    embedding_backend: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- basic accessors ---------------------------------------------------

    @property
    def nodes(self) -> int:
        return self.graph.number_of_nodes() if self.graph is not None else 0

    @property
    def edges(self) -> int:
        return self.graph.number_of_edges() if self.graph is not None else 0

    def fact_nodes(self) -> List[str]:
        return sorted(
            n
            for n, data in (self.graph.nodes(data=True) if self.graph else [])
            if data.get("node_type") == "fact"
        )

    def intent_nodes(self) -> List[str]:
        return sorted(
            n
            for n, data in (self.graph.nodes(data=True) if self.graph else [])
            if data.get("node_type") == "intent"
        )

    def to_networkx(self) -> Any:
        """Return the underlying ``networkx.DiGraph`` (may be ``None``)."""

        return self.graph

    def successors(self, node_id: str) -> List[str]:
        if self.graph is None or node_id not in self.graph:
            return []
        return sorted(self.graph.successors(node_id))

    def predecessors(self, node_id: str) -> List[str]:
        if self.graph is None or node_id not in self.graph:
            return []
        return sorted(self.graph.predecessors(node_id))

    def is_empty(self) -> bool:
        return self.nodes == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "facts": len(self.facts),
            "intents": len(self.intents),
            "embedding_backend": self.embedding_backend,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def attack_graph_from_rows(
    facts: Iterable[Any] = (),
    intents: Iterable[Any] = (),
    sources: Iterable[Any] = (),
    *,
    project_id: Any = None,
    verified_only: bool = False,
    include_candidates: bool = True,
) -> AttackGraph:
    """Build :class:`AttackGraph` from already-loaded rows.

    Parameters
    ----------
    facts:
        Fact rows/mappings. ``verified_only`` drops unverified candidates.
    intents:
        Intent rows/mappings. Unverified/absent intents do not prevent a
        ``PRODUCES`` edge: the fact's ``derived_from_intent_id`` is
        authoritative (invariant I4).
    sources:
        ``intent_source`` hyperedges; each becomes a ``TRIGGERS`` edge with
        ``role``/``weight`` attributes.
    """

    try:  # pragma: no cover - networkx is a hard dependency at runtime
        import networkx as nx
    except Exception as exc:  # pragma: no cover
        log.warning("attack_graph.networkx_unavailable", extra={"error": str(exc)})
        return AttackGraph(project_id=_id_of(project_id) or None)

    g = nx.DiGraph()
    fact_index: Dict[str, Any] = {}
    intent_index: Dict[str, Any] = {}
    source_rows: List[Any] = []

    # -- fact nodes --------------------------------------------------------
    for fact in facts or ():
        fid = _id_of(fact)
        if not fid:
            continue
        verified = bool(_attr(fact, "verified", default=False))
        if verified_only and not verified:
            continue
        kind = _name(_attr(fact, "kind", default=None)) or "NOTE"
        fact_index[fid] = fact
        label = _fact_label(fact, kind)
        g.add_node(
            fact_node_id(fid),
            node_type="fact",
            entity="fact",
            fact_id=fid,
            kind=kind,
            verified=verified,
            confidence=float(_attr(fact, "confidence", default=0.0) or 0.0),
            label=label,
            fact_key=_attr(fact, "fact_key", default=None),
            mitre_technique=_attr(fact, "mitre_technique", default=None),
            scope_id=_str_or_none(_attr(fact, "scope_id", default=None)),
            derived_from=_str_or_none(_attr(fact, "derived_from_intent_id", default=None)),
        )

    # -- intent nodes ------------------------------------------------------
    for intent in intents or ():
        iid = _id_of(intent)
        if not iid:
            continue
        intent_index[iid] = intent
        g.add_node(
            intent_node_id(iid),
            node_type="intent",
            entity="intent",
            intent_id=iid,
            state=_name(_attr(intent, "state", default=None)) or _STATE_CREATED,
            action_type=_name(_attr(intent, "action_type", default=None)),
            tactic=_attr(intent, "tactic", default=None),
            objective=_attr(intent, "objective", default=None),
            target_entity=_attr(intent, "target_entity", default=None),
            success=bool(_attr(intent, "success", default=False)),
            depth=int(_attr(intent, "depth", default=0) or 0),
            pattern_id=_str_or_none(_attr(intent, "pattern_id", default=None)),
            label=_intent_label(intent),
        )

    # -- E_f->i (TRIGGERS) -------------------------------------------------
    for source in sources or ():
        src_fact = _str_or_none(
            _attr(source, "fact_id", "source_fact_id", "predecessor_id", default=None)
        )
        src_intent = _str_or_none(
            _attr(source, "intent_id", "target_intent_id", "successor_id", default=None)
        )
        if not src_fact or not src_intent:
            continue
        source_rows.append(source)
        src_node, dst_node = fact_node_id(src_fact), intent_node_id(src_intent)
        # Materialise endpoints even if the row lists were partial; a dangling
        # hyperedge must never silently disappear from the graph.
        if src_node not in g:
            g.add_node(src_node, node_type="fact", entity="fact", fact_id=src_fact,
                       kind="NOTE", verified=bool(_attr(source, "fact_verified", default=False)))
        if dst_node not in g:
            g.add_node(dst_node, node_type="intent", entity="intent", intent_id=src_intent,
                       state=_STATE_CREATED, action_type="EXPLORE")
        g.add_edge(
            src_node,
            dst_node,
            relation=EDGE_TRIGGERS,
            role=_name(_attr(source, "role", default=None)) or "PRECONDITION",
            weight=float(_attr(source, "weight", default=1.0) or 1.0),
            source_fact_id=src_fact,
            target_intent_id=src_intent,
        )

    # -- E_i->f (PRODUCES) -------------------------------------------------
    for fid, fact in fact_index.items():
        producer = _str_or_none(_attr(fact, "derived_from_intent_id", default=None))
        if not producer:
            continue
        src_node, dst_node = intent_node_id(producer), fact_node_id(fid)
        if src_node not in g:
            # I4: the fact is authoritative, so synthesise the producer node.
            g.add_node(src_node, node_type="intent", entity="intent", intent_id=producer,
                       state=_STATE_CONCLUDED, action_type="EXPLORE")
        g.add_edge(
            src_node,
            dst_node,
            relation=EDGE_PRODUCES,
            producing_intent_id=producer,
            fact_id=fid,
        )

    # ``include_candidates=False`` is a strict alias for ``verified_only``
    # (kept for call-site symmetry with the merge stage).
    if not include_candidates and not verified_only:
        unverified = [n for n, d in g.nodes(data=True)
                      if d.get("node_type") == "fact" and not d.get("verified")]
        g.remove_nodes_from(unverified)
        for fid in list(fact_index):
            if fact_node_id(fid) not in g:
                fact_index.pop(fid, None)

    project = _id_of(project_id) or _common_project(fact_index, intent_index)
    return AttackGraph(
        project_id=project or None,
        graph=g,
        facts=fact_index,
        intents=intent_index,
        sources=source_rows,
        metadata={
            "verified_only": verified_only,
            "include_candidates": include_candidates,
            "fact_nodes": len(fact_index),
            "intent_nodes": len(intent_index),
        },
    )


async def build_attack_graph(
    session_or_rows: Any = None,
    *,
    project_id: Any = None,
    verified_only: bool = False,
    include_candidates: bool = True,
    limit: Optional[int] = None,
) -> AttackGraph:
    """Rebuild ``G_a`` for a project.

    ``session_or_rows`` may be:

    * an ``AsyncSession`` (preferred) - rows are selected with SQLAlchemy;
    * a mapping/object bundle of already-loaded rows (tests, evaluation);
    * ``None`` - an empty graph is returned (the caller then has nothing to
      retrieve against; this never raises).

    Rows that fail to load (import error, missing table) degrade to an empty
    graph with a logged warning rather than an exception, matching the
    "never fabricate graph edges" rule of the API retrieval layer.
    """

    if session_or_rows is None:
        return attack_graph_from_rows(project_id=project_id,
                                      verified_only=verified_only,
                                      include_candidates=include_candidates)

    if not _is_session_like(session_or_rows):
        bundle = _row_bundle(session_or_rows)
        project = project_id or _attr(session_or_rows, "project_id", default=None)
        return attack_graph_from_rows(
            bundle["facts"],
            bundle["intents"],
            bundle["sources"],
            project_id=project,
            verified_only=verified_only,
            include_candidates=include_candidates,
        )

    try:
        facts, intents, sources = await _load_rows(
            session_or_rows, project_id=project_id, limit=limit
        )
    except Exception as exc:  # pragma: no cover - defensive DB path
        log.warning(
            "attack_graph.load_failed",
            extra={"project_id": _str_or_none(project_id), "error": str(exc)},
        )
        return attack_graph_from_rows(project_id=project_id,
                                      verified_only=verified_only,
                                      include_candidates=include_candidates)

    return attack_graph_from_rows(
        facts,
        intents,
        sources,
        project_id=project_id,
        verified_only=verified_only,
        include_candidates=include_candidates,
    )


async def load_attack_graph(session: Any, project_id: Any, **kwargs: Any) -> AttackGraph:
    """Convenience wrapper with the ``(session, project_id)`` argument order."""

    return await build_attack_graph(session, project_id=project_id, **kwargs)


async def _load_rows(
    session: Any,
    *,
    project_id: Any = None,
    limit: Optional[int] = None,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Select fact/intent/intent_source rows for one project."""

    from sqlalchemy import select  # local import keeps module load cheap

    from app.db.models.fact import Fact
    from app.db.models.intent import Intent
    from app.db.models.intent_source import IntentSource

    pid = _coerce_uuid(project_id)

    fact_stmt = select(Fact)
    intent_stmt = select(Intent)
    source_stmt = select(IntentSource)

    if pid is not None:
        fact_stmt = fact_stmt.where(Fact.project_id == pid)
        intent_stmt = intent_stmt.where(Intent.project_id == pid)
        source_stmt = source_stmt.where(IntentSource.intent_id.in_(
            select(Intent.id).where(Intent.project_id == pid)
        ))

    if limit:
        fact_stmt = fact_stmt.limit(int(limit))
        intent_stmt = intent_stmt.limit(int(limit))

    facts = list((await session.execute(fact_stmt)).scalars().all())
    intents = list((await session.execute(intent_stmt)).scalars().all())
    sources = list((await session.execute(source_stmt)).scalars().all())
    return facts, intents, sources


def _coerce_uuid(value: Any) -> Any:
    if value is None:
        return None
    import uuid as _uuid

    if isinstance(value, _uuid.UUID):
        return value
    try:
        return _uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Projections used by Algorithm 2 / Algorithm 3 / the graph API
# ---------------------------------------------------------------------------


def frontier_facts(attack: AttackGraph, *, verified_only: bool = False) -> List[Any]:
    """Return facts that no intent has yet consumed (the exploration frontier).

    A fact is on the frontier when it has no outgoing ``TRIGGERS`` edge. This is
    exactly the set Algorithm 2 uses as "predecessor attribute vector" sources
    when no explicit predecessor list is supplied.
    """

    if attack.graph is None:
        return []
    out: List[Any] = []
    for fid, fact in attack.facts.items():
        node = fact_node_id(fid)
        if node not in attack.graph:
            continue
        data = attack.graph.nodes[node]
        if verified_only and not data.get("verified"):
            continue
        if not any(
            attack.graph.edges[e].get("relation") == EDGE_TRIGGERS
            for e in attack.graph.out_edges(node)
        ):
            out.append(fact)
    return out


def pending_intents(attack: AttackGraph) -> List[Any]:
    """Intents in ``CREATED``/``CLAIMED`` (the scheduler's candidate set)."""

    if attack.graph is None:
        return []
    out: List[Any] = []
    for iid, intent in attack.intents.items():
        state = _name(_attr(intent, "state", default=None))
        if state in _PENDING_STATES:
            out.append(intent)
    return out


def primary_chain(attack: AttackGraph, *, max_length: int = 64) -> List[str]:
    """Longest path of the attack graph, used as the S4 "primary chain".

    The DAG is acyclic by construction (invariant I5), so the longest path is
    well defined. Ties are broken deterministically (sorted node ids) so the
    alignment score in stage S4 is reproducible.

    Returns the ordered list of node ids (``f:``/``i:`` prefixed).
    """

    if attack.graph is None or attack.graph.number_of_nodes() == 0:
        return []

    try:
        import networkx as nx
    except Exception:  # pragma: no cover
        return []

    try:
        paths = nx.dag_longest_path(attack.graph, weight=None)
        if len(paths) > max_length:
            paths = paths[:max_length]
        return list(paths)
    except Exception:
        # Fall back to a DFS over sorted roots if the graph is not strictly a
        # DAG (defensive: the integrity check reports that separately).
        roots = sorted(n for n in attack.graph.nodes if attack.graph.in_degree(n) == 0)
        best: List[str] = []
        for root in roots:
            stack: List[Tuple[str, List[str]]] = [(root, [root])]
            seen: Set[str] = set()
            while stack:
                node, path = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                if len(path) > len(best):
                    best = list(path)
                for succ in sorted(attack.graph.successors(node)):
                    stack.append((succ, path + [succ]))
        return best[:max_length]


def neighbourhood(
    attack: AttackGraph,
    node_ids: Sequence[Any],
    *,
    depth: int = 2,
    max_nodes: int = 60,
) -> Any:
    """Bounded breadth-first neighbourhood around ``node_ids``.

    Used by stage S3 (Jaccard over attributed neighbourhood token sets) and by
    the bounded subgraph isomorphism query of stage S2. ``depth`` is clamped to
    ``[1, 4]`` and the result to ``max_nodes`` nodes so retrieval cost stays
    bounded (plan C9).
    """

    if attack.graph is None:
        return None
    try:
        import networkx as nx
    except Exception:  # pragma: no cover
        return None

    depth = max(1, min(int(depth or 2), 4))
    max_nodes = max(1, int(max_nodes or 60))

    seeds: List[str] = []
    for raw in node_ids or ():
        node = str(raw)
        if node.startswith(FACT_PREFIX) or node.startswith(INTENT_PREFIX):
            seed = node
        else:
            # Accept bare ids and infer the node type from the loaded rows.
            if node in attack.facts:
                seed = fact_node_id(node)
            elif node in attack.intents:
                seed = intent_node_id(node)
            else:
                seed = fact_node_id(node)
        if seed in attack.graph:
            seeds.append(seed)
    if not seeds:
        return nx.DiGraph()

    # Undirected BFS so both predecessors and successors are included.
    undirected = attack.graph.to_undirected(as_view=True)
    selected: Set[str] = set(seeds)
    frontier: Set[str] = set(seeds)
    for _ in range(depth):
        nxt: Set[str] = set()
        for node in frontier:
            if node not in undirected:
                continue
            for nb in sorted(undirected.neighbors(node)):
                if nb not in selected:
                    nxt.add(nb)
        if not nxt:
            break
        selected |= nxt
        frontier = nxt
        if len(selected) >= max_nodes:
            break

    ordered = sorted(selected)[:max_nodes]
    return attack.graph.subgraph(ordered).copy()


def subgraph_for_facts(
    attack: AttackGraph,
    fact_ids: Sequence[Any],
    *,
    depth: int = 2,
) -> Any:
    """Depth-``d`` predecessor pattern rooted at ``fact_ids`` (S2 query graph).

    Only ancestor (predecessor) edges are traversed, matching the paper's
    "depth-2 predecessor pattern". The returned graph has at most
    ``SUBGRAPH_MAX_QUERY_NODES`` nodes; oversized queries are truncated by
    deterministic (sorted) node id so results are reproducible.
    """

    if attack.graph is None:
        return None
    try:
        import networkx as nx
    except Exception:  # pragma: no cover
        return None

    try:
        from app.config import settings as _settings

        max_nodes = int(getattr(_settings, "subgraph_max_query_nodes", 12) or 12)
    except Exception:
        max_nodes = 12

    depth = max(1, min(int(depth or 2), 4))
    keep: Set[str] = set()
    frontier: Set[str] = set()
    for raw in fact_ids or ():
        node = str(raw)
        if not (node.startswith(FACT_PREFIX) or node.startswith(INTENT_PREFIX)):
            node = fact_node_id(node)
        if node in attack.graph:
            keep.add(node)
            frontier.add(node)

    for _ in range(depth):
        nxt: Set[str] = set()
        for node in frontier:
            if node not in attack.graph:
                continue
            for pred in attack.graph.predecessors(node):
                if pred not in keep:
                    nxt.add(pred)
        if not nxt:
            break
        keep |= nxt
        frontier = nxt

    ordered = sorted(keep)[:max_nodes]
    return attack.graph.subgraph(ordered).copy()


def graph_stats(attack: AttackGraph) -> Dict[str, Any]:
    """Aggregate counters shown on the Project Detail header / Graph Explorer."""

    if attack.graph is None:
        return {
            "nodes": 0,
            "edges": 0,
            "facts": 0,
            "intents": 0,
            "facts_by_kind": {},
            "intents_by_state": {},
            "facts_by_verification": {"verified": 0, "candidate": 0},
            "verified_only_ratio": 0.0,
            "max_depth": 0,
            "frontier": 0,
        }

    facts_by_kind: Dict[str, int] = {}
    intents_by_state: Dict[str, int] = {}
    verified = 0
    frontier = 0

    for node, data in attack.graph.nodes(data=True):
        if data.get("node_type") == "fact":
            kind = str(data.get("kind") or "NOTE")
            facts_by_kind[kind] = facts_by_kind.get(kind, 0) + 1
            if data.get("verified"):
                verified += 1
            if not any(
                attack.graph.edges[e].get("relation") == EDGE_TRIGGERS
                for e in attack.graph.out_edges(node)
            ):
                frontier += 1
        elif data.get("node_type") == "intent":
            state = str(data.get("state") or _STATE_CREATED)
            intents_by_state[state] = intents_by_state.get(state, 0) + 1

    total_facts = sum(facts_by_kind.values())
    try:
        max_depth = int(getattr(attack.graph, "graph", {}).get("max_depth", 0) or 0)
    except Exception:
        max_depth = 0

    return {
        "nodes": attack.graph.number_of_nodes(),
        "edges": attack.graph.number_of_edges(),
        "facts": total_facts,
        "intents": sum(intents_by_state.values()),
        "facts_by_kind": dict(sorted(facts_by_kind.items())),
        "intents_by_state": dict(sorted(intents_by_state.items())),
        "facts_by_verification": {
            "verified": verified,
            "candidate": max(0, total_facts - verified),
        },
        "verified_only_ratio": (verified / total_facts) if total_facts else 0.0,
        "max_depth": max_depth,
        "frontier": frontier,
    }


def to_cytoscape(attack: AttackGraph, *, limit: int = 500) -> Dict[str, Any]:
    """Serialise ``G_a`` into the REST graph view payload.

    Node ids use the ``f:``/``i:`` prefixes; edge ids are ``efi:`` (TRIGGERS)
    and ``eif:`` (PRODUCES), matching ``app/api/routes/graph.py`` and the
    React Flow frontend contract documented in API.md.
    """

    if attack.graph is None:
        return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0, "truncated": False}

    nodes: List[Dict[str, Any]] = []
    max_nodes = max(1, int(limit or 500))
    truncated = False
    for node, data in sorted(attack.graph.nodes(data=True)):
        if len(nodes) >= max_nodes:
            truncated = True
            break
        payload = {k: v for k, v in data.items()}
        payload["id"] = node
        nodes.append(payload)

    kept = {n["id"] for n in nodes}
    edges: List[Dict[str, Any]] = []
    for source, target, data in sorted(
        attack.graph.edges(data=True), key=lambda e: (str(e[0]), str(e[1]))
    ):
        if source not in kept or target not in kept:
            continue
        relation = data.get("relation") or EDGE_TRIGGERS
        prefix = "efi" if relation == EDGE_TRIGGERS else "eif"
        payload = {k: v for k, v in data.items()}
        payload.update(
            {
                "id": f"{prefix}:{source}->{target}",
                "source": source,
                "target": target,
                "relation": relation,
            }
        )
        edges.append(payload)

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": attack.graph.number_of_nodes(),
        "total_edges": attack.graph.number_of_edges(),
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Labels / misc helpers
# ---------------------------------------------------------------------------


def _fact_label(fact: Any, kind: str) -> str:
    """Short human label for a fact node (shown in the Graph Explorer)."""

    value = _attr(fact, "value", default=None)
    attrs = _attr(fact, "attributes", default=None)
    if isinstance(value, (str, int, float)) and str(value).strip():
        text = str(value)
    elif isinstance(attrs, Mapping) and attrs:
        try:
            text = ", ".join(f"{k}={attrs[k]}" for k in sorted(attrs)[:3])
        except Exception:
            text = kind
    else:
        text = kind
    text = text.replace("\n", " ").strip()
    return f"{kind}: {text[:80]}" if text else kind


def _intent_label(intent: Any) -> str:
    action = _name(_attr(intent, "action_type", default=None)) or "INTENT"
    objective = _attr(intent, "objective", default=None)
    target = _attr(intent, "target_entity", default=None)
    bits = [str(x) for x in (objective, target) if x]
    text = " -> ".join(bits) if bits else action
    return f"{action}: {text[:80]}"


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    if raw is None:
        return None
    text = str(raw)
    return text or None


def _common_project(facts: Mapping[str, Any], intents: Mapping[str, Any]) -> Optional[str]:
    for row in list(facts.values()) + list(intents.values()):
        pid = _attr(row, "project_id", default=None)
        if pid is not None:
            return _str_or_none(pid)
    return None
