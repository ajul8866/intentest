"""Stage S2 - bounded subgraph isomorphism (Algorithm 2, line 2).

S2 is the structural verifier that sits between S1 semantic probing and S3
degraded fuzzy inference.  It builds a depth-2 predecessor pattern from the
runtime attack graph ``G_a`` and tests it for sub-graph isomorphism against
**only** the Top-k candidates produced by S1.  That restriction is what keeps
the cost of this stage ``O(k)`` instead of the exponential blow-up of an
unbounded subgraph search.

Bounds (all configurable, see ``LIMITATIONS.md`` section 2):

* ``SUBGRAPH_MAX_QUERY_NODES``    = 12  - query pattern node cap
* ``SUBGRAPH_MAX_PATTERN_NODES``  = 60  - pattern subgraph node cap
* ``SUBGRAPH_TIMEOUT_S``          = 5.0 - wall clock budget per candidate

Candidates that exceed a bound or time out are *skipped* and recorded in the
trace with reason ``"timeout"`` / ``"too_large"``; they are never silently
dropped without a trace entry.

The stage never mutates the DAG and never creates an intent - any candidate it
returns must still pass ``retrieval.submit_for_validation()`` (invariant I3 /
IPIA defence).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

STAGE = "S2"
STAGE_NAME = "bounded_subgraph_isomorphism"

DEFAULT_MAX_QUERY_NODES = 12
DEFAULT_MAX_PATTERN_NODES = 60
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_TOP_K = 10


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def _config_float(name: str, default: float) -> float:
    """Read a float from the settings singleton, tolerating absence."""
    try:
        from app.config import get_settings

        settings = get_settings()
        value = getattr(settings, name, None)
        if value is None:
            value = getattr(settings, name.lower(), None)
        return float(value) if value is not None else default
    except Exception:  # pragma: no cover - settings optional in tests
        return default


def _config_int(name: str, default: int) -> int:
    return int(_config_float(name, float(default)))


# ---------------------------------------------------------------------------
# result containers
# ---------------------------------------------------------------------------
@dataclass
class S2Candidate:
    """One S1 candidate that survived (or failed) the isomorphism test."""

    pattern_id: str
    pattern: Any = None
    score: float = 0.0
    matched: bool = False
    reason: str = ""
    query_nodes: int = 0
    pattern_nodes: int = 0
    node_matches: int = 0
    edge_matches: int = 0
    method: str = "isomorphism"
    name: str = ""
    tactic: str = ""
    objective: str = ""
    target_entity_type: str = ""
    vulnerability_type: str = ""
    mitre_attack_id: str = ""

    def to_dict(self, *, include_pattern: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "pattern_id": self.pattern_id,
            "score": round(float(self.score), 6),
            "matched": bool(self.matched),
            "method": self.method,
            "name": self.name,
            "tactic": self.tactic,
            "objective": self.objective,
            "target_entity_type": self.target_entity_type,
            "vulnerability_type": self.vulnerability_type,
            "mitre_attack_id": self.mitre_attack_id,
            "query_nodes": self.query_nodes,
            "pattern_nodes": self.pattern_nodes,
            "node_matches": self.node_matches,
            "edge_matches": self.edge_matches,
        }
        if self.reason:
            payload["reason"] = self.reason
        if include_pattern:
            payload["pattern"] = self.pattern
        return payload


@dataclass
class S2Result:
    """Outcome of the bounded isomorphism stage."""

    stage: str = STAGE
    stage_name: str = STAGE_NAME
    candidates: List[S2Candidate] = field(default_factory=list)
    excluded: List[S2Candidate] = field(default_factory=list)
    examined: int = 0
    top_k: int = DEFAULT_TOP_K
    timeout_s: float = DEFAULT_TIMEOUT_S
    duration_ms: float = 0.0
    degraded: bool = False
    reason: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.candidates

    def candidate_dicts(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.candidates]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_name": self.stage_name,
            "candidates": [c.to_dict() for c in self.candidates],
            "excluded": [c.to_dict() for c in self.excluded],
            "examined": self.examined,
            "top_k": self.top_k,
            "timeout_s": self.timeout_s,
            "duration_ms": self.duration_ms,
            "degraded": self.degraded,
            "reason": self.reason,
            "stats": dict(self.stats),
        }


# ---------------------------------------------------------------------------
# tolerant accessors
# ---------------------------------------------------------------------------
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, dict):
            if name in obj and obj[name] is not None:
                return obj[name]
            continue
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _pattern_id(candidate: Any) -> str:
    value = _attr(candidate, "pattern_id", "id", "pattern", default="")
    if not value:
        pattern = _attr(candidate, "pattern", default=None)
        value = _attr(pattern, "id", "pattern_id", default="")
    return str(value) if value else ""


# ---------------------------------------------------------------------------
# graph construction helpers
# ---------------------------------------------------------------------------
def _import_networkx() -> Any:
    try:
        import networkx as nx

        return nx
    except Exception:  # pragma: no cover - networkx is a hard dependency
        return None


def _node_token(graph: Any, node: Any) -> str:
    """Structural token: node type plus kind/tactic, never instance ids."""
    data = graph.nodes.get(node, {}) if hasattr(graph, "nodes") else {}
    node_type = str(data.get("node_type") or "").lower()
    if node_type == "fact":
        return "fact:%s" % str(data.get("kind") or "").upper()
    if node_type == "intent":
        return "intent:%s" % str(data.get("action_type") or "").upper()
    if node_type == "pattern":
        return "pattern:%s" % str(data.get("tactic") or "").upper()
    return "node:%s" % node_type


def _edge_token(graph: Any, source: Any, target: Any) -> str:
    data = graph.edges.get((source, target), {}) if hasattr(graph, "edges") else {}
    return str(data.get("relation") or "EDGE").upper()


def _bounded_subgraph(graph: Any, seeds: Sequence[Any], *, depth: int = 2, max_nodes: int = 60) -> Any:
    """Breadth-first bounded neighbourhood around ``seeds`` (undirected BFS)."""
    nx = _import_networkx()
    if nx is None or graph is None:
        return None
    selected: List[Any] = []
    seen = set()
    frontier = [s for s in seeds if s in graph]
    for seed in frontier:
        seen.add(seed)
        selected.append(seed)
    current = list(frontier)
    for _ in range(max(1, depth)):
        if not current or len(selected) >= max_nodes:
            break
        nxt: List[Any] = []
        for node in current:
            neighbours = list(graph.successors(node)) + list(graph.predecessors(node))
            for neighbour in sorted(neighbours, key=str):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                selected.append(neighbour)
                nxt.append(neighbour)
                if len(selected) >= max_nodes:
                    break
            if len(selected) >= max_nodes:
                break
        current = nxt
    return graph.subgraph(selected).copy()


def _to_token_graph(graph: Any, *, max_nodes: int) -> Any:
    """Collapse a typed graph into a structural token graph for matching."""
    nx = _import_networkx()
    if nx is None or graph is None:
        return None
    token_graph = nx.DiGraph()
    nodes = sorted(graph.nodes(), key=str)[: max(1, max_nodes)]
    keep = set(nodes)
    for node in nodes:
        token_graph.add_node(node, token=_node_token(graph, node))
    for source, target in graph.edges():
        if source in keep and target in keep:
            token_graph.add_edge(source, target, token=_edge_token(graph, source, target))
    return token_graph


def build_query_graph(
    attack_graph: Any,
    fact_ids: Sequence[str] = (),
    *,
    depth: int = 2,
    max_nodes: Optional[int] = None,
) -> Any:
    """Depth-D predecessor pattern around the predecessor facts.

    Delegates to :func:`app.core.intent.attack_graph.subgraph_for_facts` when
    the runtime graph wrapper is supplied so node-id conventions stay in one
    place; falls back to a local BFS for a raw ``networkx`` graph.
    """
    limit = max_nodes if max_nodes is not None else _config_int(
        "subgraph_max_query_nodes", DEFAULT_MAX_QUERY_NODES
    )
    if attack_graph is None:
        return None
    inner = getattr(attack_graph, "graph", attack_graph)
    node_ids: List[str] = []
    for fact_id in fact_ids or ():
        text = str(fact_id)
        node_ids.append(text if ":" in text else "f:%s" % text)
    try:
        from app.core.intent.attack_graph import subgraph_for_facts

        sub = subgraph_for_facts(attack_graph, fact_ids, depth=depth)
        if sub is not None:
            return sub
    except Exception:  # pragma: no cover - optional import path
        log.debug("s2.subgraph_helper_unavailable", exc_info=True)
    seeds = [n for n in node_ids if inner is not None and n in inner]
    if inner is None:
        return None
    return _bounded_subgraph(inner, seeds, depth=depth, max_nodes=limit)


def _candidate_pattern_graph(candidate: Any, attack_graph: Any) -> Any:
    """Structural pattern to compare against the query.

    Real pattern rows carry a ``preconditions`` structure; we materialise a
    tokenised graph from it when present, else we derive one from the runtime
    neighbourhood of the candidate's linked fact (merged-graph cross-links).
    """
    nx = _import_networkx()
    if nx is None:
        return None
    pattern = _attr(candidate, "pattern", default=None)
    source = pattern if pattern is not None else candidate
    preconditions = _attr(source, "preconditions", default=None)
    graph = nx.DiGraph()
    if isinstance(preconditions, dict):
        nodes = preconditions.get("nodes") or preconditions.get("facts") or []
        edges = preconditions.get("edges") or preconditions.get("links") or []
        for entry in nodes:
            if isinstance(entry, dict):
                node_id = str(entry.get("id") or entry.get("node") or len(graph))
                graph.add_node(node_id, token=str(entry.get("token") or entry.get("kind") or entry))
            else:
                graph.add_node(str(entry), token=str(entry))
        for entry in edges:
            if isinstance(entry, dict):
                src = str(entry.get("source") or entry.get("from") or "")
                dst = str(entry.get("target") or entry.get("to") or "")
                if src in graph and dst in graph:
                    graph.add_edge(src, dst, token=str(entry.get("relation") or "EDGE"))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                src, dst = str(entry[0]), str(entry[1])
                if src in graph and dst in graph:
                    graph.add_edge(src, dst, token="EDGE")
        return graph
    if isinstance(preconditions, (list, tuple)):
        for entry in preconditions:
            token = str(entry.get("kind") if isinstance(entry, dict) else entry)
            graph.add_node(token, token=token)
        return graph
    # Fall back to the merged-graph neighbourhood attached to the candidate.
    merged = _attr(candidate, "pattern", default=None)
    linked = _attr(merged, "linked_fact_ids", default=None)
    inner = getattr(attack_graph, "graph", attack_graph)
    if linked and inner is not None:
        seeds = [n for n in (str(x) if ":" in str(x) else "f:%s" % x for x in linked) if n in inner]
        return _bounded_subgraph(inner, seeds, max_nodes=_config_int("subgraph_max_pattern_nodes", DEFAULT_MAX_PATTERN_NODES))
    return None


def _match_graphs(query: Any, pattern: Any, *, timeout_s: float) -> Tuple[bool, int, int, str]:
    """Run the bounded isomorphism test; returns (matched, nodes, edges, reason)."""
    nx = _import_networkx()
    if nx is None or query is None or pattern is None:
        return False, 0, 0, "no_graph"
    try:
        from networkx.algorithms.isomorphism import DiGraphMatcher
    except Exception:  # pragma: no cover
        return False, 0, 0, "matcher_unavailable"

    def node_match(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        return str(left.get("token", "")) == str(right.get("token", ""))

    def edge_match(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        return str(left.get("token", "")) == str(right.get("token", ""))

    deadline = time.monotonic() + max(0.05, float(timeout_s))
    matcher = DiGraphMatcher(pattern, query, node_match=node_match, edge_match=edge_match)
    try:
        for mapping in matcher.subgraph_isomorphisms_iter():
            if time.monotonic() > deadline:
                return False, 0, 0, "timeout"
            return True, len(mapping), max(0, len(mapping) - 1), "matched"
    except Exception as exc:  # pragma: no cover - defensive
        return False, 0, 0 "error:%s" % type(exc).__name__
    if time.monotonic() > deadline:
        return False, 0, 0, "timeout"
    return False, 0, 0, "isomorphism_mismatch"


# ---------------------------------------------------------------------------
# stage entry points
# ---------------------------------------------------------------------------
def run_s2(
    candidates: Iterable[Any] = (),
    attack_graph: Any = None,
    *,
    predecessor_fact_ids: Sequence[str] = (),
    top_k: int = DEFAULT_TOP_K,
    max_query_nodes: Optional[int] = None,
    max_pattern_nodes: Optional[int] = None,
    timeout_s: Optional[float] = None,
) -> S2Result:
    """Run S2 over the S1 Top-k candidate set only (cost O(k))."""
    started = time.monotonic()
    limit_k = max(1, int(top_k or DEFAULT_TOP_K))
    query_limit = max_query_nodes if max_query_nodes is not None else _config_int(
        "subgraph_max_query_nodes", DEFAULT_MAX_QUERY_NODES
    )
    pattern_limit = max_pattern_nodes if max_pattern_nodes is not None else _config_int(
        "subgraph_max_pattern_nodes", DEFAULT_MAX_PATTERN_NODES
    )
    budget = timeout_s if timeout_s is not None else _config_float(
        "subgraph_timeout_s", DEFAULT_TIMEOUT_S
    )

    pool = list(candidates or ())[:limit_k]
    result = S2Result(top_k=limit_k, timeout_s=float(budget))
    if not pool:
        result.reason = "no_candidates"
        result.duration_ms = (time.monotonic() - started) * 1000.0
        return result

    query_raw = build_query_graph(attack_graph, predecessor_fact_ids, max_nodes=query_limit)
    query = _to_token_graph(query_raw, max_nodes=query_limit) if query_raw is not None else None
    result.stats["query_nodes"] = len(query.nodes()) if query is not None else 0
    if query is None or len(query.nodes()) == 0:
        result.reason = "empty_query_graph"
        result.duration_ms = (time.monotonic() - started) * 1000.0
        # Nothing structural to compare: report every candidate as examined but
        # unmatched so the orchestrator falls through to S3.
        for candidate in pool:
            result.excluded.append(
                S2Candidate(
                    pattern_id=_pattern_id(candidate),
                    pattern=_attr(candidate, "pattern", default=None),
                    score=float(_attr(candidate, "score", default=0.0) or 0.0),
                    reason="empty_query_graph",
                    name=str(_attr(candidate, "name", default="") or ""),
                    tactic=str(_attr(candidate, "tactic", default="") or ""),
                    objective=str(_attr(candidate, "objective", default="") or ""),
                )
            )
        result.examined = len(pool)
        return result

    for candidate in pool:
        result.examined += 1
        pattern_raw = _candidate_pattern_graph(candidate, attack_graph)
        if pattern_raw is None:
            result.excluded.append(
                S2Candidate(
                    pattern_id=_pattern_id(candidate),
                    pattern=_attr(candidate, "pattern", default=None),
                    score=float(_attr(candidate, "score", default=0.0) or 0.0),
                    reason="no_pattern_graph",
                )
            )
            continue
        if len(pattern_raw.nodes()) > pattern_limit:
            result.excluded.append(
                S2Candidate(
                    pattern_id=_pattern_id(candidate),
                    pattern=_attr(candidate, "pattern", default=None),
                    score=float(_attr(candidate, "score", default=0.0) or 0.0),
                    reason="too_large",
                    pattern_nodes=len(pattern_raw.nodes()),
                )
            )
            continue
        pattern = _to_token_graph(pattern_raw, max_nodes=pattern_limit)
        matched, nodes, edges, reason = _match_graphs(query, pattern, timeout_s=budget)
        entry = S2Candidate(
            pattern_id=_pattern_id(candidate),
            pattern=_attr(candidate, "pattern", default=None),
            score=float(_attr(candidate, "score", default=0.0) or 0.0),
            matched=matched,
            reason="" if matched else reason,
            query_nodes=len(query.nodes()),
            pattern_nodes=len(pattern.nodes()) if pattern is not None else 0,
            node_matches=nodes,
            edge_matches=edges,
            name=str(_attr(candidate, "name", default="") or ""),
            tactic=str(_attr(candidate, "tactic", default="") or ""),
            objective=str(_attr(candidate, "objective", default="") or ""),
            target_entity_type=str(_attr(candidate, "target_entity_type", default="") or ""),
            vulnerability_type=str(_attr(candidate, "vulnerability_type", default="") or ""),
            mitre_attack_id=str(_attr(candidate, "mitre_attack_id", default="") or ""),
        )
        if matched:
            result.candidates.append(entry)
        else:
            result.excluded.append(entry)

    result.candidates.sort(key=lambda c: (-c.score, c.pattern_id))
    result.stats.update(
        {
            "input": len(pool),
            "matched": len(result.candidates),
            "excluded": len(result.excluded),
            "max_query_nodes": query_limit,
            "max_pattern_nodes": pattern_limit,
        }
    )
    if not result.candidates:
        result.reason = "no_structural_match"
        result.degraded = True
    result.duration_ms = (time.monotonic() - started) * 1000.0
    log.debug(
        "s2.completed",
        extra={
            "examined": result.examined,
            "matched": len(result.candidates),
            "duration_ms": result.duration_ms,
        },
    )
    return result


async def s2_bounded_isomorphism(candidates: Iterable[Any] = (), attack_graph: Any = None, **kwargs: Any) -> S2Result:
    """Async alias used by the retrieval orchestrator."""
    return run_s2(candidates, attack_graph, **kwargs)


async def run_stage(context: Any = None, **kwargs: Any) -> S2Result:
    """Uniform stage entry point consumed by ``stages.load_stage``."""
    ctx = context
    candidates = kwargs.pop("candidates", None)
    if candidates is None:
        candidates = getattr(ctx, "candidates", ()) or ()
    attack_graph = kwargs.pop("attack_graph", None)
    if attack_graph is None:
        attack_graph = getattr(ctx, "attack_graph", None)
    fact_ids = kwargs.pop("predecessor_fact_ids", None)
    if fact_ids is None:
        fact_ids = getattr(ctx, "predecessor_fact_ids", ()) or ()
    top_k = kwargs.pop("top_k", None)
    if top_k is None:
        top_k = getattr(ctx, "top_k", DEFAULT_TOP_K)
    return run_s2(candidates, attack_graph, predecessor_fact_ids=fact_ids, top_k=top_k, **kwargs)


__all__ = [
    "STAGE",
    "STAGE_NAME",
    "DEFAULT_MAX_PATTERN_NODES",
    "DEFAULT_MAX_QUERY_NODES",
    "DEFAULT_TIMEOUT_S",
    "S2Candidate",
    "S2Result",
    "build_query_graph",
    "run_s2",
    "run_stage",
    "s2_bounded_isomorphism",
]
