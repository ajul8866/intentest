"""Merge the runtime attack graph ``G_a`` with the offline intent graph ``G_o``.

Paper §4.4 / plan C8 ("Merge"): the runtime attack graph is derived from the
verified facts and the two hyperedge families ``E_f→i`` / ``E_i→f``; the
offline graph is the MITRE-mapped intent pattern graph built from the corpus.
Retrieval (Algorithm 2) needs both: S1/S2 select pattern nodes from ``G_o`` but
the query is a *runtime* predecessor pattern, and S4 alignment needs the
runtime primary chain.

Cross-links are established with a strict, deterministic precedence and each
link is tagged with the method that produced it (``exact`` → ``cosine`` →
``jaccard``), so the Retrieval Trace UI can show *how* a runtime fact was
related to an offline pattern:

1. ``exact``   — exact ``(kind, entity, vulnerability)`` triple match.
2. ``cosine``  — cosine similarity of the two node embeddings ``>= τ`` (0.75).
3. ``jaccard`` — attribute-token Jaccard similarity ``>= JACCARD_THRESHOLD``.

Only the first method that fires is recorded for a pair (precedence order is
``CROSSLINK_PRECEDENCE`` from :mod:`app.core.intent`).

The module is deliberately dependency-light at import time: ``networkx`` and
SQLAlchemy are only imported inside functions so that ``import
app.core.intent.merge`` stays cheap for the eval harness and unit tests.

Public interface
----------------
- :class:`CrossLink`            — one ``(fact → pattern)`` correspondence.
- :func:`normalise_entity`      — canonical entity/target string.
- :func:`pattern_triple`        — ``(kind, entity, vuln)`` of a pattern row.
- :func:`fact_triple`           — ``(kind, entity, vuln)`` of a fact row.
- :func:`attribute_tokens`      — token set used by the Jaccard stage.
- :func:`jaccard`               — ``J(A,B) = |A∩B| / |A∪B|``.
- :func:`cross_link_pairs`      — compute the links between facts/patterns.
- :func:`merge_graphs`          — attach ``CORRESPONDS`` edges to ``G_a``.
- :class:`MergedGraph`          — result bundle (graph + links + stats).
- :func:`build_merged_graph`    — convenience builder from rows/session.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - constants module always exists in-tree
    from app.core.intent import CROSSLINK_PRECEDENCE, DEFAULT_PARAMETERS
except Exception:  # pragma: no cover - defensive
    CROSSLINK_PRECEDENCE = ("exact", "cosine", "jaccard")  # type: ignore[assignment]
    DEFAULT_PARAMETERS = {  # type: ignore[assignment]
        "crosslink_cosine": 0.75,
        "jaccard_threshold": 0.5,
    }

logger = logging.getLogger(__name__)

__all__ = [
    "CROSSLINK_COSINE",
    "DEFAULT_JACCARD_THRESHOLD",
    "EDGE_CORRESPONDS",
    "CrossLink",
    "MergedGraph",
    "attribute_tokens",
    "build_merged_graph",
    "cosine_similarity",
    "cross_link_pairs",
    "fact_triple",
    "jaccard",
    "merge_graphs",
    "normalise_entity",
    "pattern_triple",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EDGE_CORRESPONDS = "CORRESPONDS"

CROSSLINK_COSINE: float = float(
    DEFAULT_PARAMETERS.get("crosslink_cosine", 0.75)  # type: ignore[union-attr]
)
DEFAULT_JACCARD_THRESHOLD: float = float(
    DEFAULT_PARAMETERS.get("jaccard_threshold", 0.5)  # type: ignore[union-attr]
)

#: Attribute keys that carry the "entity" of a fact, in priority order.
_ENTITY_KEYS: Tuple[str, ...] = (
    "host",
    "hostname",
    "url",
    "uri",
    "endpoint",
    "path",
    "target",
    "target_entity",
    "entity",
    "ip",
    "address",
    "service",
    "port",
)
#: Attribute keys that carry a vulnerability identifier.
_VULN_KEYS: Tuple[str, ...] = (
    "vulnerability",
    "vulnerability_type",
    "vuln",
    "vuln_type",
    "cwe",
    "cve",
    "class",
    "type",
    "category",
)
#: Attributes that must not participate in similarity computation.
_VOLATILE_KEYS = frozenset(
    {
        "timestamp",
        "ts",
        "scanned_at",
        "observed_at",
        "request_id",
        "duration_ms",
        "elapsed_ms",
        "nonce",
        "date",
        "time",
        "created_at",
        "updated_at",
        "id",
        "project_id",
    }
)
_WS_RE = re.compile(r"\s+")
_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-zA-Z_.:/-]+")


# ---------------------------------------------------------------------------
# Tolerant accessors
# ---------------------------------------------------------------------------
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first present attribute/key from ``obj``."""
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if value is not None:
            return value
    return default


def _name(value: Any) -> str:
    """Normalise an enum/string to an upper-case name."""
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().upper()


def _as_mapping(value: Any) -> Dict[str, Any]:
    """Best-effort conversion of an attributes blob into a plain dict."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str):
        try:
            import json

            parsed = json.loads(value)
        except Exception:
            return {"value": value}
        return {str(k): v for k, v in parsed.items()} if isinstance(parsed, Mapping) else {"value": parsed}
    if hasattr(value, "model_dump"):
        try:
            data = value.model_dump()
            if isinstance(data, Mapping):
                return {str(k): v for k, v in data.items()}
        except Exception:
            pass
    try:
        return {str(k): v for k, v in vars(value).items() if not str(k).startswith("_")}
    except Exception:
        return {}


def _maybe_await(value: Any) -> Any:
    """Return an awaitable marker if ``value`` is awaitable (caller awaits)."""
    return value


async def _resolve(value: Any) -> Any:
    import inspect

    if inspect.isawaitable(value):
        return await value
    return value


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------
def normalise_entity(value: Any) -> str:
    """Canonical entity string used for exact cross-link matching.

    Lower-cases, NFKC-normalises, strips scheme/default-port noise from URLs
    and collapses whitespace so ``HTTPS://Host.Example:443/login`` and
    ``host.example/login`` compare equal.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            text = normalise_entity(item)
            if text:
                return text
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).strip()
    lowered = text.lower()
    for prefix in ("https://", "http://", "//"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            text = text[len(prefix) :]
            break
    lowered = lowered.rstrip("/")
    # drop default ports
    lowered = re.sub(r":(80|443)(?=/|$)", "", lowered)
    lowered = _WS_RE.sub(" ", lowered).strip()
    return lowered


def _entity_from(attributes: Mapping[str, Any], fallback: Any = None) -> str:
    for key in _ENTITY_KEYS:
        if key in attributes and attributes[key] is not None:
            text = normalise_entity(attributes[key])
            if text:
                return text
    return normalise_entity(fallback)


def _vuln_from(attributes: Mapping[str, Any], fallback: Any = None) -> str:
    for key in _VULN_KEYS:
        if key in attributes and attributes[key] is not None:
            text = normalise_entity(attributes[key])
            if text:
                return text
    return normalise_entity(fallback)


def fact_triple(fact: Any) -> Tuple[str, str, str]:
    """Return the ``(kind, entity, vulnerability)`` triple of a fact."""
    attributes = _as_mapping(_attr(fact, "attributes", "attrs"))
    kind = _name(_attr(fact, "kind", "fact_kind"))
    entity = _entity_from(
        attributes,
        _attr(fact, "target_entity", "entity", "target"),
    )
    vuln = _vuln_from(attributes, _attr(fact, "vulnerability_type", "vuln"))
    return kind, entity, vuln


def pattern_triple(pattern: Any) -> Tuple[str, str, str]:
    """Return the ``(kind, entity, vulnerability)`` triple of a pattern row."""
    attributes = _as_mapping(_attr(pattern, "preconditions", "attributes"))
    kind = _name(
        _attr(
            pattern,
            "target_entity_type",
            "target_kind",
            "kind",
            "fact_kind",
        )
    )
    entity = _entity_from(
        attributes,
        _attr(pattern, "target_entity", "target", "entity", "name"),
    )
    vuln = _vuln_from(
        attributes,
        _attr(pattern, "vulnerability_type", "vuln", "mitre_attack_id"),
    )
    return kind, entity, vuln


def _tokens_from(value: Any) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        out: List[str] = []
        for item in value:
            out.extend(_tokens_from(item))
        return out
    if isinstance(value, Mapping):
        out = []
        for key, item in value.items():
            key_text = normalise_entity(key)
            if key_text:
                out.append(f"{key_text}:")
            out.extend(_tokens_from(item))
        return out
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    if not text:
        return ()
    parts = [p for p in _TOKEN_SPLIT_RE.split(text) if p]
    return tuple(parts)


def attribute_tokens(row: Any, *, include_labels: bool = True) -> frozenset:
    """Token set of a fact/pattern row used by the Jaccard fallback stage.

    The set is the union of: the kind, entity, vulnerability, every
    ``key=value`` attribute pair (keys and values), and (optionally) the
    textual label fields (``tactic``/``objective``/``name``).
    """
    tokens: set = set()
    attributes = _as_mapping(_attr(row, "attributes", "preconditions", "attrs"))
    for key, value in attributes.items():
        key_text = str(key).strip().lower()
        if key_text in _VOLATILE_KEYS:
            continue
        if key_text:
            tokens.add(key_text)
        for token in _tokens_from(value):
            tokens.add(token)
        if not isinstance(value, (Mapping, list, tuple, set, frozenset)):
            primitive = normalise_entity(value)
            if primitive:
                tokens.add(f"{key_text}={primitive}")
    for key in ("kind", "fact_kind", "target_entity_type"):
        text = _name(_attr(row, key))
        if text:
            tokens.add(text.lower())
    for key in (
        "target_entity",
        "entity",
        "target",
        "vulnerability_type",
        "vuln",
        "mitre_attack_id",
    ):
        text = normalise_entity(_attr(row, key))
        if text:
            tokens.add(text)
    if include_labels:
        for key in ("tactic", "objective", "name", "action_type", "label"):
            text = _attr(row, key)
            if text is None:
                continue
            for token in _tokens_from(text):
                tokens.add(token)
    return frozenset(t for t in tokens if t)


def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    """``J(A,B) = |A∩B| / |A∪B|`` with a zero-safe denominator."""
    set_a = set(a or ())
    set_b = set(b or ())
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Eq. 2 — cosine similarity, zero-safe."""
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(size):
        try:
            x = float(a[i])
            y = float(b[i])
        except Exception:
            continue
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# ---------------------------------------------------------------------------
# Cross links
# ---------------------------------------------------------------------------
@dataclass
class CrossLink:
    """One ``fact → pattern`` correspondence edge."""

    fact_id: str
    pattern_id: str
    method: str
    score: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def edge_id(self) -> str:
        return f"corr:{self.fact_id}->{self.pattern_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.edge_id,
            "fact_id": self.fact_id,
            "pattern_id": self.pattern_id,
            "method": self.method,
            "score": float(self.score),
            "relation": EDGE_CORRESPONDS,
            "detail": dict(self.detail),
        }


def _id_of(row: Any) -> str:
    value = _attr(row, "id", "uuid", "row_id")
    return "" if value is None else str(value)


def _embedding_of(row: Any) -> Optional[List[float]]:
    raw = _attr(row, "embedding", "vector", "obj_embedding")
    if raw is None:
        return None
    try:
        return [float(x) for x in raw]
    except Exception:
        return None


def cross_link_pairs(
    facts: Iterable[Any],
    patterns: Iterable[Any],
    *,
    embeddings: Optional[Mapping[str, Sequence[float]]] = None,
    cosine_threshold: float = CROSSLINK_COSINE,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    methods: Sequence[str] = CROSSLINK_PRECEDENCE,
    max_links_per_fact: int = 3,
) -> List[CrossLink]:
    """Compute cross-links between runtime facts and offline patterns.

    Each ``(fact, pattern)`` pair is tested with the methods in
    ``CROSSLINK_PRECEDENCE`` order; the first method that fires is recorded and
    later methods are not evaluated for that pair (so the tag is unambiguous).
    """
    links: List[CrossLink] = []
    methods_norm = tuple(str(m).lower() for m in methods)
    pattern_rows = list(patterns)
    if not pattern_rows:
        return links

    token_cache: Dict[str, frozenset] = {}
    embedding_map: Dict[str, List[float]] = {}
    if embeddings:
        for key, vector in embeddings.items():
            try:
                embedding_map[str(key)] = [float(x) for x in vector]
            except Exception:
                continue

    def _tokens(row: Any) -> frozenset:
        key = _id_of(row)
        if key and key in token_cache:
            return token_cache[key]
        value = attribute_tokens(row)
        if key:
            token_cache[key] = value
        return value

    for fact in facts:
        fact_id = _id_of(fact)
        if not fact_id:
            continue
        fact_kind, fact_entity, fact_vuln = fact_triple(fact)
        fact_tokens = _tokens(fact)
        fact_vector = embedding_map.get(fact_id) or _embedding_of(fact)

        per_fact: List[CrossLink] = []
        for pattern in pattern_rows:
            pattern_id = _id_of(pattern)
            if not pattern_id:
                continue
            pat_kind, pat_entity, pat_vuln = pattern_triple(pattern)
            method = ""
            score = 0.0
            detail: Dict[str, Any] = {}

            # Stage 1 — exact (kind, entity, vulnerability) triple.
            if "exact" in methods_norm:
                if (
                    fact_kind
                    and pat_kind
                    and fact_kind == pat_kind
                    and fact_vuln
                    and pat_vuln
                    and fact_vuln == pat_vuln
                    and (
                        not fact_entity
                        or not pat_entity
                        or fact_entity == pat_entity
                    )
                ):
                    method = "exact"
                    score = 1.0
                    detail = {
                        "kind": fact_kind,
                        "entity": fact_entity or pat_entity,
                        "vulnerability": fact_vuln,
                    }

            # Stage 2 — embedding cosine similarity.
            if not method and "cosine" in methods_norm:
                pattern_vector = (
                    embedding_map.get(pattern_id) or _embedding_of(pattern)
                )
                if fact_vector and pattern_vector:
                    value = cosine_similarity(fact_vector, pattern_vector)
                    if value >= cosine_threshold:
                        method = "cosine"
                        score = value
                        detail = {"cosine": round(value, 6)}

            # Stage 3 — attribute Jaccard fallback.
            if not method and "jaccard" in methods_norm:
                pattern_tokens = _tokens(pattern)
                value = jaccard(fact_tokens, pattern_tokens)
                if value >= jaccard_threshold:
                    method = "jaccard"
                    score = value
                    detail = {"jaccard": round(value, 6)}

            if method:
                per_fact.append(
                    CrossLink(
                        fact_id=fact_id,
                        pattern_id=pattern_id,
                        method=method,
                        score=score,
                        detail=detail,
                    )
                )

        # Keep only the strongest links per fact to bound the merged graph.
        per_fact.sort(key=lambda link: (-link.score, link.pattern_id))
        links.extend(per_fact[: max(1, int(max_links_per_fact))])

    links.sort(key=lambda link: (link.fact_id, link.pattern_id, link.method))
    return links


# ---------------------------------------------------------------------------
# Merged graph
# ---------------------------------------------------------------------------
@dataclass
class MergedGraph:
    """Result of merging ``G_a`` with ``G_o``."""

    graph: Any
    links: List[CrossLink] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)
    patterns: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def nodes(self) -> int:
        try:
            return int(self.graph.number_of_nodes())
        except Exception:
            return len(self.facts) + len(self.patterns)

    @property
    def edges(self) -> int:
        try:
            return int(self.graph.number_of_edges())
        except Exception:
            return len(self.links)

    def cross_links(self) -> List[Dict[str, Any]]:
        return [link.to_dict() for link in self.links]

    def to_dict(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for link in self.links:
            counts[link.method] = counts.get(link.method, 0) + 1
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "links": len(self.links),
            "facts": len(self.facts),
            "patterns": len(self.patterns),
            "by_method": counts,
            "metadata": dict(self.metadata),
        }


def _pattern_node_id(pattern_id: str) -> str:
    return f"p:{pattern_id}"


def merge_graphs(
    attack_graph: Any,
    patterns: Iterable[Any] = (),
    *,
    embeddings: Optional[Mapping[str, Sequence[float]]] = None,
    cosine_threshold: float = CROSSLINK_COSINE,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    include_pattern_nodes: bool = True,
    max_links_per_fact: int = 3,
) -> MergedGraph:
    """Attach ``CORRESPONDS`` edges from ``G_a`` facts to ``G_o`` patterns.

    ``attack_graph`` may be an :class:`app.core.intent.attack_graph.AttackGraph`
    (anything with ``.graph``/``.facts``), a raw ``networkx.DiGraph`` or a
    mapping with ``facts``/``patterns`` keys. The input graph is **not**
    mutated: a copy is returned when possible.
    """
    graph = getattr(attack_graph, "graph", None)
    if graph is None and hasattr(attack_graph, "number_of_nodes"):
        graph = attack_graph
    facts: Dict[str, Any] = dict(
        getattr(attack_graph, "facts", {}) or {}
    ) if not isinstance(attack_graph, Mapping) else dict(
        attack_graph.get("facts") or {}
    )
    pattern_rows = list(patterns)
    pattern_map: Dict[str, Any] = {}
    for pattern in pattern_rows:
        pid = _id_of(pattern)
        if pid:
            pattern_map[pid] = pattern

    links: List[CrossLink] = []
    merged = graph
    if graph is not None:
        try:
            merged = graph.copy()
        except Exception:
            merged = graph

    if merged is not None and hasattr(merged, "add_node"):
        for pattern_id, pattern in pattern_map.items():
            if include_pattern_nodes:
                try:
                    merged.add_node(
                        _pattern_node_id(pattern_id),
                        node_type="pattern",
                        pattern_id=pattern_id,
                        name=_attr(pattern, "name"),
                        tactic=_attr(pattern, "tactic"),
                        mitre_attack_id=_attr(pattern, "mitre_attack_id"),
                        vulnerability_type=_attr(pattern, "vulnerability_type"),
                        confirmed=bool(_attr(pattern, "confirmed", default=True)),
                    )
                except Exception:
                    pass

    fact_rows = list(facts.values())
    if not fact_rows and isinstance(attack_graph, Mapping):
        fact_rows = list(attack_graph.get("fact_rows") or [])

    if fact_rows and pattern_map:
        links = cross_link_pairs(
            fact_rows,
            pattern_map.values(),
            embeddings=embeddings,
            cosine_threshold=cosine_threshold,
            jaccard_threshold=jaccard_threshold,
            max_links_per_fact=max_links_per_fact,
        )

    if merged is not None and hasattr(merged, "add_edge"):
        for link in links:
            source = f"f:{link.fact_id}"
            target = _pattern_node_id(link.pattern_id)
            try:
                merged.add_edge(
                    source,
                    target,
                    relation=EDGE_CORRESPONDS,
                    method=link.method,
                    score=link.score,
                    fact_id=link.fact_id,
                    pattern_id=link.pattern_id,
                    id=link.edge_id,
                )
            except Exception:
                continue

    metadata = {
        "patterns": len(pattern_map),
        "facts": len(fact_rows),
        "links": len(links),
        "cosine_threshold": cosine_threshold,
        "jaccard_threshold": jaccard_threshold,
        "methods": list(CROSSLINK_PRECEDENCE),
    }
    return MergedGraph(
        graph=merged,
        links=links,
        facts=facts,
        patterns=pattern_map,
        metadata=metadata,
    )


async def _load_patterns(session: Any, *, limit: Optional[int] = None) -> List[Any]:
    """Load offline patterns through the repository or a raw select."""
    try:
        from app.db.repositories.pattern_repo import PatternRepository

        repo = PatternRepository(session)
        for name in ("list", "list_all", "list_patterns", "all", "list_confirmed"):
            method = getattr(repo, name, None)
            if method is None:
                continue
            try:
                rows = await _resolve(method(limit=limit) if limit else method())
            except TypeError:
                rows = await _resolve(method())
            if rows is not None:
                return list(rows)
    except Exception:  # pragma: no cover - repository optional
        pass
    try:
        from sqlalchemy import select

        from app.db.models.intent_pattern import IntentPattern

        statement = select(IntentPattern)
        if limit:
            statement = statement.limit(int(limit))
        result = await _resolve(session.execute(statement))
        scalars = getattr(result, "scalars", None)
        if callable(scalars):
            return list(scalars().all())
        return list(result)
    except Exception as exc:  # pragma: no cover - DB optional
        logger.debug("merge.pattern_load_failed", extra={"error": str(exc)})
        return []


async def build_merged_graph(
    session: Any = None,
    *,
    project_id: Any = None,
    attack_graph: Any = None,
    patterns: Optional[Iterable[Any]] = None,
    embeddings: Optional[Mapping[str, Sequence[float]]] = None,
    cosine_threshold: float = CROSSLINK_COSINE,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    include_pattern_nodes: bool = True,
    limit: Optional[int] = None,
) -> MergedGraph:
    """Build ``G_a`` (if not supplied), load ``G_o`` patterns and merge them."""
    graph = attack_graph
    if graph is None:
        try:
            from app.core.intent.attack_graph import build_attack_graph

            graph = await build_attack_graph(session, project_id=project_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("merge.attack_graph_failed", extra={"error": str(exc)})
            graph = None

    pattern_rows = list(patterns) if patterns is not None else []
    if patterns is None and session is not None:
        pattern_rows = await _load_patterns(session, limit=limit)

    merged = merge_graphs(
        graph,
        pattern_rows,
        embeddings=embeddings,
        cosine_threshold=cosine_threshold,
        jaccard_threshold=jaccard_threshold,
        include_pattern_nodes=include_pattern_nodes,
    )
    merged.metadata["project_id"] = str(project_id) if project_id else None
    return merged
