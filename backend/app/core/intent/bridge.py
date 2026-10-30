"""Algorithm 3 -- the asynchronous intent bridge (paper Sec. 4.5, plan C10).

After an EXPLORE intent concludes successfully with >= 1 new fact, the
retrieval pipeline (Algorithm 2) may have produced a recommendation ``i*``.
This module turns that recommendation into a *three block* textual payload
which is injected into the next EXPLORE's context window.

Activation guard (all must hold)
--------------------------------
1. the concluded intent is an EXPLORE whose ``success`` is true,
2. it produced at least one new fact (``result.new_fact_ids``),
3. a recommendation ``i*`` exists (retrieval accepted it).

Similarity gate
---------------
``s = cos(state_vector, embed(i*.objective))``; if ``s < BRIDGE_THETA (0.3)``
the payload is dropped (``bridge.dropped``).

Dedupe
------
``sha256(i*.id | hash_blocks(B1, B2, B3))`` -- a repeated recommendation with
identical blocks is injected at most once (see ``core.intent.dedupe``).

Metadata stripping
------------------
``strip_meta`` is an **allow-list serializer**: it copies only known-safe keys
and structurally removes similarity scores, source-report names, matching
method details and stage indices.  It is never a regex scrub.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from app.core.intent import BRIDGE_BLOCK_NAMES, DEFAULT_PARAMETERS
from app.core.intent.dedupe import bridge_dedupe_key, canonical_json, hash_blocks

log = logging.getLogger(__name__)

__all__ = [
    "BRIDGE_THETA",
    "BridgeDecision",
    "BridgePayload",
    "strip_meta",
    "activation_guard",
    "build_blocks",
    "build_bridge",
    "preview_bridge",
    "inject_bridge",
    "bridge_summary",
]

#: Paper-stated activation threshold (plan C10).
BRIDGE_THETA: float = 0.3

_PAPER_DEFAULT_THETA = float(DEFAULT_PARAMETERS.get("bridge_theta", 0.3))
_MAX_ADJACENT = int(DEFAULT_PARAMETERS.get("bridge_max_adjacent_patterns", 3))
_RECENT_FACTS = int(DEFAULT_PARAMETERS.get("bridge_recent_facts", 10))

#: Allow-list of keys that may reach the LLM context (``strip_meta``).
_SAFE_RECOMMENDATION_KEYS: Tuple[str, ...] = (
    "action_type",
    "tactic",
    "objective",
    "target_entity",
    "vulnerability_type",
    "mitre_attack_id",
    "preconditions",
    "action_spec",
    "confidence",
)
_SAFE_LINK_KEYS: Tuple[str, ...] = (
    "name",
    "target_entity_type",
    "vulnerability_type",
    "tactic",
    "mitre_attack_id",
    "confidence",
    "relation",
)
#: Keys that must never survive serialisation (belt-and-braces; the allow-list
#: above already excludes them, this is asserted in tests).
_FORBIDDEN_KEY_MARKERS: Tuple[str, ...] = (
    "score",
    "similarity",
    "cosine",
    "jaccard",
    "method",
    "matching_method",
    "stage",
    "report",
    "source_doc",
    "source_type",
    "trace",
    "pattern_id",
    "embedding",
)


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Tolerant attribute / mapping accessor."""
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping) and name in obj and obj[name] is not None:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Eq. 2 -- ``(a . b) / (||a|| ||b||)``; 0.0 on degenerate input."""
    try:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for i in range(n):
            x = float(a[i])
            y = float(b[i])
            dot += x * y
            norm_a += x * x
            norm_b += y * y
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    except Exception:  # pragma: no cover - defensive
        return 0.0


def _is_forbidden(key: str) -> bool:
    lowered = str(key).lower()
    return any(marker in lowered for marker in _FORBIDDEN_KEY_MARKERS)


def strip_meta(value: Any, *, _depth: int = 0) -> Any:
    """Allow-list serializer used on every bridge payload.

    Only whitelisted keys survive; nested mappings are dropped wholesale if any
    of their keys looks like retrieval metadata (score/method/stage/report).
    """
    if _depth > 6 or value is None:
        return None
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            key = str(key)
            if _is_forbidden(key):
                continue
            if key in _SAFE_RECOMMENDATION_KEYS or key in _SAFE_LINK_KEYS:
                out[key] = strip_meta(item, _depth=_depth + 1)
            elif key in ("attributes", "action_spec", "preconditions", "value"):
                # structured payloads are kept but scrubbed recursively
                out[key] = _scrub(item, _depth=_depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [strip_meta(item, _depth=_depth + 1) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _scrub(value: Any, *, _depth: int = 0) -> Any:
    """Recursively remove forbidden keys but keep all other data."""
    if _depth > 6:
        return None
    if isinstance(value, Mapping):
        return {
            str(k): _scrub(v, _depth=_depth + 1)
            for k, v in value.items()
            if not _is_forbidden(str(k))
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(item, _depth=_depth + 1) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class BridgePayload:
    """The stored artefact (``intent.injected_context``)."""

    intent_id: Optional[str]
    source_intent_id: Optional[str]
    blocks: Dict[str, str] = field(default_factory=dict)
    dedupe_key: str = ""
    score: float = 0.0
    theta: float = BRIDGE_THETA
    injected: bool = False
    reason: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(
            self.blocks[name] for name in BRIDGE_BLOCK_NAMES if self.blocks.get(name)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "source_intent_id": self.source_intent_id,
            "blocks": dict(self.blocks),
            "block_text": self.text,
            "dedupe_key": self.dedupe_key,
            "score": self.score,
            "theta": self.theta,
            "injected": self.injected,
            "reason": self.reason,
        }


@dataclass
class BridgeDecision:
    """Outcome of Algorithm 3 for one concluded EXPLORE."""

    activated: bool
    injected: bool
    reason: str
    score: float = 0.0
    payload: Optional[BridgePayload] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activated": self.activated,
            "injected": self.injected,
            "reason": self.reason,
            "score": self.score,
            "payload": self.payload.to_dict() if self.payload else None,
        }


def activation_guard(
    concluded_intent: Any,
    recommendation: Any,
    *,
    new_fact_ids: Optional[Iterable[Any]] = None,
) -> Tuple[bool, str]:
    """Return ``(activated, reason)`` for Algorithm 3's first guard."""
    if recommendation is None:
        return False, "no_recommendation"
    if concluded_intent is None:
        return False, "no_intent"
    action_type = _name(_attr(concluded_intent, "action_type")).upper()
    if action_type and action_type != "EXPLORE":
        return False, "not_explore"
    success = _attr(concluded_intent, "success", default=None)
    if success is False:
        return False, "explore_failed"
    if success is None and action_type == "EXPLORE":
        # older rows: infer success from produced facts
        success = bool(_as_list(new_fact_ids or _attr(concluded_intent, "new_fact_ids")))
    facts = _as_list(new_fact_ids)
    if not facts:
        result = _attr(concluded_intent, "result", default={}) or {}
        facts = _as_list(_attr(result, "new_fact_ids", default=None))
    if not facts:
        return False, "no_new_facts"
    return True, "ok"


def _pattern_line(pattern: Any) -> str:
    name = _name(_attr(pattern, "name", default="")) or "pattern"
    vuln = _name(_attr(pattern, "vulnerability_type", default="")) or "unknown"
    tactic = _name(_attr(pattern, "tactic", default="")) or "unknown"
    mitre = _name(_attr(pattern, "mitre_attack_id", default="")) or "T0000"
    return f"- {name}: {tactic} -> {vuln} [{mitre}]"


def build_blocks(
    recommendation: Any,
    *,
    goal_text: str = "",
    phase: str = "",
    allowed_targets: Sequence[str] = (),
    recent_facts: Sequence[Any] = (),
    pending_intents: Sequence[Any] = (),
    historical_chain: Sequence[Any] = (),
    adjacent_patterns: Sequence[Any] = (),
    cross_links: Sequence[Any] = (),
) -> Dict[str, str]:
    """Build the three bridge blocks (B1 / B2 / B3)."""
    rec = strip_meta(recommendation)
    tactic = rec.get("tactic") or "EXPLORE"
    objective = rec.get("objective") or "continue exploration of the current target"
    target = rec.get("target_entity") or "the authorised target"
    vuln = rec.get("vulnerability_type") or ""
    mitre = rec.get("mitre_attack_id") or ""

    b1 = [
        "### B1 - Recommended next tactical action",
        f"Tactic: {tactic}",
        f"Objective: {objective}",
        f"Target: {target}",
    ]
    if vuln:
        b1.append(f"Vulnerability class: {vuln}")
    if mitre:
        b1.append(f"MITRE ATT&CK: {mitre}")
    preconditions = rec.get("preconditions")
    if preconditions:
        b1.append("Preconditions satisfied: " + ", ".join(_as_list(preconditions)))
    b1.append(
        "Treat this as advice only: verify every claim against tool output "
        "before acting."
    )

    b2 = ["### B2 - Supporting evidence chain"]
    if historical_chain:
        b2.append("Historical chain (offline knowledge graph):")
        b2.extend(_pattern_line(item) for item in _as_list(historical_chain)[:8])
    if adjacent_patterns:
        b2.append("Adjacent patterns (Jaccard neighbourhood, top-3):")
        b2.extend(_pattern_line(item) for item in _as_list(adjacent_patterns)[:_MAX_ADJACENT])
    if cross_links:
        b2.append("Cross-links to observed facts:")
        for link in _as_list(cross_links)[:5]:
            name = _name(_attr(link, "name", default="")) or "link"
            rel = _name(_attr(link, "relation", default="")) or "ENABLES"
            b2.append(f"- {name} ({rel})")
    if len(b2) == 1:
        b2.append("- no historical precedent available for this target")

    b3 = ["### B3 - Current project context"]
    b3.append(f"Goal: {goal_text or 'obtain the flag from the target application'}")
    b3.append(f"Phase: {phase or 'EXPLORE'}")
    if allowed_targets:
        b3.append("Allowed targets: " + ", ".join(str(t) for t in allowed_targets))
    facts = _as_list(recent_facts)[:_RECENT_FACTS]
    if facts:
        b3.append("Recent verified facts:")
        for fact in facts:
            kind = _name(_attr(fact, "kind", default="")) or "FACT"
            value = _attr(fact, "value", default="")
            if value in (None, ""):
                value = _attr(fact, "attributes", default="")
            text = canonical_json(value) if not isinstance(value, str) else value
            b3.append(f"- {kind}: {text[:180]}")
    pending = _as_list(pending_intents)[:10]
    if pending:
        b3.append("Pending intents:")
        for intent in pending:
            b3.append(
                f"- {_name(_attr(intent, 'tactic', default='EXPLORE'))}: "
                f"{_name(_attr(intent, 'objective', default=''))[:120]}"
            )

    return {"B1": "\n".join(b1), "B2": "\n".join(b2), "B3": "\n".join(b3)}


async def _embed_objective(recommendation: Any, embedder: Any = None) -> Optional[List[float]]:
    if embedder is None:
        return None
    text = _name(_attr(recommendation, "objective", default="")) or _name(
        _attr(recommendation, "name", default="")
    )
    if not text:
        return None
    try:
        vector = embedder.embed(text)
        if hasattr(vector, "__await__"):
            vector = await vector
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("bridge.embed_failed", extra={"error": str(exc)})
        return None
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    try:
        return [float(x) for x in vector]
    except Exception:
        return None


async def build_bridge(
    concluded_intent: Any,
    recommendation: Any,
    *,
    state_vector: Optional[Sequence[float]] = None,
    embedder: Any = None,
    goal_text: str = "",
    phase: str = "",
    allowed_targets: Sequence[str] = (),
    recent_facts: Sequence[Any] = (),
    pending_intents: Sequence[Any] = (),
    historical_chain: Sequence[Any] = (),
    adjacent_patterns: Sequence[Any] = (),
    cross_links: Sequence[Any] = (),
    new_fact_ids: Optional[Iterable[Any]] = None,
    theta: float = BRIDGE_THETA,
    seen_keys: Any = None,
) -> BridgeDecision:
    """Algorithm 3 end-to-end (activation -> theta -> dedupe -> blocks -> strip)."""
    activated, reason = activation_guard(
        concluded_intent, recommendation, new_fact_ids=new_fact_ids
    )
    if not activated:
        return BridgeDecision(False, False, reason)

    score = 1.0
    if state_vector is not None:
        objective_vector = await _embed_objective(recommendation, embedder)
        if objective_vector is not None:
            score = _cosine(state_vector, objective_vector)
        if score < float(theta):
            log.info(
                "bridge.dropped",
                extra={"reason": "below_theta", "score": score, "theta": theta},
            )
            return BridgeDecision(
                True, False, f"below_theta({score:.4f}<{theta})", score=score
            )

    blocks = build_blocks(
        recommendation,
        goal_text=goal_text,
        phase=phase,
        allowed_targets=allowed_targets,
        recent_facts=recent_facts,
        pending_intents=pending_intents,
        historical_chain=historical_chain,
        adjacent_patterns=adjacent_patterns,
        cross_links=cross_links,
    )
    rec_id = _attr(recommendation, "id", "pattern_id", default=None)
    key = bridge_dedupe_key(rec_id or "anonymous", blocks)

    if seen_keys is not None:
        try:
            if not seen_keys.add(key):
                log.info("bridge.deduplicated", extra={"dedupe_key": key})
                return BridgeDecision(True, False, "duplicate", score=score)
        except Exception:  # pragma: no cover - defensive
            pass

    payload = BridgePayload(
        intent_id=_attr(recommendation, "intent_id", default=None),
        source_intent_id=_attr(concluded_intent, "id", default=None),
        blocks=blocks,
        dedupe_key=key,
        score=score,
        theta=float(theta),
        injected=True,
        reason="ok",
    )
    log.info(
        "bridge.injected",
        extra={
            "source_intent_id": payload.source_intent_id,
            "dedupe_key": key,
            "score": score,
        },
    )
    return BridgeDecision(True, True, "ok", score=score, payload=payload)


async def preview_bridge(
    recommendation: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build blocks without an activation guard (used by the REST preview)."""
    blocks = build_blocks(recommendation, **{
        k: kwargs.get(k, v)
        for k, v in (
            ("goal_text", ""),
            ("phase", ""),
            ("allowed_targets", ()),
            ("recent_facts", ()),
            ("pending_intents", ()),
            ("historical_chain", ()),
            ("adjacent_patterns", ()),
            ("cross_links", ()),
        )
    })
    key = bridge_dedupe_key(_attr(recommendation, "id", "pattern_id", default="anonymous"), blocks)
    return {
        "injected": False,
        "reason": "preview",
        "score": kwargs.get("score", 0.0),
        "blocks": blocks,
        "payload": {"blocks": blocks, "block_text": "\n\n".join(blocks.values()), "dedupe_key": key},
        "dedupe_key": key,
    }


def inject_bridge(target_intent: Any, payload: BridgePayload) -> Any:
    """Attach the payload to ``intent.injected_context`` (in-memory helper)."""
    if target_intent is None or payload is None or not payload.injected:
        return target_intent
    document = {
        "source_intent_id": payload.source_intent_id,
        "blocks": dict(payload.blocks),
        "dedupe_key": payload.dedupe_key,
        "text": payload.text,
    }
    meta = {
        "bridge": True,
        "dedupe_key": payload.dedupe_key,
        "theta": payload.theta,
        "source_intent_id": payload.source_intent_id,
    }
    if isinstance(target_intent, dict):
        target_intent["injected_context"] = document
        target_intent["injection_meta"] = meta
    else:
        try:
            target_intent.injected_context = document
            target_intent.injection_meta = meta
        except Exception:  # pragma: no cover - read-only row
            return target_intent
    return target_intent


def bridge_summary(decision: BridgeDecision) -> Dict[str, Any]:
    """Compact view for the WS event payload."""
    document = decision.to_dict()
    if document.get("payload"):
        document["payload"].pop("blocks", None)
    return document
