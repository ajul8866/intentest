"""Goal anchoring and termination detection (plan C2, §11).

A project declares a *goal anchor*: either a regular expression matched against
the ``value`` of a ``FLAG`` fact (``re.fullmatch(regex, value, re.IGNORECASE)``)
or a structured predicate evaluated against a fact's attributes.

This module is pure: it never touches the database.  Callers (the worker agent
loop, the harvester and the scheduler) pass fact objects/mappings in and receive
a :class:`GoalMatch` describing the hit.

Contract
--------
``match_fact(fact, anchor)``           -> ``GoalMatch | None``
``find_goal_fact(facts, anchor)``      -> ``(fact, GoalMatch) | (None, None)``
``TerminationReason``                  -> GOAL | UNREACHABLE | STALLED | BUDGET | CANCELLED
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_FLAG_REGEX",
    "GOAL_FACT_KINDS",
    "TerminationReason",
    "GoalMatch",
    "GoalAnchor",
    "normalise_anchor",
    "match_fact",
    "find_goal_fact",
    "is_goal_fact_kind",
    "evaluate_predicate",
    "goal_reached",
]

#: Fallback anchor regex used by the project API when none is supplied.
DEFAULT_FLAG_REGEX = r"flag\{[^}]+\}"

#: Fact kinds that can satisfy a goal anchor.
GOAL_FACT_KINDS: Tuple[str, ...] = ("FLAG", "GOAL", "NOTE")


class TerminationReason(str, Enum):
    """Why an episode stopped (plan C12)."""

    GOAL = "GOAL"
    UNREACHABLE = "UNREACHABLE"
    STALLED = "STALLED"
    BUDGET = "BUDGET"
    CANCELLED = "CANCELLED"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class GoalMatch:
    """Result of a successful goal match."""

    matched: bool = True
    kind: str = "FLAG"
    value: Any = None
    fact_key: Optional[str] = None
    fact_id: Optional[str] = None
    regex: Optional[str] = None
    method: str = "flag_regex"
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "matched": self.matched,
            "kind": self.kind,
            "value": self.value,
            "fact_key": self.fact_key,
            "fact_id": self.fact_id,
            "regex": self.regex,
            "method": self.method,
            "details": dict(self.details or {}),
        }


# ---------------------------------------------------------------------------
# tolerant accessors (ORM rows, dataclasses and mappings)
# ---------------------------------------------------------------------------
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return default
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:  # pragma: no cover - defensive
            continue
        if value is not None:
            return value
    return default


def _name(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().upper()


def is_goal_fact_kind(kind: Any) -> bool:
    return _name(kind) in GOAL_FACT_KINDS


# ---------------------------------------------------------------------------
# anchor handling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GoalAnchor:
    """Normalised goal anchor."""

    flag_regex: Optional[str] = None
    fact_kind: str = "FLAG"
    predicate: Optional[Mapping[str, Any]] = None
    require_verified: bool = False
    min_confidence: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.flag_regex and not self.predicate

    def compiled(self) -> Optional[re.Pattern]:
        if not self.flag_regex:
            return None
        try:
            return re.compile(self.flag_regex, re.IGNORECASE)
        except re.error:  # invalid regex -> treat as no anchor
            return None


def normalise_anchor(anchor: Any = None) -> GoalAnchor:
    """Coerce ``dict``/ORM/str anchors into a :class:`GoalAnchor`."""

    if anchor is None:
        return GoalAnchor(flag_regex=DEFAULT_FLAG_REGEX)
    if isinstance(anchor, GoalAnchor):
        return anchor
    if isinstance(anchor, str):
        return GoalAnchor(flag_regex=anchor or DEFAULT_FLAG_REGEX)

    raw = anchor if isinstance(anchor, Mapping) else {}
    if not raw:
        for name in ("goal_anchor", "goal_anchor_fact", "attributes"):
            candidate = _attr(anchor, name)
            if isinstance(candidate, Mapping):
                raw = candidate
                break

    regex = _attr(raw, "flag_regex", "regex", "pattern")
    predicate = _attr(raw, "predicate", "matcher", "match")
    kind = _attr(raw, "fact_kind", "kind", default="FLAG") or "FLAG"
    require_verified = bool(_attr(raw, "require_verified", default=False))
    try:
        min_confidence = float(_attr(raw, "min_confidence", "confidence", default=0.0) or 0.0)
    except (TypeError, ValueError):
        min_confidence = 0.0

    if regex is None and not predicate:
        regex = DEFAULT_FLAG_REGEX
    return GoalAnchor(
        flag_regex=str(regex) if regex else None,
        fact_kind=_name(kind) or "FLAG",
        predicate=dict(predicate) if isinstance(predicate, Mapping) else None,
        require_verified=require_verified,
        min_confidence=min_confidence,
    )


# ---------------------------------------------------------------------------
# structured predicate
# ---------------------------------------------------------------------------
def evaluate_predicate(attributes: Any, predicate: Mapping[str, Any]) -> bool:
    """Evaluate a small declarative matcher against a fact's attributes.

    Supported keys (all optional, AND-combined):
      * ``kind`` / ``fact_kind`` – required fact kind
      * ``key`` – attribute name to test
      * ``equals`` / ``eq`` – exact value
      * ``contains`` – substring (case-insensitive for strings)
      * ``regex`` – ``re.search`` against ``str(value)``
      * ``in`` – membership list
      * ``all`` / ``any`` – nested predicate lists
    """

    if not predicate:
        return False
    attrs = attributes if isinstance(attributes, Mapping) else {}

    if "all" in predicate:
        nested = predicate.get("all") or []
        if not all(isinstance(p, Mapping) and evaluate_predicate(attrs, p) for p in nested):
            return False
    if "any" in predicate:
        nested = predicate.get("any") or []
        if not any(isinstance(p, Mapping) and evaluate_predicate(attrs, p) for p in nested):
            return False

    for key in ("kind", "fact_kind"):
        if key in predicate and _name(predicate[key]) != _name(attrs.get("kind")):
            return False

    attr_key = predicate.get("key") or predicate.get("attribute")
    if attr_key is None:
        return True  # only kind/nested constraints were requested

    value = attrs.get(attr_key)
    if value is None:
        return False

    if "equals" in predicate and value != predicate["equals"]:
        return False
    if "eq" in predicate and value != predicate["eq"]:
        return False
    if "contains" in predicate:
        needle = str(predicate["contains"]).lower()
        if needle not in str(value).lower():
            return False
    if "regex" in predicate:
        try:
            if not re.search(str(predicate["regex"]), str(value), re.IGNORECASE):
                return False
        except re.error:
            return False
    if "in" in predicate and value not in (predicate["in"] or []):
        return False
    return True


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
def match_fact(fact: Any, anchor: Any = None) -> Optional[GoalMatch]:
    """Return a :class:`GoalMatch` when ``fact`` satisfies the anchor."""

    goal = normalise_anchor(anchor)
    kind = _name(_attr(fact, "kind"))
    if goal.fact_kind and kind and kind != goal.fact_kind and kind != "FLAG":
        return None

    if goal.require_verified and not bool(_attr(fact, "verified", default=False)):
        return None
    try:
        confidence = float(_attr(fact, "confidence", default=1.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < goal.min_confidence:
        return None

    attrs = _attr(fact, "attributes", "attrs", default={}) or {}
    value = _attr(fact, "value")
    candidates = [value]
    if isinstance(attrs, Mapping):
        for key in ("flag", "value", "answer"):
            if attrs.get(key) is not None:
                candidates.append(attrs[key])

    if goal.predicate:
        merged = dict(attrs) if isinstance(attrs, Mapping) else {}
        merged.setdefault("kind", kind)
        if value is not None:
            merged.setdefault("value", value)
        if evaluate_predicate(merged, goal.predicate):
            return GoalMatch(
                kind=kind or goal.fact_kind,
                value=value,
                fact_key=_attr(fact, "fact_key"),
                fact_id=str(_attr(fact, "id", default="") or "") or None,
                method="predicate",
                details={"predicate": dict(goal.predicate)},
            )

    compiled = goal.compiled()
    if compiled is None:
        return None
    for candidate in candidates:
        if candidate is None:
            continue
        text = candidate if isinstance(candidate, str) else str(candidate)
        if compiled.fullmatch(text):
            return GoalMatch(
                kind=kind or goal.fact_kind,
                value=candidate,
                fact_key=_attr(fact, "fact_key"),
                fact_id=str(_attr(fact, "id", default="") or "") or None,
                regex=goal.flag_regex,
                method="flag_regex",
            )
    return None


def find_goal_fact(
    facts: Iterable[Any], anchor: Any = None
) -> Tuple[Optional[Any], Optional[GoalMatch]]:
    """Return the first ``(fact, match)`` pair satisfying the anchor."""

    for fact in facts or []:
        match = match_fact(fact, anchor)
        if match is not None:
            return fact, match
    return None, None


def goal_reached(facts: Iterable[Any], anchor: Any = None) -> bool:
    """Convenience boolean wrapper used by the harvester."""

    fact, _match = find_goal_fact(facts, anchor)
    return fact is not None
