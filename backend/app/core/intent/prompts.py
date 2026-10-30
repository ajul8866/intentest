"""Prompt registry for the intent-knowledge plane (offline graph + Algorithm 2/3).

The paper publishes only the frozen ``EMERGENCY_TRUNCATION`` content and the four
extractor constraints; every other prompt is authored here from the role
descriptions in the paper (§4.2–4.4) and kept versioned on disk under
``backend/app/prompts/`` so the behaviour is auditable.

This module is a *loading and version-tracking* layer:

* every prompt file lives in ``backend/app/prompts/<name>.md`` and starts with a
  YAML-ish front-matter block (``---`` fence) carrying ``name``, ``version``,
  ``role``, ``temperature`` and ``description``;
* the body is the template itself, using ``{{placeholder}}`` markers;
* :func:`load_prompt` renders a template with a strict placeholder substitution
  (unknown placeholders are left untouched, missing values become ``""``);
* :func:`prompt_metadata` exposes the front matter for the trace / report so a
  retrieval trace records *which* prompt version produced an S5 recommendation.

Nothing here performs I/O at import time: the registry is populated lazily and
cache-invalidated by file mtime, which keeps ``import app.core.intent.prompts``
cheap (alembic, eval harness, unit tests).
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "PROMPTS_DIR",
    "PROMPT_NAMES",
    "FROZEN_PROMPTS",
    "PromptInfo",
    "PromptError",
    "PromptNotFound",
    "prompt_path",
    "available_prompts",
    "load_prompt",
    "load_raw_prompt",
    "prompt_metadata",
    "render_template",
    "extract_placeholders",
    "prompt_hash",
    "registry_snapshot",
    "clear_cache",
    "STAGE5_LLM_PROMPT",
    "EXTRACTOR_RULES",
    "BRIDGE_BLOCK_PROMPTS",
    "EMERGENCY_TRUNCATION_PROMPT",
    "SYSTEM_ROLE_PROMPTS",
]

# ---------------------------------------------------------------------------
# registry layout
# ---------------------------------------------------------------------------

#: Directory that holds the versioned prompt templates.
PROMPTS_DIR: Path = Path(__file__).resolve().parents[2] / "prompts"

#: Every prompt file the knowledge plane / worker consider canonical.
PROMPT_NAMES: Tuple[str, ...] = (
    "system_executor",
    "system_planner",
    "system_extractor",
    "emergency_truncation",
    "bridge_block1",
    "bridge_block2",
    "bridge_block3",
    "stage5_heuristic",
)

#: Prompts whose wording is quoted from the paper and therefore must not be
#: paraphrased by this reproduction. ``load_prompt`` still allows rendering but
#: :func:`prompt_metadata` flags them so the report can say "frozen".
FROZEN_PROMPTS: Tuple[str, ...] = ("emergency_truncation",)

#: Convenience handles used by S5, the offline extractor and the bridge.
STAGE5_LLM_PROMPT: str = "stage5_heuristic"
SYSTEM_EXECUTOR_PROMPT: str = "system_executor"
SYSTEM_PLANNER_PROMPT: str = "system_planner"
SYSTEM_EXTRACTOR_PROMPT: str = "system_extractor"
EMERGENCY_TRUNCATION_PROMPT: str = "emergency_truncation"
BRIDGE_BLOCK_PROMPTS: Tuple[str, ...] = ("bridge_block1", "bridge_block2", "bridge_block3")
SYSTEM_ROLE_PROMPTS: Tuple[str, ...] = (
    "system_executor",
    "system_planner",
    "system_extractor",
)

#: The four extractor constraints stated by the paper (§4.4). The offline graph
#: builder injects these into ``system_extractor`` so the rule set is testable
#: independently of the prose template.
EXTRACTOR_RULES: Tuple[str, ...] = (
    "Every extracted record MUST name the target network entity it applies to "
    "(host, endpoint, service or parameter).",
    "Every extracted record MUST name the associated vulnerability (or the "
    "concrete weakness class) it exploits.",
    "Every extracted record MUST carry a MITRE ATT&CK technique id "
    "(e.g. T1190). Records without one are rejected.",
    "Error paths / dead ends belong in the separate `error_paths` collection, "
    "never in the success graph. Respond with strict JSON only.",
)

# Placeholder syntax: ``{{ name }}``.
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*\}\}")
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class PromptError(RuntimeError):
    """Raised when a prompt template cannot be parsed."""


class PromptNotFound(PromptError):
    """Raised when a prompt name has no corresponding file on disk."""

    def __init__(self, name: str, *, directory: Optional[Path] = None) -> None:
        self.name = name
        self.directory = directory
        super().__init__(
            f"prompt {name!r} was not found in {directory or PROMPTS_DIR}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": "PROMPT_NOT_FOUND",
            "name": self.name,
            "directory": str(self.directory or PROMPTS_DIR),
        }


@dataclass(frozen=True)
class PromptInfo:
    """Parsed prompt: front matter + body template."""

    name: str
    body: str
    version: str = "0"
    role: str = ""
    temperature: Optional[float] = None
    description: str = ""
    frozen: bool = False
    path: str = ""
    sha256: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def placeholders(self) -> Tuple[str, ...]:
        return extract_placeholders(self.body)

    def render(self, **values: Any) -> str:
        return render_template(self.body, values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "role": self.role,
            "temperature": self.temperature,
            "description": self.description,
            "frozen": self.frozen,
            "path": self.path,
            "sha256": self.sha256,
            "placeholders": list(self.placeholders),
            "chars": len(self.body),
        }


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

_CACHE: Dict[str, PromptInfo] = {}
_CACHE_MTIME: Dict[str, float] = {}
_LOCK = threading.RLock()


def clear_cache() -> None:
    """Drop the parsed-prompt cache (tests / hot reload)."""
    with _LOCK:
        _CACHE.clear()
        _CACHE_MTIME.clear()


def prompt_path(name: str) -> Path:
    """Absolute path of ``<PROMPTS_DIR>/<name>.md``."""
    return PROMPTS_DIR / f"{name}.md"


def _parse_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    raw = match.group(1)
    meta: Dict[str, Any] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        meta[key] = value
    body = text[match.end():]
    return meta, body


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load(name: str) -> PromptInfo:
    """Load (and cache) one prompt; raises :class:`PromptNotFound`."""
    path = prompt_path(name)
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:  # missing file
        raise PromptNotFound(name) from exc

    with _LOCK:
        cached = _CACHE.get(name)
        if cached is not None and _CACHE_MTIME.get(name) == mtime:
            return cached

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - unreadable file
        raise PromptError(f"prompt {name!r} could not be read: {exc}") from exc

    meta, body = _parse_front_matter(text)
    info = PromptInfo(
        name=str(meta.get("name") or name),
        body=body,
        version=str(meta.get("version") or "0"),
        role=str(meta.get("role") or ""),
        temperature=_coerce_float(meta.get("temperature")),
        description=str(meta.get("description") or ""),
        frozen=bool(name in FROZEN_PROMPTS or str(meta.get("frozen", "")).lower() in {"1", "true", "yes"}),
        path=str(path),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        meta=dict(meta),
    )
    with _LOCK:
        _CACHE[name] = info
        _CACHE_MTIME[name] = mtime
    return info


# ---------------------------------------------------------------------------
# public helpers
# ---------------------------------------------------------------------------


def available_prompts() -> Tuple[str, ...]:
    """Prompt names that actually exist on disk (sorted, canonical first)."""
    found: List[str] = []
    for name in PROMPT_NAMES:
        if prompt_path(name).is_file():
            found.append(name)
    if PROMPTS_DIR.is_dir():
        for path in sorted(PROMPTS_DIR.glob("*.md")):
            stem = path.stem
            if stem not in found:
                found.append(stem)
    return tuple(found)


def load_raw_prompt(name: str) -> str:
    """Return the raw template body (front matter stripped)."""
    return _load(name).body


def load_prompt(name: str, **values: Any) -> str:
    """Render one prompt template.

    Parameters
    ----------
    name:
        Template stem, e.g. ``"stage5_heuristic"``.
    values:
        Placeholder substitutions. Missing values render as ``""`` (never raise,
        so a prompt change cannot break the retrieval pipeline).
    """
    info = _load(name)
    return render_template(info.body, values)


def prompt_metadata(name: str) -> Dict[str, Any]:
    """Front matter + hash for tracing/reporting (``{}`` when absent)."""
    try:
        return _load(name).to_dict()
    except PromptError as exc:
        logger.debug("prompt.metadata_unavailable", extra={"prompt": name, "error": str(exc)})
        return {}


def prompt_hash(name: str) -> str:
    """SHA-256 of the raw template text (version pinning in traces)."""
    try:
        return _load(name).sha256
    except PromptError:
        return ""


def render_template(template: str, values: Optional[Mapping[str, Any]] = None) -> str:
    """Substitute ``{{placeholder}}`` markers.

    Values are stringified; ``None`` becomes ``""``; unknown placeholders are
    left verbatim (so a partial render is visible in a trace rather than
    silently dropped). No regex substitution is applied to *values*, therefore
    tool output cannot smuggle another placeholder into the prompt through this
    helper.
    """
    mapping = values or {}

    def _replace(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in mapping:
            return match.group(0)
        value = mapping[key]
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value)
        if isinstance(value, Mapping):
            return ", ".join(f"{k}={v}" for k, v in value.items())
        return str(value)

    return _PLACEHOLDER_RE.sub(_replace, template or "")


def extract_placeholders(template: str) -> Tuple[str, ...]:
    """Unique placeholder names in first-appearance order."""
    seen: Dict[str, None] = {}
    for match in _PLACEHOLDER_RE.finditer(template or ""):
        seen.setdefault(match.group(1), None)
    return tuple(seen)


def registry_snapshot() -> Dict[str, Any]:
    """Metadata for every prompt that exists (used by ``/settings`` + reports)."""
    snapshot: Dict[str, Any] = {}
    for name in available_prompts():
        info = prompt_metadata(name)
        if info:
            snapshot[name] = info
    return snapshot


# ---------------------------------------------------------------------------
# bridge block helpers (Algorithm 3)
# ---------------------------------------------------------------------------


def render_bridge_blocks(**values: Any) -> Dict[str, str]:
    """Render B1/B2/B3 in order, tolerant of a missing template.

    ``bridge.py`` builds its blocks programmatically (the paper describes their
    *purpose*, not verbatim text); this helper exists so the versioned templates
    under ``prompts/bridge_block*.md`` stay exercised and auditable.
    """
    blocks: Dict[str, str] = {}
    for index, name in enumerate(BRIDGE_BLOCK_PROMPTS, start=1):
        try:
            blocks[f"B{index}"] = load_prompt(name, **values)
        except PromptError:
            blocks[f"B{index}"] = ""
    return blocks


def render_extractor_prompt(**values: Any) -> str:
    """Render ``system_extractor`` with the four paper rules appended."""
    rules = "\n".join(f"{i}. {rule}" for i, rule in enumerate(EXTRACTOR_RULES, start=1))
    payload = {"rules": rules, "extractor_rules": rules}
    payload.update(values)
    try:
        return load_prompt(SYSTEM_EXTRACTOR_PROMPT, **payload)
    except PromptError:  # pragma: no cover - template always shipped
        return rules


def render_stage5_prompt(**values: Any) -> str:
    """Render the S5 heuristic-reasoning prompt, always carrying the JSON rules."""
    payload = {
        "output_schema": values.pop(
            "output_schema",
            '{"tactic": "...", "objective": "...", "target_entity": "...", '
            '"vulnerability_type": "...", "mitre_attack_id": "T....", '
            '"confidence": 0.0}',
        )
    }
    payload.update(values)
    try:
        return load_prompt(STAGE5_LLM_PROMPT, **payload)
    except PromptError:  # pragma: no cover
        return "Respond with strict JSON only."


# Backwards/forwards compatible alias used by early S5 drafts.
stage5_prompt = render_stage5_prompt
bridge_block_prompts = render_bridge_blocks


def iter_prompts(names: Optional[Iterable[str]] = None) -> List[PromptInfo]:
    """Load a batch of prompts, skipping missing ones."""
    out: List[PromptInfo] = []
    for name in names or available_prompts():
        try:
            out.append(_load(name))
        except PromptError:
            continue
    return out
