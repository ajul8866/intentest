"""Settings and LLM-capability REST surface (paper §4.4 / plan C11, C13).

Endpoints
---------
``GET  /settings``                 effective settings (secrets redacted) + capability matrix
``GET  /settings/llm``             per-role model ids, strategy resolution order, capability matrix
``POST /settings/llm/probe``       re-run the mandatory startup capability probe
``GET  /settings/strategies``      the declared strategy precedence ladder
``GET  /settings/embedding``       embedding provider identity + fallback chain status
``GET  /settings/sandbox``         sandbox mode / Docker availability
``GET  /settings/features``        behavioural flags (ablation switch, verify escalation, ...)

Design notes
------------
* Routes stay thin: every value comes from :mod:`app.config` (pydantic-settings) through the
  dependency module :mod:`app.api.deps`; nothing here reads ``os.environ`` directly so the
  probe/settings view can never diverge from the runtime configuration.
* Secrets are **never** returned. :func:`redact_settings` walks the settings object and
  replaces anything whose key looks secret (``*_key``, ``*_token``, ``*_secret``,
  ``*_password``, ``database_url`` credentials) with ``"***"`` — it is an allow-list shaped
  serializer, not a regex scrub of a serialized blob.
* Capability probing is *live*: ``POST /settings/llm/probe`` reuses the same
  ``LLMClient.probe_capabilities()`` coroutine used at startup, resets the cached client so a
  changed endpoint/key takes effect, and persists the resulting matrix via
  ``app.core.llm.capabilities.persist_capabilities`` when available.
* If the LLM is unreachable the endpoint still answers with HTTP 200 and
  ``{"ok": false, "error": ...}`` so the Settings UI can render the failure instead of a 503
  page; the *startup* gate (fail if ``chat`` is unsupported) is enforced by the application
  lifespan, not here.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Mapping, Optional, Sequence

from fastapi import APIRouter, Depends, Query

from app.api import deps
from app.api.errors import AppError, ErrorCode, ValidationAppError

__all__ = [
    "router",
    "redact_settings",
    "settings_to_dict",
    "capabilities_to_dict",
    "STRATEGY_ORDER",
    "SECRET_MARKERS",
]

log = logging.getLogger(__name__)
router = APIRouter(prefix="/settings", tags=["settings"])

_MISS = object()

#: Strategy resolution ladder (paper §4.4 / plan C11). The first capability that is
#: available wins; ``PROMPT_JSON`` is the always-available last resort (fenced JSON plus a
#: repair round-trip), never an unparsed pass-through.
STRATEGY_ORDER: tuple[str, ...] = ("TOOL_CALLING", "JSON_SCHEMA", "JSON_OBJECT", "PROMPT_JSON")

#: Substrings that mark a settings key as secret. Anything matching is redacted.
SECRET_MARKERS: tuple[str, ...] = ("api_key", "apikey", "token", "secret", "password", "passwd")

#: Keys that are redacted verbatim (they embed credentials inside a URL).
_REDACT_ENDPOINT_KEYS: tuple[str, ...] = ("database_url", "DATABASE_URL", "db_url", "redis_url")

#: Capability names surfaced by the probe (plan C11).
CAPABILITY_FIELDS: tuple[str, ...] = (
    "models",
    "chat",
    "tool_calling",
    "json_schema",
    "json_object",
    "embeddings",
    "streaming",
    "max_context",
)

#: Role names used for per-role model configuration.
ROLES: tuple[str, ...] = ("planning", "execution", "verification", "reporting")


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Tolerant attribute/mapping access (ORM row, dataclass, pydantic model, dict)."""
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        value = getattr(obj, name, _MISS)
        if value is not _MISS:
            return value
    return default


def _iso(value: Any) -> Any:
    """Best-effort JSON-safe scalar."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_iso(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_iso(item) for item in value)
    if isinstance(value, Mapping):
        return {str(k): _iso(v) for k, v in value.items()}
    enum_value = getattr(value, "value", _MISS)
    if enum_value is not _MISS and isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    text = getattr(value, "isoformat", None)
    if callable(text):
        try:
            return text()
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in SECRET_MARKERS)


def _redact_url(url: Any) -> Any:
    """``postgresql+asyncpg://user:pass@host/db`` -> ``postgresql+asyncpg://user:***@host/db``."""
    if not isinstance(url, str) or "@" not in url:
        return url
    try:
        scheme, rest = url.split("://", 1)
    except ValueError:
        return "***"
    credentials, host = rest.rsplit("@", 1)
    if ":" in credentials:
        user, _sep, _pw = credentials.partition(":")
        return f"{scheme}://{user}:***@{host}"
    return f"{scheme}://***@{host}"


def _settings_mapping(settings: Any) -> dict[str, Any]:
    """Materialise a settings object into a plain dict (attribute or model based)."""
    if settings is None:
        return {}
    dump = getattr(settings, "model_dump", None)
    if callable(dump):
        try:
            return dict(dump())
        except Exception:  # pragma: no cover - defensive
            pass
    as_dict = getattr(settings, "dict", None)
    if callable(as_dict):
        try:
            return dict(as_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(settings, Mapping):
        return dict(settings)
    values: dict[str, Any] = {}
    for name in dir(settings):
        if name.startswith("_"):
            continue
        try:
            value = getattr(settings, name)
        except Exception:  # pragma: no cover - defensive
            continue
        if callable(value):
            continue
        values[name] = value
    return values


def _redact_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in values.items():
        name = str(key)
        if _is_secret_key(name):
            redacted[name] = "***" if value not in (None, "") else ""
            continue
        if name in _REDACT_ENDPOINT_KEYS:
            redacted[name] = _redact_url(value)
            continue
        redacted[name] = _iso(value)
    return redacted


# --------------------------------------------------------------------------------------
# public serializers (importable by the frontend typed-client generator / tests)
# --------------------------------------------------------------------------------------
def redact_settings(settings: Any) -> dict[str, Any]:
    """Return a JSON-safe settings dict with every secret replaced by ``"***"``.

    Accepts a pydantic model, a mapping, or any object with public attributes. Never raises.
    """
    return _redact_mapping(_settings_mapping(settings))


def settings_to_dict(settings: Any) -> dict[str, Any]:
    """Alias of :func:`redact_settings` kept for call-site symmetry with schemas."""
    return redact_settings(settings)


def _capability_value(caps: Any, name: str) -> Any:
    value = _attr(caps, name, default=_MISS)
    if value is _MISS:
        extra = _attr(caps, "extra", "details", "raw", default=None)
        if isinstance(extra, Mapping) and name in extra:
            return extra[name]
        return None
    return _iso(value)


def capabilities_to_dict(caps: Any) -> dict[str, Any]:
    """Normalise a capability matrix (object or mapping) into a flat JSON dict."""
    if caps is None:
        return {"available": False, "reason": "no capability probe has been run"}
    flat: dict[str, Any] = {}
    for name in CAPABILITY_FIELDS:
        flat[name] = _capability_value(caps, name)
    # Preserve any vendor-specific extras without letting them shadow canonical fields.
    if isinstance(caps, Mapping):
        for key, value in caps.items():
            if key not in flat:
                flat[str(key)] = _iso(value)
    else:
        extra = _attr(caps, "extra", "details", "raw", default=None)
        if isinstance(extra, Mapping):
            for key, value in extra.items():
                flat.setdefault(str(key), _iso(value))
    flat["available"] = bool(flat.get("chat"))
    if flat.get("probed_at") is None and _attr(caps, "probed_at", default=None) is not None:
        flat["probed_at"] = _iso(_attr(caps, "probed_at"))
    return flat


def resolve_strategy(caps: Any) -> str:
    """Pick the highest-precedence usable strategy for this capability matrix."""
    if caps is None:
        return "PROMPT_JSON"
    flags = {
        "TOOL_CALLING": bool(_capability_value(caps, "tool_calling")),
        "JSON_SCHEMA": bool(_capability_value(caps, "json_schema")),
        "JSON_OBJECT": bool(_capability_value(caps, "json_object")),
        "PROMPT_JSON": True,
    }
    for strategy in STRATEGY_ORDER:
        if flags.get(strategy):
            return strategy
    return "PROMPT_JSON"


# --------------------------------------------------------------------------------------
# optional module bridges (all lazily imported so the route module stays import-light)
# --------------------------------------------------------------------------------------
def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover - optional dependency
        log.debug("settings.module_unavailable", extra={"module": name})
        return None


def _function(module: Any, names: Sequence[str]) -> Any:
    if module is None:
        return None
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    for name in names:
        cls = getattr(module, name, None)
        if inspect.isclass(cls):
            return cls
    return None


async def _probe_llm(settings: Any) -> tuple[dict[str, Any], Any]:
    """Run (or re-use) the LLM capability probe.

    Returns ``(capability_dict, client)``. Never raises: failures are reported inside the
    dict so the Settings page can display them.
    """
    client: Any = None
    probe_error: Optional[str] = None

    # Prefer a freshly-built client so a changed base_url/key is honoured by the re-probe.
    deps_reset = getattr(deps, "reset_llm_client", None)
    if callable(deps_reset):
        try:
            deps_reset()
        except Exception:  # pragma: no cover - defensive
            pass

    try:
        client = deps.get_llm_client(settings)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        probe_error = f"{type(exc).__name__}: {exc}"
        return (
            {
                "available": False,
                "strategy": "PROMPT_JSON",
                "error": probe_error,
                "capabilities": {name: None for name in CAPABILITY_FIELDS},
            },
            None,
        )

    caps: Any = None
    probe = getattr(client, "probe_capabilities", None) or getattr(client, "probe", None)
    if callable(probe):
        try:
            caps = await _maybe_await(probe())
        except Exception as exc:  # noqa: BLE001
            probe_error = f"{type(exc).__name__}: {exc}"
        persist = _function(
            _module("app.core.llm.capabilities"),
            ("persist_capabilities", "save_capabilities", "store_capabilities"),
        )
        if caps is not None and callable(persist):
            try:
                await _maybe_await(persist(caps, settings))
            except Exception:  # pragma: no cover - persistence is best effort
                log.warning("settings.capabilities_persist_failed", exc_info=True)
    else:
        caps = getattr(client, "capabilities", None)
        if caps is None:
            probe_error = "LLM client does not expose probe_capabilities()"

    if caps is None and probe_error is None:
        probe_error = "capability probe returned no result"

    matrix = capabilities_to_dict(caps)
    payload = {
        "available": bool(matrix.get("chat")) and probe_error is None,
        "strategy": resolve_strategy(caps),
        "capabilities": matrix,
        "model": _attr(client, "model", "model_id", default=None),
        "base_url": _iso(_attr(client, "base_url", default=None)),
    }
    if probe_error:
        payload["error"] = probe_error
    return payload, client


async def _embedding_status() -> dict[str, Any]:
    provider: Any = None
    error: Optional[str] = None
    try:
        provider = deps.get_embedding_provider(deps.get_settings())
        if inspect.isawaitable(provider):
            provider = await provider
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    backend = _attr(provider, "backend", "provider", "name", default=None)
    dim = _attr(provider, "dim", "dimension", "dimensions", default=None)
    model = _attr(provider, "model", "model_id", default=None)
    payload: dict[str, Any] = {
        "ok": provider is not None and error is None,
        "backend": _iso(backend) if backend is not None else None,
        "model": _iso(model) if model is not None else None,
        "dim": dim,
        "fallback_chain": ["api", "sentence-transformers", "hash"],
        "degraded": (str(backend).lower() == "hash") if backend is not None else None,
    }
    if error:
        payload["error"] = error
    return payload


async def _sandbox_status() -> dict[str, Any]:
    getter = getattr(deps, "get_sandbox_status", None)
    if not callable(getter):
        return {"ok": False, "mode": "unknown", "detail": "sandbox status provider not available"}
    try:
        status = getter()
        if inspect.isawaitable(status):
            status = await status
    except AppError as exc:
        return {"ok": False, "mode": "unavailable", "detail": exc.message}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "mode": "unavailable", "detail": f"{type(exc).__name__}: {exc}"}
    if isinstance(status, Mapping):
        return {str(k): _iso(v) for k, v in status.items()}
    return {"ok": bool(_attr(status, "ok", default=False)), "detail": _iso(_attr(status, "detail"))}


def _role_models(settings: Any) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for role in ROLES:
        value = _attr(
            settings,
            f"llm_model_{role}",
            f"LLM_MODEL_{role.upper()}",
            default=None,
        )
        models[role] = _iso(value)
    fallback = _attr(settings, "llm_model", "LLM_MODEL", default=None)
    models["default"] = _iso(fallback)
    return models


def _features(settings: Any) -> dict[str, Any]:
    def flag(*names: str, default: Any = None) -> Any:
        return _iso(_attr(settings, *names, default=default))

    return {
        "intent_guidance_enabled": flag(
            "intent_guidance_enabled", "INTENT_GUIDANCE_ENABLED", default=True
        ),
        "verify_escalation_enabled": flag(
            "verify_escalation_enabled", "VERIFY_ESCALATION_ENABLED", default=False
        ),
        "count_reason_toward_budget": flag(
            "count_reason_toward_budget", "COUNT_REASON_TOWARD_BUDGET", default=True
        ),
        "sandbox_mode": flag("sandbox_mode", "SANDBOX_MODE", default="docker"),
    }


def _timings(settings: Any) -> dict[str, Any]:
    return {
        "max_rounds": _attr(settings, "max_rounds", "MAX_ROUNDS", default=40),
        "main_timeout_s": _attr(settings, "main_timeout_s", "MAIN_TIMEOUT_S", default=180),
        "degradation_timeout_s": _attr(
            settings, "degradation_timeout_s", "DEGRADATION_TIMEOUT_S", default=90
        ),
        "intent_lease_s": _attr(settings, "intent_lease_s", "INTENT_LEASE_S", default=360),
        "intent_heartbeat_s": _attr(
            settings, "intent_heartbeat_s", "INTENT_HEARTBEAT_S", default=20
        ),
        "project_lease_s": _attr(settings, "project_lease_s", "PROJECT_LEASE_S", default=600),
        "tick_seconds": _attr(settings, "tick_seconds", "TICK_SECONDS", default=2),
        "stall_ticks": _attr(settings, "stall_ticks", "STALL_TICKS", default=15),
    }


def _retrieval_params(settings: Any) -> dict[str, Any]:
    return {
        "bridge_theta": _attr(settings, "bridge_theta", "BRIDGE_THETA", default=0.3),
        "subgraph_top_k": _attr(settings, "subgraph_top_k", "SUBGRAPH_TOP_K", default=10),
        "subgraph_max_query_nodes": _attr(
            settings, "subgraph_max_query_nodes", "SUBGRAPH_MAX_QUERY_NODES", default=12
        ),
        "subgraph_max_pattern_nodes": _attr(
            settings, "subgraph_max_pattern_nodes", "SUBGRAPH_MAX_PATTERN_NODES", default=60
        ),
        "subgraph_timeout_s": _attr(
            settings, "subgraph_timeout_s", "SUBGRAPH_TIMEOUT_S", default=5
        ),
        "jaccard_threshold": _attr(
            settings, "jaccard_threshold", "JACCARD_THRESHOLD", default=0.5
        ),
        "crosslink_cosine": _attr(settings, "crosslink_cosine", "CROSSLINK_COSINE", default=0.75),
        "goal_align_threshold": _attr(
            settings, "goal_align_threshold", "GOAL_ALIGN_THRESHOLD", default=0.55
        ),
        "confidence_threshold": _attr(
            settings, "confidence_threshold", "CONFIDENCE_THRESHOLD", default=0.6
        ),
        "align_alpha": _attr(settings, "align_alpha", "ALIGN_ALPHA", default=0.5),
        "align_beta": _attr(settings, "align_beta", "ALIGN_BETA", default=0.3),
        "align_gamma": _attr(settings, "align_gamma", "ALIGN_GAMMA", default=0.2),
        "embedding_dim": _attr(settings, "embedding_dim", "EMBEDDING_DIM", default=2560),
        "embedding_model": _iso(
            _attr(settings, "embedding_model", "EMBEDDING_MODEL", default="Qwen3-Embedding-4B")
        ),
    }


# --------------------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------------------
@router.get("", summary="Effective settings (secrets redacted) + capability matrix")
async def get_settings_view(
    include: Optional[str] = Query(
        default=None,
        description="Comma separated sections: capabilities,features,timings,retrieval,embedding,sandbox",
    ),
) -> dict[str, Any]:
    settings = deps.get_settings()
    redacted = redact_settings(settings)

    sections = {
        "capabilities": None,
        "features": _features(settings),
        "timings": _timings(settings),
        "retrieval": _retrieval_params(settings),
        "embedding": None,
        "sandbox": None,
    }

    wanted: Optional[set[str]] = None
    if include:
        wanted = {part.strip().lower() for part in include.split(",") if part.strip()}

    if wanted is None or "capabilities" in wanted:
        try:
            caps = deps.get_llm_capabilities()
            if inspect.isawaitable(caps):
                caps = await caps
        except Exception as exc:  # noqa: BLE001 - degrade gracefully in the UI
            caps = None
            sections["capabilities"] = {
                "available": False,
                "strategy": "PROMPT_JSON",
                "error": f"{type(exc).__name__}: {exc}",
            }
        if sections["capabilities"] is None:
            sections["capabilities"] = {
                "available": bool(_capability_value(caps, "chat")),
                "strategy": resolve_strategy(caps),
                "capabilities": capabilities_to_dict(caps),
            }
    if wanted is None or "embedding" in wanted:
        sections["embedding"] = await _embedding_status()
    if wanted is None or "sandbox" in wanted:
        sections["sandbox"] = await _sandbox_status()

    if wanted is not None:
        for name in list(sections):
            if name not in wanted:
                sections.pop(name)

    return {
        "settings": redacted,
        "llm": {
            "base_url": _iso(_attr(settings, "llm_base_url", "LLM_BASE_URL", default=None)),
            "models": _role_models(settings),
            "strategy_order": list(STRATEGY_ORDER),
        },
        "sections": sections,
        "paper_constants": {
            "max_rounds": 40,
            "main_timeout_s": 180,
            "degradation_timeout_s": 90,
            "bridge_theta": 0.3,
            "embedding_dim": 2560,
            "note": "paper-stated values; every other threshold is a documented default",
        },
    }


@router.get("/llm", summary="Per-role model ids, strategy ladder and capability matrix")
async def get_llm_settings() -> dict[str, Any]:
    settings = deps.get_settings()
    caps_error: Optional[str] = None
    caps: Any = None
    try:
        caps = deps.get_llm_capabilities()
        if inspect.isawaitable(caps):
            caps = await caps
    except Exception as exc:  # noqa: BLE001
        caps_error = f"{type(exc).__name__}: {exc}"

    client: Any = None
    try:
        client = deps.get_llm_client(settings)
    except Exception:  # noqa: BLE001 - optional
        client = None

    payload: dict[str, Any] = {
        "base_url": _iso(_attr(settings, "llm_base_url", "LLM_BASE_URL", default=None)),
        "roles": _role_models(settings),
        "strategy_order": list(STRATEGY_ORDER),
        "resolved_strategy": resolve_strategy(caps),
        "capabilities": capabilities_to_dict(caps),
        "available": bool(_capability_value(caps, "chat")) and caps_error is None,
        "models_report": _iso(_attr(client, "models", "model_list", default=None)),
        "timeout_s": _attr(settings, "llm_timeout_s", "LLM_TIMEOUT_S", default=None),
        "max_retries": _attr(settings, "llm_max_retries", "LLM_MAX_RETRIES", default=None),
        "one_sample_per_decision": True,
        "extractor": {
            "temperature": _attr(
                settings, "extractor_temperature", "EXTRACTOR_TEMPERATURE", default=0.0
            ),
            "max_retries": _attr(
                settings, "extractor_max_retries", "EXTRACTOR_MAX_RETRIES", default=3
            ),
            "strict_json": True,
        },
        "exploration_temperature": _attr(
            settings, "exploration_temperature", "EXPLORATION_TEMPERATURE", default=0.2
        ),
    }
    if caps_error:
        payload["error"] = caps_error
    return payload


@router.post("/llm/probe", summary="Re-run the capability probe and persist the matrix")
async def probe_llm(
    persist: bool = Query(default=True, description="Persist the matrix for the next startup."),
) -> dict[str, Any]:
    settings = deps.get_settings()
    payload, client = await _probe_llm(settings)
    if not persist:
        payload["persisted"] = False
    else:
        payload["persisted"] = client is not None

    if not payload.get("available"):
        log.warning("settings.llm_probe_failed", extra={"error": payload.get("error")})
    else:
        log.info(
            "settings.llm_probe_ok",
            extra={
                "strategy": payload.get("strategy"),
                "model": payload.get("model"),
                "max_context": (payload.get("capabilities") or {}).get("max_context"),
            },
        )
    return payload


@router.get("/strategies", summary="Declared strategy resolution ladder")
async def get_strategies() -> dict[str, Any]:
    settings = deps.get_settings()
    caps: Any = None
    try:
        caps = deps.get_llm_capabilities()
        if inspect.isawaitable(caps):
            caps = await caps
    except Exception:  # noqa: BLE001 - capability detail is optional here
        caps = None

    ladder = []
    matrix = capabilities_to_dict(caps)
    for index, strategy in enumerate(STRATEGY_ORDER):
        key = strategy.lower()
        available = True if strategy == "PROMPT_JSON" else bool(matrix.get(key))
        ladder.append(
            {
                "order": index,
                "strategy": strategy,
                "available": available,
                "selected": strategy == resolve_strategy(caps),
                "description": {
                    "TOOL_CALLING": "native OpenAI tool_calls; validated against the tool allow-list",
                    "JSON_SCHEMA": "response_format json_schema with a per-role Pydantic model",
                    "JSON_OBJECT": "response_format json_object; parser enforces the schema",
                    "PROMPT_JSON": "fenced JSON in prompt + one repair round-trip on failure",
                }.get(strategy, ""),
            }
        )
    return {
        "order": list(STRATEGY_ORDER),
        "resolved": resolve_strategy(caps),
        "ladder": ladder,
        "embedding_model": _iso(
            _attr(settings, "embedding_model", "EMBEDDING_MODEL", default="Qwen3-Embedding-4B")
        ),
        "failure_mode": "LLMMalformedOutput triggers the two-phase degradation path",
    }


@router.get("/embedding", summary="Embedding provider identity and fallback chain")
async def get_embedding_settings() -> dict[str, Any]:
    settings = deps.get_settings()
    status = await _embedding_status()
    return {
        "model": _iso(
            _attr(settings, "embedding_model", "EMBEDDING_MODEL", default="Qwen3-Embedding-4B")
        ),
        "dim": _attr(settings, "embedding_dim", "EMBEDDING_DIM", default=2560),
        "base_url": _iso(
            _attr(settings, "embedding_base_url", "EMBEDDING_BASE_URL", default=None)
        ),
        "status": status,
        "fallback_chain": ["api", "sentence-transformers", "hash"],
        "note": "the backend that produced each vector is recorded on the retrieval trace",
    }


@router.get("/sandbox", summary="Sandbox mode and Docker availability")
async def get_sandbox_settings() -> dict[str, Any]:
    settings = deps.get_settings()
    status = await _sandbox_status()
    return {
        "mode": _iso(_attr(settings, "sandbox_mode", "SANDBOX_MODE", default="docker")),
        "image": _iso(_attr(settings, "sandbox_image", "SANDBOX_IMAGE", default=None)),
        "status": status,
        "isolation": {
            "network": "per-project bridge network",
            "rootfs": "read-only",
            "tmpfs": {"/tmp": "256m", "/work": "512m"},
            "cap_drop": ["ALL"],
            "no_new_privileges": True,
            "labels": ["intentest.project", "intentest.intent_id", "intentest.task_run_id"],
            "teardown": "SIGTERM -> 5s -> SIGKILL in finally; orphan reaper by label",
        },
    }


@router.get("/features", summary="Behavioural flags (ablation switch and documented extensions)")
async def get_feature_flags() -> dict[str, Any]:
    settings = deps.get_settings()
    return {
        "features": _features(settings),
        "descriptions": {
            "intent_guidance_enabled": (
                "false keeps the DAG, FSM, scheduler and degradation intact but disables "
                "retrieval (Algorithm 2) and the intent bridge (Algorithm 3) — the noIntent ablation"
            ),
            "verify_escalation_enabled": (
                "exposes VERIFY as an optional escalation task; verification otherwise runs inline "
                "in the worker (paper extension, see LIMITATIONS.md)"
            ),
            "count_reason_toward_budget": (
                "whether REASON agent turns count toward MAX_ROUNDS (paper does not state)"
            ),
            "sandbox_mode": "docker (default) | local (tests only, loudly flagged)",
        },
    }


@router.get("/{section}", summary="Single settings section by name")
async def get_settings_section(section: str) -> dict[str, Any]:
    """Convenience accessor so the UI can refresh one card without the whole payload."""
    name = (section or "").strip().lower()
    if not name:
        raise ValidationAppError("Section name is required", details={"section": section})

    if name in {"llm", "capabilities", "strategy", "strategies"}:
        return await get_llm_settings()
    if name in {"embedding", "embeddings"}:
        return await get_embedding_settings()
    if name in {"sandbox", "docker"}:
        return await get_sandbox_settings()
    if name in {"features", "flags"}:
        return await get_feature_flags()
    if name in {"timings", "timeouts", "leases"}:
        return {"timings": _timings(deps.get_settings())}
    if name in {"retrieval", "thresholds"}:
        return {"retrieval": _retrieval_params(deps.get_settings())}
    if name in {"all", "effective"}:
        return await get_settings_view(include=None)

    raise AppError(
        f"Unknown settings section {section!r}",
        code=ErrorCode.NOT_FOUND,
        details={
            "section": section,
            "known": [
                "all",
                "llm",
                "strategies",
                "embedding",
                "sandbox",
                "features",
                "timings",
                "retrieval",
            ],
        },
    )
