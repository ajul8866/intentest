"""Runtime configuration for Intentest.

Every knob that the paper states explicitly (40-round budget, 180 s / 90 s
two-phase degradation timeouts, ``theta = 0.3`` bridge activation threshold,
2560-d embeddings, extractor temperature 0 / <=3 retries) and every knob the
paper leaves unpublished (lease TTLs, tick interval, Top-k, Jaccard /
alignment / confidence thresholds, ``alpha``/``beta``/``gamma``,
``STALL_TICKS``) is declared here with an environment variable and the
documented default from ``LIMITATIONS.md`` section 2.

The module is import-light: it only pulls ``pydantic`` (always required by the
API layer) and *optionally* ``pydantic-settings``.  If ``pydantic-settings`` is
not installed we fall back to a plain ``os.environ`` loader so that alembic,
the evaluation harness and unit tests can import settings without the full API
dependency set.

Access patterns used across the codebase::

    from app.config import settings          # process-wide singleton
    from app.config import get_settings      # callable accessor (tests)
    settings.database_url                    # normalised async DSN
    settings.llm_model_for("execution")      # per-role model id
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, field_validator

try:  # pragma: no cover - exercised implicitly by the import path taken
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except Exception:  # pragma: no cover - optional dependency
    BaseSettings = BaseModel  # type: ignore[assignment,misc]
    SettingsConfigDict = None  # type: ignore[assignment]
    _HAS_PYDANTIC_SETTINGS = False


__all__ = [
    "Settings",
    "SettingsBase",
    "get_settings",
    "reload_settings",
    "settings",
    "ROLE_NAMES",
    "DEFAULT_API_PREFIX",
    "DEFAULT_WS_PATH",
    "DEFAULT_EVENT_CHANNEL",
    "PAPER_CONSTANTS",
    "normalise_async_url",
    "normalise_http_url",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLE_NAMES: tuple[str, ...] = ("planning", "execution", "verification", "reporting")

DEFAULT_API_PREFIX = "/api/v1"
DEFAULT_WS_PATH = "/ws"
DEFAULT_EVENT_CHANNEL = "intentest_events"

#: Values the paper states verbatim (``ALGORITHMS.md`` section 13).  Exposed so
#: the Settings UI can label them as paper-stated rather than chosen defaults.
PAPER_CONSTANTS: Dict[str, Any] = {
    "max_rounds": 40,
    "main_timeout_s": 180,
    "degradation_timeout_s": 90,
    "bridge_theta": 0.3,
    "embedding_dim": 2560,
    "extractor_temperature": 0.0,
    "extractor_max_retries": 3,
}


def normalise_async_url(url: Optional[str]) -> str:
    """Return a SQLAlchemy async DSN.

    ``postgres://`` / ``postgresql://`` / ``postgresql+psycopg://`` are rewritten
    to ``postgresql+asyncpg://`` so the same ``DATABASE_URL`` works for alembic,
    psql and the async engine.
    """

    if not url:
        return ""
    value = str(url).strip()
    for prefix in ("postgresql+psycopg2://", "postgresql+psycopg://", "postgres://"):
        if value.startswith(prefix):
            value = "postgresql+asyncpg://" + value[len(prefix) :]
            break
    if value.startswith("postgresql://"):
        value = "postgresql+asyncpg://" + value[len("postgresql://") :]
    return value


def normalise_http_url(url: Optional[str]) -> str:
    """Strip trailing slashes from an HTTP base URL (LLM / embedding endpoint)."""

    if not url:
        return ""
    return str(url).strip().rstrip("/")


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class SettingsBase(BaseModel):
    """Typed view over the environment.

    ``alias`` on every field is the upper-case environment variable; when
    ``pydantic-settings`` is available it performs the env lookup, otherwise
    :meth:`SettingsBase.from_env` does it manually.
    """

    model_config = {"extra": "ignore", "populate_by_name": True}

    # -- identity / deployment --------------------------------------------
    service_name: str = Field(default="backend", alias="INTENTEST_SERVICE_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")

    api_prefix: str = Field(default=DEFAULT_API_PREFIX, alias="API_PREFIX")
    ws_path: str = Field(default=DEFAULT_WS_PATH, alias="WS_PATH")
    event_channel: str = Field(default=DEFAULT_EVENT_CHANNEL, alias="EVENT_CHANNEL")
    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")

    # -- persistence -------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://intentest:intentest@localhost:5432/intentest",
        alias="DATABASE_URL",
    )
    db_pool_size: int = Field(default=10, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, alias="DB_MAX_OVERFLOW")
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # -- LLM plane ---------------------------------------------------------
    llm_base_url: str = Field(default="https://api.deepseek.com/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model_planning: str = Field(default="deepseek-chat", alias="LLM_MODEL_PLANNING")
    llm_model_execution: str = Field(default="deepseek-chat", alias="LLM_MODEL_EXECUTION")
    llm_model_verification: str = Field(
        default="deepseek-chat", alias="LLM_MODEL_VERIFICATION"
    )
    llm_model_reporting: str = Field(default="deepseek-chat", alias="LLM_MODEL_REPORTING")
    llm_timeout_s: float = Field(default=120.0, alias="LLM_TIMEOUT_S")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    llm_probe_on_startup: bool = Field(default=True, alias="LLM_PROBE_ON_STARTUP")
    llm_probe_timeout_s: float = Field(default=20.0, alias="LLM_PROBE_TIMEOUT_S")
    extractor_temperature: float = Field(default=0.0, alias="EXTRACTOR_TEMPERATURE")
    extractor_max_retries: int = Field(default=3, alias="EXTRACTOR_MAX_RETRIES")
    exploration_temperature: float = Field(default=0.2, alias="EXPLORATION_TEMPERATURE")
    verification_temperature: float = Field(
        default=0.0, alias="VERIFICATION_TEMPERATURE"
    )
    llm_capabilities_path: str = Field(
        default="/tmp/intentest/llm_capabilities.json",
        alias="LLM_CAPABILITIES_PATH",
    )

    # -- embedding plane ---------------------------------------------------
    embedding_base_url: str = Field(default="", alias="EMBEDDING_BASE_URL")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="Qwen3-Embedding-4B", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=2560, alias="EMBEDDING_DIM")
    embedding_timeout_s: float = Field(default=60.0, alias="EMBEDDING_TIMEOUT_S")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    embedding_cache_size: int = Field(default=4096, alias="EMBEDDING_CACHE_SIZE")
    embedding_allow_fallback: bool = Field(default=True, alias="EMBEDDING_ALLOW_FALLBACK")

    # -- scheduler (Algorithm 1) -------------------------------------------
    tick_seconds: float = Field(default=2.0, alias="TICK_SECONDS")
    project_lease_s: int = Field(default=600, alias="PROJECT_LEASE_S")
    intent_lease_s: int = Field(default=360, alias="INTENT_LEASE_S")
    intent_heartbeat_s: int = Field(default=20, alias="INTENT_HEARTBEAT_S")
    stall_ticks: int = Field(default=15, alias="STALL_TICKS")
    project_queue_ttl_s: int = Field(default=900, alias="PROJECT_QUEUE_TTL_S")
    scheduler_id: str = Field(default="", alias="SCHEDULER_ID")
    max_dispatches_per_tick: int = Field(default=4, alias="MAX_DISPATCHES_PER_TICK")
    backpressure_retry_s: int = Field(default=5, alias="BACKPRESSURE_RETRY_S")

    # -- execution / degradation ------------------------------------------
    max_rounds: int = Field(default=40, alias="MAX_ROUNDS")
    main_timeout_s: int = Field(default=180, alias="MAIN_TIMEOUT_S")
    degradation_timeout_s: int = Field(default=90, alias="DEGRADATION_TIMEOUT_S")
    tool_timeout_s: int = Field(default=120, alias="TOOL_TIMEOUT_S")
    max_tool_output_bytes: int = Field(default=262144, alias="MAX_TOOL_OUTPUT_BYTES")
    agent_max_actions_per_round: int = Field(default=1, alias="AGENT_MAX_ACTIONS_PER_ROUND")

    # -- retrieval (Algorithm 2) ------------------------------------------
    intent_guidance_enabled: bool = Field(default=True, alias="INTENT_GUIDANCE_ENABLED")
    subgraph_top_k: int = Field(default=10, alias="SUBGRAPH_TOP_K")
    jaccard_threshold: float = Field(default=0.5, alias="JACCARD_THRESHOLD")
    crosslink_cosine: float = Field(default=0.75, alias="CROSSLINK_COSINE")
    confidence_threshold: float = Field(default=0.6, alias="CONFIDENCE_THRESHOLD")
    align_alpha: float = Field(default=0.5, alias="ALIGN_ALPHA")
    align_beta: float = Field(default=0.3, alias="ALIGN_BETA")
    align_gamma: float = Field(default=0.2, alias="ALIGN_GAMMA")
    goal_align_threshold: float = Field(default=0.55, alias="GOAL_ALIGN_THRESHOLD")
    subgraph_max_query_nodes: int = Field(default=12, alias="SUBGRAPH_MAX_QUERY_NODES")
    subgraph_max_pattern_nodes: int = Field(
        default=60, alias="SUBGRAPH_MAX_PATTERN_NODES"
    )
    subgraph_timeout_s: float = Field(default=5.0, alias="SUBGRAPH_TIMEOUT_S")
    retrieval_trace_enabled: bool = Field(default=True, alias="RETRIEVAL_TRACE_ENABLED")
    offline_graph_path: str = Field(default="", alias="OFFLINE_GRAPH_PATH")

    # -- intent bridge (Algorithm 3) --------------------------------------
    bridge_theta: float = Field(default=0.3, alias="BRIDGE_THETA")
    bridge_recent_facts: int = Field(default=10, alias="BRIDGE_RECENT_FACTS")
    bridge_adjacent_patterns: int = Field(default=3, alias="BRIDGE_ADJACENT_PATTERNS")

    # -- behavioural extensions (documented in LIMITATIONS.md) ------------
    verify_escalation_enabled: bool = Field(
        default=False, alias="VERIFY_ESCALATION_ENABLED"
    )
    count_reason_toward_budget: bool = Field(
        default=True, alias="COUNT_REASON_TOWARD_BUDGET"
    )

    # -- worker plane ------------------------------------------------------
    worker_name: str = Field(default="", alias="WORKER_NAME")
    worker_tools: str = Field(
        default="http,curl,shell,nmap,sqlmap,ffuf,python", alias="WORKER_TOOLS"
    )
    worker_max_parallelism: int = Field(default=2, alias="WORKER_MAX_PARALLELISM")
    worker_priority_weight: float = Field(default=1.0, alias="WORKER_PRIORITY_WEIGHT")
    worker_poll_interval_s: float = Field(default=2.0, alias="WORKER_POLL_INTERVAL_S")
    worker_heartbeat_s: float = Field(default=10.0, alias="WORKER_HEARTBEAT_S")
    worker_max_consecutive_failures: int = Field(
        default=3, alias="WORKER_MAX_CONSECUTIVE_FAILURES"
    )
    worker_cooldown_s: int = Field(default=60, alias="WORKER_COOLDOWN_S")

    # -- sandbox -----------------------------------------------------------
    sandbox_mode: str = Field(default="docker", alias="SANDBOX_MODE")
    sandbox_image: str = Field(default="intentest/tools:latest", alias="SANDBOX_IMAGE")
    docker_host: str = Field(default="", alias="DOCKER_HOST")
    sandbox_network_prefix: str = Field(default="intentest-net", alias="SANDBOX_NETWORK_PREFIX")
    sandbox_tmp_size: str = Field(default="256m", alias="SANDBOX_TMP_SIZE")
    sandbox_work_size: str = Field(default="512m", alias="SANDBOX_WORK_SIZE")
    sandbox_memory_limit: str = Field(default="2g", alias="SANDBOX_MEMORY_LIMIT")
    sandbox_cpu_limit: float = Field(default=1.0, alias="SANDBOX_CPU_LIMIT")
    sandbox_pids_limit: int = Field(default=256, alias="SANDBOX_PIDS_LIMIT")
    sandbox_teardown_grace_s: int = Field(default=5, alias="SANDBOX_TEARDOWN_GRACE_S")
    sandbox_reap_interval_s: int = Field(default=30, alias="SANDBOX_REAP_INTERVAL_S")

    # -- evaluation harness -----------------------------------------------
    experiment_repeats: int = Field(default=1, alias="EXPERIMENT_REPEATS")
    experiment_seed: int = Field(default=42, alias="EXPERIMENT_SEED")
    labs_manifest_path: str = Field(
        default="eval/challenges/manifest.yaml", alias="LABS_MANIFEST_PATH"
    )
    results_dir: str = Field(default="eval/results", alias="RESULTS_DIR")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator(
        "api_prefix",
        "ws_path",
        "event_channel",
        "log_level",
        "sandbox_mode",
        mode="before",
    )
    @classmethod
    def _strip_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("log_level", mode="after")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return (value or "INFO").upper()

    @field_validator("sandbox_mode", mode="after")
    @classmethod
    def _check_sandbox_mode(cls, value: str) -> str:
        mode = (value or "docker").lower()
        if mode not in ("docker", "local"):
            return "docker"
        return mode

    @field_validator("api_prefix", mode="after")
    @classmethod
    def _normalise_prefix(cls, value: str) -> str:
        prefix = value or DEFAULT_API_PREFIX
        if not prefix.startswith("/"):
            prefix = "/" + prefix
        return prefix.rstrip("/") or DEFAULT_API_PREFIX

    @field_validator("ws_path", mode="after")
    @classmethod
    def _normalise_ws_path(cls, value: str) -> str:
        path = value or DEFAULT_WS_PATH
        if not path.startswith("/"):
            path = "/" + path
        return path

    @field_validator("llm_max_retries", "extractor_max_retries", mode="after")
    @classmethod
    def _at_least_one_retry(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator("max_rounds", mode="after")
    @classmethod
    def _positive_budget(cls, value: int) -> int:
        return max(1, int(value))

    # ------------------------------------------------------------------
    # Environment loading
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        """Build settings from ``os.environ`` (used when pydantic-settings is absent)."""

        source = dict(environ or os.environ)
        values: Dict[str, Any] = {}
        for name, field in cls.model_fields.items():  # type: ignore[attr-defined]
            alias = field.alias or name
            raw = source.get(alias)
            if raw is None:
                raw = source.get(alias.upper())
            if raw is None:
                continue
            values[name] = _coerce(raw, field.annotation)
        return cls(**values)

    # ------------------------------------------------------------------
    # Convenience accessors (used by client / retrieval / worker code)
    # ------------------------------------------------------------------
    @property
    def sqlalchemy_url(self) -> str:
        return normalise_async_url(self.database_url)

    @property
    def DATABASE_URL(self) -> str:  # noqa: N802 - compatibility alias
        return self.database_url

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key)

    @property
    def embedding_url(self) -> str:
        """Embedding endpoint, falling back to the LLM base URL."""

        return normalise_http_url(self.embedding_base_url or self.llm_base_url)

    @property
    def embedding_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key

    @property
    def embedding_configured(self) -> bool:
        return bool(self.embedding_url and self.embedding_model)

    @property
    def tools_available(self) -> list[str]:
        return [t.strip() for t in str(self.worker_tools or "").split(",") if t.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in str(self.cors_origins or "").split(",") if o.strip()]

    def llm_model_for(self, role: str) -> str:
        """Return the model id for ``planning|execution|verification|reporting``."""

        key = (role or "execution").lower()
        mapping = {
            "planning": self.llm_model_planning,
            "planner": self.llm_model_planning,
            "execution": self.llm_model_execution,
            "executor": self.llm_model_execution,
            "verification": self.llm_model_verification,
            "verifier": self.llm_model_verification,
            "reporting": self.llm_model_reporting,
            "reporter": self.llm_model_reporting,
        }
        return mapping.get(key, self.llm_model_execution)

    def role_models(self) -> Dict[str, str]:
        return {role: self.llm_model_for(role) for role in ROLE_NAMES}

    def llm_endpoint(self, role: str = "execution") -> Dict[str, Any]:
        """Per-role connection descriptor consumed by ``core/llm/client.py``."""

        return {
            "role": role,
            "base_url": normalise_http_url(self.llm_base_url),
            "api_key": self.llm_api_key,
            "model": self.llm_model_for(role),
            "timeout_s": self.llm_timeout_s,
            "max_retries": self.llm_max_retries,
        }

    def embedding_endpoint(self) -> Dict[str, Any]:
        return {
            "base_url": self.embedding_url,
            "api_key": self.embedding_key,
            "model": self.embedding_model,
            "dim": self.embedding_dim,
            "timeout_s": self.embedding_timeout_s,
            "batch_size": self.embedding_batch_size,
            "allow_fallback": self.embedding_allow_fallback,
        }

    def retrieval_thresholds(self) -> Dict[str, float]:
        """Threshold bundle surfaced by ``GET /retrieval/stages``."""

        return {
            "subgraph_top_k": self.subgraph_top_k,
            "jaccard_threshold": self.jaccard_threshold,
            "goal_align_threshold": self.goal_align_threshold,
            "align_alpha": self.align_alpha,
            "align_beta": self.align_beta,
            "align_gamma": self.align_gamma,
            "crosslink_cosine": self.crosslink_cosine,
            "confidence_threshold": self.confidence_threshold,
            "subgraph_max_query_nodes": self.subgraph_max_query_nodes,
            "subgraph_max_pattern_nodes": self.subgraph_max_pattern_nodes,
            "subgraph_timeout_s": self.subgraph_timeout_s,
            "bridge_theta": self.bridge_theta,
        }

    def timing(self) -> Dict[str, float]:
        return {
            "max_rounds": self.max_rounds,
            "main_timeout_s": self.main_timeout_s,
            "degradation_timeout_s": self.degradation_timeout_s,
            "project_lease_s": self.project_lease_s,
            "intent_lease_s": self.intent_lease_s,
            "intent_heartbeat_s": self.intent_heartbeat_s,
            "tick_seconds": self.tick_seconds,
            "stall_ticks": self.stall_ticks,
        }

    def features(self) -> Dict[str, Any]:
        return {
            "intent_guidance_enabled": self.intent_guidance_enabled,
            "verify_escalation_enabled": self.verify_escalation_enabled,
            "count_reason_toward_budget": self.count_reason_toward_budget,
            "sandbox_mode": self.sandbox_mode,
        }

    def as_dict(self, redact: bool = False) -> Dict[str, Any]:
        """Flat settings mapping (optionally with credentials masked)."""

        data: Dict[str, Any] = {}
        for name in type(self).model_fields:  # type: ignore[attr-defined]
            data[name] = getattr(self, name, None)
        if redact:
            for key in list(data):
                lowered = key.lower()
                if any(
                    marker in lowered
                    for marker in ("api_key", "apikey", "token", "secret", "password")
                ):
                    data[key] = "***" if data[key] else ""
            if data.get("database_url"):
                data["database_url"] = _redact_url(str(data["database_url"]))
        return data

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        kwargs.pop("redact", None)
        return super().model_dump(*args, **kwargs)


class Settings(SettingsBase):
    """Concrete settings object.

    When ``pydantic-settings`` is available the environment is read via
    :class:`pydantic_settings.BaseSettings` (supporting ``.env`` files); the
    manual :meth:`SettingsBase.from_env` path is used otherwise.
    """

    if _HAS_PYDANTIC_SETTINGS:  # pragma: no branch - configuration-time decision
        model_config = SettingsConfigDict(  # type: ignore[assignment]
            env_file=(".env", "../.env", "../../.env"),
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
            populate_by_name=True,
        )

    @classmethod
    def load(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        if environ is not None or not _HAS_PYDANTIC_SETTINGS:
            return Settings.from_env(environ)
        try:
            return cls()  # type: ignore[call-arg]
        except Exception:  # pragma: no cover - malformed env
            return Settings.from_env()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce(raw: Any, annotation: Any) -> Any:
    """Best-effort string -> python coercion for the manual env loader."""

    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    target = annotation
    # Optional[bool] / bool
    if target in (bool, Optional[bool]):
        lowered = text.lower()
        if lowered in ("1", "true", "yes", "on", "y"):
            return True
        if lowered in ("0", "false", "no", "off", "n", ""):
            return False
        return bool(text)
    if target in (int, Optional[int]):
        try:
            return int(float(text))
        except ValueError:
            return 0
    if target in (float, Optional[float]):
        try:
            return float(text)
        except ValueError:
            return 0.0
    if target in (str, Optional[str]):
        return text
    return text


def _redact_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except Exception:  # pragma: no cover - defensive
        return "***"
    if not parts.password:
        return url
    netloc = parts.netloc.replace(f":{parts.password}@", ":***@")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_SETTINGS: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the memoised settings singleton (creating it on first use)."""

    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.load()
    return _SETTINGS


def reload_settings(environ: Optional[Mapping[str, str]] = None) -> Settings:
    """Clear the cache and re-read the environment (tests / SIGHUP)."""

    global _SETTINGS
    _SETTINGS = Settings.load(environ)
    return _SETTINGS


#: ``app.api.deps`` and ``backend/alembic/env.py`` import ``settings`` directly.
settings: Settings = get_settings()
