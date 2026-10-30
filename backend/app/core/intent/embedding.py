"""Embedding provider for the intent knowledge plane (paper §4.4).

Paper: ``v = phi(predecessor attribute vector)`` with ``phi`` = Qwen3-Embedding-4B
(2560-d).  Every vector that enters Algorithm 2 stage S1 (Eq. 2) or Algorithm 3's
``cos(state_vector, phi(i*.objective))`` gate comes through this module.

Because the paper's embedding endpoint is not published, the provider implements the
documented fallback chain (``LIMITATIONS.md`` §2):

    api  ->  sentence-transformers  ->  hash (deterministic)

* ``api``                     : OpenAI-compatible ``POST {base_url}/embeddings``.
* ``sentence-transformers``   : optional local model (torch not required at import).
* ``hash``                    : pure-python deterministic bag-of-token hashing, always
                                available, always the last resort.

The *active backend is always recorded* (``provider.backend``) so retrieval traces and
the Settings capability matrix can show which vectors produced a recommendation rather
than silently degrading.

Also implemented here:

* :func:`cosine_similarity` -- Eq. 2 ``sim_cos(v, phi(i)) = (v . phi(i)) / (|v||phi(i)|)``.
* :func:`embedding_cache_key` -- cache keyed by ``(model, sha256(text))`` (plan C11).
* :func:`attribute_vector_text` -- the canonical serialisation of a fact/predecessor
  attribute set into the text that is embedded ("predecessor attribute vector").
* :func:`objective_text` -- the canonical embedding text for an intent objective /
  ``intent_pattern`` node, used by S1, S4 alignment and the bridge gate.
* :class:`EmbeddingCache` -- bounded LRU so repeated retrieval passes are cheap.
* :func:`get_embedding_provider` / :func:`get_provider` / :func:`default_provider`
  module-level accessors consumed by ``app.api.deps.get_embedding_provider``.

Import-light: no HTTP client, no numpy, no torch at module import time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DIMENSION",
    "DEFAULT_MODEL",
    "EMBEDDING_FALLBACK_CHAIN",
    "BACKEND_API",
    "BACKEND_SENTENCE_TRANSFORMERS",
    "BACKEND_HASH",
    "EmbeddingResult",
    "EmbeddingProvider",
    "EmbeddingCache",
    "EmbeddingError",
    "cosine_similarity",
    "cosine",
    "euclidean_similarity",
    "embedding_cache_key",
    "attribute_vector_text",
    "objective_text",
    "normalise_vector",
    "get_embedding_provider",
    "get_provider",
    "get_embeddings",
    "embed_text",
    "embed_texts",
    "embed_objective",
    "cosine_between_texts",
    "reset_embedding_provider",
    "set_embedding_provider",
    "embedding_fallback_chain",
    "status",
]

# --------------------------------------------------------------------------------------
# Constants (mirrors app.core.intent.EMBEDDING_FALLBACK_CHAIN / app.config defaults)
# --------------------------------------------------------------------------------------

DEFAULT_DIMENSION: int = 2560
DEFAULT_MODEL: str = "Qwen3-Embedding-4B"

BACKEND_API: str = "api"
BACKEND_SENTENCE_TRANSFORMERS: str = "sentence-transformers"
BACKEND_HASH: str = "hash"

EMBEDDING_FALLBACK_CHAIN: Tuple[str, ...] = (
    BACKEND_API,
    BACKEND_SENTENCE_TRANSFORMERS,
    BACKEND_HASH,
)

_DEFAULT_CACHE_SIZE = 4096
_HTTP_TIMEOUT_S = 30.0
_HASH_NGRAM = 2
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9_./:{}\-]+", re.IGNORECASE)


class EmbeddingError(RuntimeError):
    """Raised when no backend in the fallback chain can produce a vector.

    In practice the ``hash`` backend is always available, so this is only raised when
    a caller explicitly requests a single backend via ``backend=`` and it is unusable.
    """

    def __init__(self, message: str, *, backend: Optional[str] = None, detail: Any = None):
        super().__init__(message)
        self.backend = backend
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        return {"code": "EMBEDDING_UNAVAILABLE", "message": str(self), "backend": self.backend}


# --------------------------------------------------------------------------------------
# Small numeric helpers (pure python; numpy is optional and never required)
# --------------------------------------------------------------------------------------


def normalise_vector(vector: Sequence[float], *, dimension: Optional[int] = None) -> List[float]:
    """L2-normalise ``vector`` and (optionally) pad/truncate to ``dimension``.

    Normalising at write time makes Eq. 2 a plain dot product, but the public
    :func:`cosine_similarity` still divides by the norms so that un-normalised
    vectors (e.g. hand-built test vectors) behave identically.
    """
    values = [float(v) for v in (vector or [])]
    if dimension is not None and dimension > 0:
        if len(values) < dimension:
            values.extend([0.0] * (dimension - len(values)))
        elif len(values) > dimension:
            values = values[:dimension]
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0.0:
        return values
    return [v / norm for v in values]


def _l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in vector))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Eq. 2 -- ``sim_cos(v, phi(i)) = (v . phi(i)) / (||v|| ||phi(i)||)``.

    Returns ``0.0`` for empty/degenerate vectors instead of raising so retrieval can
    keep scoring the remaining candidates.  Length mismatch is tolerated by comparing
    the overlapping prefix (defensive: vectors from different backends must never be
    silently combined, callers compare ``provider.backend``/``model`` for that).
    """
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    for i in range(n):
        dot += float(a[i]) * float(b[i])
    na = _l2_norm(a[:n])
    nb = _l2_norm(b[:n])
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


# Short alias used by stage modules / bridge.
cosine = cosine_similarity


def euclidean_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """``1 / (1 + ||a - b||)`` -- secondary distance used only in traces."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dist = math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)))
    return 1.0 / (1.0 + dist)


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


def embedding_cache_key(model: str, text: str) -> str:
    """Cache key ``(model, text hash)`` per plan C11."""
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return hashlib.sha256(f"{model}:{digest}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Bounded LRU cache of ``cache_key -> vector`` (process-local)."""

    def __init__(self, maxsize: int = _DEFAULT_CACHE_SIZE):
        self.maxsize = max(1, int(maxsize))
        self._data: "OrderedDict[str, List[float]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[List[float]]:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, vector: Sequence[float]) -> List[float]:
        stored = [float(v) for v in vector]
        with self._lock:
            self._data[key] = stored
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
        return stored

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._data),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
            }


# --------------------------------------------------------------------------------------
# Canonical embedding text
# --------------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(v) for v in value]
        if isinstance(value, (set, frozenset)):
            items = sorted(items, key=lambda v: json.dumps(v, sort_keys=True, default=str))
        return items
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    enum_value = getattr(value, "value", None)
    if enum_value is not None and not callable(enum_value):
        return _jsonable(enum_value)
    return str(value)


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = unicodedata.normalize("NFKC", str(value))
    return _WS_RE.sub(" ", text).strip()


def attribute_vector_text(attributes: Any = None, *, kind: Any = None, value: Any = None) -> str:
    """Serialise one fact's attributes into the text that is embedded (``v`` in S1).

    Deterministic: attributes are sorted so the same fact always yields the same
    vector and therefore the same S1 ranking (reproducible retrieval traces).
    """
    parts: List[str] = []
    if kind is not None:
        parts.append(f"kind={_normalise_text(getattr(kind, 'value', kind))}")
    if isinstance(attributes, Mapping):
        for key in sorted(attributes.keys(), key=lambda k: str(k)):
            val = attributes[key]
            if val is None or isinstance(val, (dict, list, tuple)) and not val:
                continue
            parts.append(f"{_normalise_text(key)}={_normalise_text(_jsonable(val))}")
    elif attributes not in (None, ""):
        parts.append(_normalise_text(_jsonable(attributes)))
    if value not in (None, "", [], {}):
        parts.append(f"value={_normalise_text(_jsonable(value))}")
    return " | ".join(p for p in parts if p)


def objective_text(objective: Any, *, tactic: Any = None, target: Any = None,
                   action_type: Any = None) -> str:
    """Canonical text for an intent objective / ``intent_pattern`` node.

    Used by S1 (pattern side of Eq. 2), S4 alignment (``cos(obj, goal)``) and the
    bridge's ``cos(state_vector, phi(i*.objective))`` gate.
    """
    if isinstance(objective, Mapping):
        return attribute_vector_text(objective)
    parts: List[str] = []
    if action_type is not None:
        parts.append(f"action={_normalise_text(getattr(action_type, 'value', action_type))}")
    if tactic is not None:
        parts.append(f"tactic={_normalise_text(getattr(tactic, 'value', tactic))}")
    if objective is not None:
        parts.append(_normalise_text(_jsonable(objective)))
    if target is not None:
        parts.append(f"target={_normalise_text(_jsonable(target))}")
    return " | ".join(p for p in parts if p)


def _tokenise(text: str) -> List[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return tokens or ([""] if text else [])


def _hash_embedding(text: str, dimension: int) -> List[float]:
    """Deterministic hashing embedder (always available fallback).

    Unigrams + bigrams are hashed into ``dimension`` buckets with a signed
    contribution, then L2-normalised.  This is not a semantic model, but it is
    stable, dependency-free and gives non-degenerate cosines for shared tokens,
    which keeps the pipeline exercisable when no embedding endpoint exists.
    """
    dim = max(1, int(dimension or DEFAULT_DIMENSION))
    vector = [0.0] * dim
    tokens = _tokenise(text)
    grams: List[str] = list(tokens)
    for i in range(len(tokens) - _HASH_NGRAM + 1):
        grams.append("_".join(tokens[i:i + _HASH_NGRAM]))
    if not grams:
        return vector
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) else -1.0
        weight = 1.0 + (len(gram) / 64.0)
        vector[bucket] += sign * weight
    return normalise_vector(vector, dimension=dim)


# --------------------------------------------------------------------------------------
# Result object
# --------------------------------------------------------------------------------------


@dataclass
class EmbeddingResult:
    """One embedding call, carrying the backend that produced it (trace contract)."""

    vector: List[float]
    backend: str
    model: str
    dimension: int
    text_hash: str
    cached: bool = False
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "dimension": self.dimension,
            "text_hash": self.text_hash,
            "cached": self.cached,
            "fallback_reason": self.fallback_reason,
        }


# --------------------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------------------


class EmbeddingProvider:
    """Fallback-chained embedding provider (``api`` -> ``sentence-transformers`` -> ``hash``).

    Constructor: ``EmbeddingProvider(settings=None, *, backend=None, dimension=None,
    model=None, cache_size=None)``.  When ``backend`` is given the chain starts there
    (used by tests); otherwise the configured ``EMBEDDING_BACKEND`` (if any) is tried
    first and the documented chain is walked.

    The provider never raises from :meth:`embed` as long as the ``hash`` backend is
    reachable; it records ``fallback_reason`` so the degradation is visible in the
    retrieval trace and in ``GET /settings/embedding``.
    """

    def __init__(self, settings: Any = None, *, backend: Optional[str] = None,
                 dimension: Optional[int] = None, model: Optional[str] = None,
                 cache_size: Optional[int] = None, timeout_s: Optional[float] = None):
        self.settings = settings
        self.dimension = int(dimension or _settings_value(settings, "embedding_dim", DEFAULT_DIMENSION)
                             or DEFAULT_DIMENSION)
        self.model = str(model or _settings_value(settings, "embedding_model", DEFAULT_MODEL)
                         or DEFAULT_MODEL)
        self.timeout_s = float(timeout_s or _settings_value(settings, "embedding_timeout_s",
                                                            _HTTP_TIMEOUT_S) or _HTTP_TIMEOUT_S)
        self.cache = EmbeddingCache(int(cache_size or _settings_value(settings, "embedding_cache_size",
                                                                      _DEFAULT_CACHE_SIZE)
                                        or _DEFAULT_CACHE_SIZE))
        configured = _settings_value(settings, "embedding_backend", None)
        preferred = backend or configured
        if preferred and str(preferred).lower() in EMBEDDING_FALLBACK_CHAIN:
            start = EMBEDDING_FALLBACK_CHAIN.index(str(preferred).lower())
            self.chain: Tuple[str, ...] = EMBEDDING_FALLBACK_CHAIN[start:]
        else:
            self.chain = EMBEDDING_FALLBACK_CHAIN
        self._active: Optional[str] = None
        self._fallback_reason: Optional[str] = None
        self._st_backend: Any = None
        self._st_checked = False
        self._http_client: Any = None
        self.calls = 0
        self.fallbacks = 0

    # -- configuration ---------------------------------------------------------------

    def _base_url(self) -> Optional[str]:
        return (_settings_value(self.settings, "embedding_base_url", None)
                or _settings_value(self.settings, "llm_base_url", None)
                or os.environ.get("EMBEDDING_BASE_URL")
                or os.environ.get("LLM_BASE_URL"))

    def _api_key(self) -> Optional[str]:
        return (_settings_value(self.settings, "embedding_api_key", None)
                or _settings_value(self.settings, "llm_api_key", None)
                or os.environ.get("EMBEDDING_API_KEY")
                or os.environ.get("LLM_API_KEY"))

    def _api_model(self) -> str:
        return (_settings_value(self.settings, "embedding_model", None)
                or os.environ.get("EMBEDDING_MODEL")
                or self.model)

    @property
    def backend(self) -> str:
        """Backend that will / did produce vectors (resolved lazily, cached)."""
        if self._active:
            return self._active
        return self._resolve_backend()[0]

    @property
    def active_backend(self) -> Optional[str]:
        return self._active

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    @property
    def is_degraded(self) -> bool:
        return (self._active or "") == BACKEND_HASH

    def _resolve_backend(self) -> Tuple[str, Optional[str]]:
        """Walk the chain once; cache the first usable backend and why it was chosen."""
        for name in self.chain:
            if name == BACKEND_API:
                if self._base_url():
                    self._active, self._fallback_reason = BACKEND_API, None
                    return self._active, None
                continue
            if name == BACKEND_SENTENCE_TRANSFORMERS:
                if self._sentence_transformers() is not None:
                    self._active, self._fallback_reason = BACKEND_SENTENCE_TRANSFORMERS, None
                    return self._active, None
                continue
            if name == BACKEND_HASH:
                reason = "no embedding endpoint or local model available"
                self._active, self._fallback_reason = BACKEND_HASH, reason
                log.warning("embedding.fallback_hash", extra={"model": self.model,
                                                              "reason": reason})
                return self._active, reason
        self._active, self._fallback_reason = BACKEND_HASH, "fallback chain exhausted"
        return self._active, self._fallback_reason

    def _sentence_transformers(self) -> Any:
        if self._st_checked:
            return self._st_backend
        self._st_checked = True
        try:  # optional dependency -- never required
            from sentence_transformers import SentenceTransformer  # type: ignore

            name = _settings_value(self.settings, "embedding_st_model", self.model) or self.model
            self._st_backend = SentenceTransformer(str(name))
            log.info("embedding.sentence_transformers_loaded", extra={"model": str(name)})
        except Exception as exc:  # ImportError, OSError (no weights), ...
            log.debug("embedding.sentence_transformers_unavailable", extra={"detail": str(exc)[:200]})
            self._st_backend = None
        return self._st_backend

    def _client(self) -> Any:
        if self._http_client is not None:
            return self._http_client
        try:
            import httpx  # available in the runtime deps; imported lazily
        except Exception:  # pragma: no cover - httpx is a declared dependency
            return None
        headers = {"Content-Type": "application/json"}
        key = self._api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._http_client = httpx.Client(timeout=self.timeout_s, headers=headers)
        return self._http_client

    # -- embedding -------------------------------------------------------------------

    def embed(self, text: Any, *, use_cache: bool = True) -> EmbeddingResult:
        text = _normalise_text(text)
        backend, reason = self._resolve_backend()
        key = embedding_cache_key(f"{backend}:{self._api_model() if backend == BACKEND_API else self.model}",
                                  text)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                return EmbeddingResult(cached, backend, self.model, self.dimension,
                                       hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                       cached=True, fallback_reason=self._fallback_reason)
        vector = self._embed_with(backend, text, reason)
        self.calls += 1
        if use_cache:
            self.cache.put(key, vector)
        return EmbeddingResult(vector, self._active or backend, self.model, self.dimension,
                               hashlib.sha256(text.encode("utf-8")).hexdigest(),
                               cached=False, fallback_reason=self._fallback_reason)

    def _embed_with(self, backend: str, text: str, reason: Optional[str]) -> List[float]:
        if backend == BACKEND_API:
            vector = self._embed_api(text)
            if vector is not None:
                return normalise_vector(vector, dimension=self.dimension)
            self.fallbacks += 1
        elif backend == BACKEND_SENTENCE_TRANSFORMERS:
            vector = self._embed_sentence_transformers(text)
            if vector is not None:
                return normalise_vector(vector, dimension=self.dimension)
            self.fallbacks += 1
        return _hash_embedding(text, self.dimension)

    def _embed_api(self, text: str) -> Optional[List[float]]:
        base = self._base_url()
        if not base:
            return None
        client = self._client()
        if client is None:
            self._fallback_reason = "httpx is not installed"
            return None
        url = base.rstrip("/")
        if not url.endswith("/embeddings"):
            url = f"{url}/embeddings"
        payload = {"model": self._api_model(), "input": text}
        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            self._fallback_reason = f"api embedding failed: {type(exc).__name__}"
            log.warning("embedding.api_failed", extra={"detail": str(exc)[:200], "url": url})
            return None
        vector = _extract_vector(data)
        if vector is None:
            self._fallback_reason = "api embedding returned an unexpected payload"
            return None
        return vector

    def _embed_sentence_transformers(self, text: str) -> Optional[List[float]]:
        model = self._sentence_transformers()
        if model is None:
            return None
        try:
            encoded = model.encode([text], normalize_embeddings=True)
            first = encoded[0]
            return [float(v) for v in first]
        except Exception as exc:
            self._fallback_reason = f"sentence-transformers failed: {type(exc).__name__}"
            log.warning("embedding.st_failed", extra={"detail": str(exc)[:200]})
            return None

    def embed_many(self, texts: Iterable[Any], *, use_cache: bool = True) -> List[EmbeddingResult]:
        return [self.embed(t, use_cache=use_cache) for t in texts]

    def embed_vector(self, text: Any, *, use_cache: bool = True) -> List[float]:
        return self.embed(text, use_cache=use_cache).vector

    def embed_objective(self, objective: Any, *, tactic: Any = None, target: Any = None,
                        action_type: Any = None) -> List[float]:
        return self.embed_vector(objective_text(objective, tactic=tactic, target=target,
                                                action_type=action_type))

    def embed_attributes(self, attributes: Any, *, kind: Any = None, value: Any = None) -> List[float]:
        return self.embed_vector(attribute_vector_text(attributes, kind=kind, value=value))

    def similarity(self, text_a: Any, text_b: Any) -> float:
        return cosine_similarity(self.embed_vector(text_a), self.embed_vector(text_b))

    def vector_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        return cosine_similarity(a, b)

    def status(self) -> Dict[str, Any]:
        backend, reason = self._resolve_backend()
        payload = {
            "ok": True,
            "backend": backend,
            "model": self.model,
            "dimension": self.dimension,
            "fallback_chain": list(EMBEDDING_FALLBACK_CHAIN),
            "active_chain": list(self.chain),
            "degraded": backend == BACKEND_HASH,
            "fallback_reason": reason,
            "base_url": self._base_url(),
            "cache": self.cache.stats(),
            "calls": self.calls,
            "fallbacks": self.fallbacks,
        }
        if backend == BACKEND_HASH:
            payload["note"] = ("deterministic hashing embedder in use; set EMBEDDING_BASE_URL "
                               "or install sentence-transformers for semantic vectors")
        return payload

    # -- aliases used by services ----------------------------------------------------

    def encode(self, text: Any, **kwargs: Any) -> List[float]:
        return self.embed_vector(text, **kwargs)

    def embed_query(self, text: Any) -> List[float]:
        return self.embed_vector(text)

    async def aembed(self, text: Any) -> List[float]:
        return self.embed_vector(text)

    async def aembed_many(self, texts: Iterable[Any]) -> List[List[float]]:
        return [r.vector for r in self.embed_many(texts)]

    def __call__(self, text: Any) -> List[float]:
        return self.embed_vector(text)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"EmbeddingProvider(backend={self.backend!r}, model={self.model!r}, dim={self.dimension})"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _settings_value(settings: Any, name: str, default: Any = None) -> Any:
    """Read ``embedding_*`` settings tolerantly (pydantic model / mapping / object)."""
    if settings is None:
        return default
    for candidate in (name, name.upper()):
        if isinstance(settings, Mapping):
            if candidate in settings:
                return settings[candidate]
            continue
        value = getattr(settings, candidate, None)
        if value is not None:
            return value
    return default


def _extract_vector(payload: Any) -> Optional[List[float]]:
    """Pull the first vector out of an OpenAI-compatible ``/embeddings`` response."""
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        data = payload.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, Mapping):
                vector = first.get("embedding")
            else:
                vector = first
            if isinstance(vector, (list, tuple)) and vector:
                try:
                    return [float(v) for v in vector]
                except (TypeError, ValueError):
                    return None
        vector = payload.get("embedding")
        if isinstance(vector, (list, tuple)) and vector:
            try:
                return [float(v) for v in vector]
            except (TypeError, ValueError):
                return None
    if isinstance(payload, (list, tuple)) and payload:
        first = payload[0]
        if isinstance(first, (int, float)):
            try:
                return [float(v) for v in payload]
            except (TypeError, ValueError):
                return None
        return _extract_vector(first)
    return None


# --------------------------------------------------------------------------------------
# Module-level singleton + convenience API
# --------------------------------------------------------------------------------------

_PROVIDER: Optional[EmbeddingProvider] = None
_PROVIDER_LOCK = threading.Lock()


def _build_settings() -> Any:
    try:
        from app.config import get_settings  # type: ignore

        return get_settings()
    except Exception:
        return None


def get_embedding_provider(settings: Any = None, *, refresh: bool = False) -> EmbeddingProvider:
    """Process-wide provider singleton (created on first use)."""
    global _PROVIDER
    if refresh:
        with _PROVIDER_LOCK:
            _PROVIDER = None
    if _PROVIDER is None:
        with _PROVIDER_LOCK:
            if _PROVIDER is None:
                _PROVIDER = EmbeddingProvider(settings if settings is not None else _build_settings())
    return _PROVIDER


def get_provider(settings: Any = None, *, refresh: bool = False) -> EmbeddingProvider:
    """Alias required by ``app.api.deps.get_embedding_provider`` fallback lookup."""
    return get_embedding_provider(settings, refresh=refresh)


def set_embedding_provider(provider: Optional[EmbeddingProvider]) -> Optional[EmbeddingProvider]:
    global _PROVIDER
    with _PROVIDER_LOCK:
        _PROVIDER = provider
    return _PROVIDER


def reset_embedding_provider() -> None:
    set_embedding_provider(None)


def embed_text(text: Any, *, provider: Optional[EmbeddingProvider] = None) -> List[float]:
    return (provider or get_embedding_provider()).embed_vector(text)


def embed_texts(texts: Iterable[Any], *, provider: Optional[EmbeddingProvider] = None) -> List[List[float]]:
    return [r.vector for r in (provider or get_embedding_provider()).embed_many(texts)]


def embed_objective(objective: Any, *, tactic: Any = None, target: Any = None,
                    action_type: Any = None,
                    provider: Optional[EmbeddingProvider] = None) -> List[float]:
    return (provider or get_embedding_provider()).embed_objective(
        objective, tactic=tactic, target=target, action_type=action_type
    )


def cosine_between_texts(text_a: Any, text_b: Any, *,
                         provider: Optional[EmbeddingProvider] = None) -> float:
    active = provider or get_embedding_provider()
    return cosine_similarity(active.embed_vector(text_a), active.embed_vector(text_b))


def embedding_fallback_chain() -> Tuple[str, ...]:
    return EMBEDDING_FALLBACK_CHAIN


def status() -> Dict[str, Any]:
    return get_embedding_provider().status()


# ``default_provider`` is one of the attributes ``app.api.deps`` probes when resolving
# the embedding dependency; expose it as the singleton for import-time access.
class _DefaultProviderProxy:
    """Lazy proxy so ``from app.core.intent.embedding import default_provider`` is safe."""

    def __getattr__(self, item: str) -> Any:
        return getattr(get_embedding_provider(), item)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return get_embedding_provider()(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        return "default_provider(<lazy EmbeddingProvider>)"


default_provider: Any = _DefaultProviderProxy()

# ``get_embeddings`` mirrors the openai-style helper name used by some call sites.
def get_embeddings(texts: Iterable[Any], *, provider: Optional[EmbeddingProvider] = None) -> List[List[float]]:
    return embed_texts(texts, provider=provider)
