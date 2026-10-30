"""Liveness, readiness and event-replay health routes.

The Intentest deployment exposes three health surfaces (see ``API.md`` and
``ARCHITECTURE.md`` §9):

``GET /healthz``
    Pure liveness.  **No** dependency on Postgres, Docker or the LLM endpoint is
    touched here -- a scheduler/worker/backend process that answers ``/healthz``
    is alive even while the database is restarting.  This is what container
    orchestrators use for restart policies.

``GET /readyz``
    Readiness.  Probes, independently and with per-check timeouts:

    * ``database``  -- ``SELECT 1`` through a short-lived async session.
    * ``docker``    -- the sandbox daemon (``SANDBOX_MODE=docker``) or a
      loudly-flagged degraded ``local`` mode used only by tests.
    * ``llm``       -- the cached capability matrix (``chat`` support required,
      see ``config.require_llm_chat`` / ``LLM_REQUIRED`` semantics).
    * ``embedding`` -- whichever backend the fallback chain resolved
      (``api`` -> ``sentence-transformers`` -> ``hash``); a ``hash`` backend is
      reported but degrades the readiness verdict to ``degraded``.

    Response shape::

        {"status": "ready" | "degraded" | "not_ready",
         "checks": {"database": {"ok": true, "detail": "..."}, ...},
         "version": "...", "service": "backend"}

    The endpoint always answers with HTTP 200 so that the frontend can render
    the individual failures; ``?strict=true`` switches to HTTP 503 when any
    *required* check fails (used by ``docker-compose`` healthchecks).

``GET /healthz/checks``
    Convenience alias returning only the ``checks`` mapping (same semantics as
    ``/readyz`` without the envelope; kept for the Settings UI).

Every handler is defensive: an unreachable dependency becomes
``{"ok": False, "detail": "<reason>"}`` rather than a 500, because readiness is
precisely the endpoint that must still answer while things are broken.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Mapping

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse

from app.logging import get_logger

try:  # pragma: no cover - import shim; app.__init__ is metadata-only
    from app import __version__ as APP_VERSION
except Exception:  # pragma: no cover
    APP_VERSION = "0.0.0"

log = get_logger(__name__)

router = APIRouter(tags=["health"])

#: Per-check timeout budget (seconds).  Readiness must never hang: a wedged
#: dependency is reported as ``ok=False`` instead of blocking the probe.
CHECK_TIMEOUT_S = 3.0

#: Service identity, stamped into every response so that a compose deployment
#: with several containers sharing one image can tell the surfaces apart.
SERVICE_NAME = os.environ.get("INTENTEST_SERVICE_NAME", "backend")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
async def _maybe_await(value: Any) -> Any:
    """Await *value* if it is awaitable, otherwise return it unchanged."""

    if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
        return await value
    return value


def _ok(detail: str = "ok", **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": True, "detail": detail}
    payload.update(extra)
    return payload


def _fail(detail: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "detail": detail}
    payload.update(extra)
    return payload


def _exc_detail(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {text}"[:500]


async def _with_timeout(coro: Any, timeout: float = CHECK_TIMEOUT_S) -> Any:
    return await asyncio.wait_for(coro, timeout=timeout)


# ---------------------------------------------------------------------------
# individual probes
# ---------------------------------------------------------------------------
async def _check_database() -> dict[str, Any]:
    """``SELECT 1`` over a throwaway async session."""

    started = time.perf_counter()
    session = None
    try:
        from app.db import get_session_factory  # noqa: WPS433 (lazy import)

        factory = await _maybe_await(get_session_factory())
        session = factory() if callable(factory) else factory
        from sqlalchemy import text as sql_text  # noqa: WPS433

        await session.execute(sql_text("SELECT 1"))
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return _ok("database reachable", latency_ms=elapsed_ms)
    except asyncio.TimeoutError:
        return _fail(f"database probe exceeded {CHECK_TIMEOUT_S}s")
    except Exception as exc:  # noqa: BLE001 - readiness must never raise
        log.warning("health.database_check_failed", error=str(exc))
        return _fail(_exc_detail(exc))
    finally:
        if session is not None:
            try:
                await session.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass


async def _check_docker() -> dict[str, Any]:
    """Sandbox daemon availability, including the degraded ``local`` mode."""

    try:
        from app.api import deps  # noqa: WPS433

        probe = getattr(deps, "get_sandbox_status", None)
        if callable(probe):
            status = await _maybe_await(probe())
            if isinstance(status, Mapping):
                return {
                    "ok": bool(status.get("ok")),
                    "mode": status.get("mode", "unknown"),
                    "detail": str(status.get("detail", "")),
                }
    except asyncio.TimeoutError:
        return _fail(f"docker probe exceeded {CHECK_TIMEOUT_S}s")
    except Exception as exc:  # noqa: BLE001
        return _fail(_exc_detail(exc))

    # Fallback probe: talk to the daemon directly.  ``docker`` is an optional
    # import at module scope everywhere else, mirror that here.
    try:
        import docker  # noqa: WPS433

        client = docker.from_env()
        try:
            version = client.version()
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        return _ok(
            "docker daemon reachable",
            api_version=str(version.get("ApiVersion", "?")),
            mode="docker",
        )
    except ImportError:
        return _fail("docker SDK is not installed", mode="unavailable")
    except Exception as exc:  # noqa: BLE001
        return _fail(_exc_detail(exc), mode="unavailable")


async def _check_llm() -> dict[str, Any]:
    """Capability matrix summary for the configured chat endpoint."""

    try:
        from app.api import deps  # noqa: WPS433

        getter = getattr(deps, "get_llm_capabilities", None) or getattr(
            deps, "get_llm_client", None
        )
        if not callable(getter):
            return _fail("LLM client provider is not registered")
        caps = await _maybe_await(getter())
        if caps is None:
            return _fail("no LLM capabilities available")

        # Capability objects may be dataclasses, pydantic models or dicts.
        as_dict: dict[str, Any]
        if isinstance(caps, Mapping):
            as_dict = dict(caps)
        elif hasattr(caps, "as_dict") and callable(caps.as_dict):
            as_dict = dict(caps.as_dict())
        elif hasattr(caps, "model_dump") and callable(caps.model_dump):
            as_dict = dict(caps.model_dump())
        else:
            as_dict = {
                key: getattr(caps, key)
                for key in (
                    "model",
                    "chat",
                    "tool_calling",
                    "json_schema",
                    "json_object",
                    "embeddings",
                    "streaming",
                    "max_context",
                    "strategy",
                )
                if hasattr(caps, key)
            }

        chat_ok = bool(as_dict.get("chat", True))
        return {
            "ok": chat_ok,
            "detail": "chat endpoint ready" if chat_ok else "chat capability missing",
            "model": str(as_dict.get("model", "")),
            "strategy": str(as_dict.get("strategy", "")),
            "capabilities": {
                key: as_dict.get(key)
                for key in (
                    "tool_calling",
                    "json_schema",
                    "json_object",
                    "embeddings",
                    "streaming",
                    "max_context",
                )
                if key in as_dict
            },
        }
    except asyncio.TimeoutError:
        return _fail(f"llm probe exceeded {CHECK_TIMEOUT_S}s")
    except Exception as exc:  # noqa: BLE001
        return _fail(_exc_detail(exc))


async def _check_embedding() -> dict[str, Any]:
    """Report the embedding backend resolved by the fallback chain."""

    try:
        from app.api import deps  # noqa: WPS433

        getter = getattr(deps, "get_embedding_provider", None)
        if not callable(getter):
            return _fail("embedding provider is not registered")
        provider = await _maybe_await(getter())
        backend = getattr(provider, "backend", None) or getattr(provider, "name", None)
        dimension = getattr(provider, "dim", None) or getattr(provider, "dimension", None)
        if backend is None and isinstance(provider, Mapping):
            backend = provider.get("backend")
            dimension = provider.get("dim") or provider.get("dimension")
        backend = str(backend or "unknown")
        return {
            "ok": True,
            "detail": f"embedding backend: {backend}",
            "backend": backend,
            "dim": int(dimension) if isinstance(dimension, (int, float)) else dimension,
            "degraded": backend == "hash",
        }
    except asyncio.TimeoutError:
        return _fail(f"embedding probe exceeded {CHECK_TIMEOUT_S}s")
    except Exception as exc:  # noqa: BLE001
        return _fail(_exc_detail(exc))


async def _check_event_bus() -> dict[str, Any]:
    """Verify the Postgres LISTEN/NOTIFY channel can be reached.

    This is a *soft* check: the API still serves REST without the event bus, but
    the graph would stop updating live in the UI, so it is surfaced separately.
    """

    try:
        from app.api import EVENT_CHANNEL  # noqa: WPS433

        from app.db import get_session_factory  # noqa: WPS433

        factory = await _maybe_await(get_session_factory())
        if not callable(factory):
            return _fail("no session factory for the event bus probe")
        session = factory()
        try:
            from sqlalchemy import text as sql_text  # noqa: WPS433

            await session.execute(sql_text("SELECT pg_notify(:ch, :payload)"),
                                  {"ch": EVENT_CHANNEL, "payload": "healthz"})
            await session.commit()
        finally:
            try:
                await session.close()
            except Exception:  # noqa: BLE001
                pass
        return _ok(f"channel {EVENT_CHANNEL} reachable")
    except Exception as exc:  # noqa: BLE001
        return _fail(_exc_detail(exc))


#: Probe registry: name -> coroutine factory.  ``required`` checks flip the
#: aggregate verdict to ``not_ready`` when false-y.
_PROBES: tuple[tuple[str, Any, bool], ...] = (
    ("database", _check_database, True),
    ("docker", _check_docker, True),
    ("llm", _check_llm, True),
    ("embedding", _check_embedding, False),
    ("event_bus", _check_event_bus, False),
)


async def run_readiness_checks() -> dict[str, Any]:
    """Execute every probe concurrently and aggregate the verdict."""

    started = time.perf_counter()
    results = await asyncio.gather(
        *(asyncio.wait_for(probe(), timeout=CHECK_TIMEOUT_S) for _, probe, _ in _PROBES),
        return_exceptions=True,
    )

    checks: dict[str, Any] = {}
    required_failures: list[str] = []
    soft_failures: list[str] = []
    for (name, _probe, required), outcome in zip(_PROBES, results):
        if isinstance(outcome, BaseException):
            checks[name] = _fail(_exc_detail(outcome))
        else:
            checks[name] = outcome
        if not checks[name].get("ok"):
            (required_failures if required else soft_failures).append(name)

    degraded_backend = bool(checks.get("embedding", {}).get("degraded"))
    if required_failures:
        status = "not_ready"
    elif soft_failures or degraded_backend:
        status = "degraded"
    else:
        status = "ready"

    return {
        "status": status,
        "checks": checks,
        "required_failures": required_failures,
        "soft_failures": soft_failures,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "version": APP_VERSION,
        "service": SERVICE_NAME,
    }


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
@router.get("/healthz", summary="Liveness probe (no dependency checks)")
async def healthz() -> dict[str, Any]:
    """Return a constant liveness payload.

    Deliberately touches **no** external dependency: Postgres, Docker and the
    LLM endpoint may all be down while the process is still alive.
    """

    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": APP_VERSION,
        "checks": {},
    }


@router.get("/healthz/checks", summary="Readiness checks (checks mapping only)")
async def healthz_checks() -> dict[str, Any]:
    """Alias of ``/readyz`` returning only the ``checks`` mapping."""

    payload = await run_readiness_checks()
    payload.pop("checks", None) if False else None  # keep mypy quiet, no-op
    return dict(payload.get("checks", {}))


@router.get("/readyz", summary="Readiness probe (database, docker, llm, embedding)")
async def readyz(
    response: Response,
    strict: bool = Query(
        default=False,
        description="Return HTTP 503 when a required check fails.",
    ),
) -> Any:
    """Aggregate readiness across every external dependency."""

    payload = await run_readiness_checks()
    if payload["status"] == "not_ready":
        log.warning(
            "health.not_ready",
            failed=payload.get("required_failures"),
            soft_failed=payload.get("soft_failures"),
        )
        if strict:
            return JSONResponse(status_code=503, content=payload)
        response.status_code = 200
    elif payload["status"] == "degraded":
        log.info(
            "health.degraded",
            soft_failed=payload.get("soft_failures"),
        )
    return payload


@router.get("/health", include_in_schema=False)
async def health_alias() -> dict[str, Any]:
    """Backwards-compatible alias for ``/healthz``."""

    return await healthz()


@router.get("/healthz/version", summary="Build/service metadata")
async def version_info(request: Request) -> dict[str, Any]:
    """Small metadata endpoint used by the frontend footer."""

    from app.api import API_PREFIX, EVENT_CHANNEL, WS_PATH  # noqa: WPS433

    return {
        "service": SERVICE_NAME,
        "version": APP_VERSION,
        "api_prefix": API_PREFIX,
        "ws_path": WS_PATH,
        "event_channel": EVENT_CHANNEL,
        "pid": os.getpid(),
        "app": getattr(request.app, "title", ""),
    }


__all__ = [
    "router",
    "healthz",
    "readyz",
    "healthz_checks",
    "health_alias",
    "version_info",
    "run_readiness_checks",
    "CHECK_TIMEOUT_S",
    "SERVICE_NAME",
]
