"""Error model and FastAPI exception handlers for the Intentest API.

Contract (see ``API.md`` §"Error envelope")
------------------------------------------
Every non-2xx response produced by this application uses the envelope::

    {"error": {"code": "...", "message": "...", "details": {...}, "request_id": "..."}}

The ``code`` values are enumerated in :class:`ErrorCode` and match ``API.md``::

    NOT_FOUND, VALIDATION_ERROR, ILLEGAL_TRANSITION, PRECONDITION_FAILED,
    SCOPE_VIOLATION, LEASE_HELD, BUDGET_EXHAUSTED, LLM_UNAVAILABLE,
    SANDBOX_UNAVAILABLE, BASELINE_UNAVAILABLE, INTEGRITY_VIOLATION,
    CONFLICT, INTERNAL

Route handlers and services raise :class:`AppError` subclasses; the handlers
registered by :func:`install_exception_handlers` translate them into the
envelope and preserve the ``X-Request-ID`` correlation header.  Nothing in this
module imports the rest of the application at import time (only FastAPI /
Starlette / SQLAlchemy), so it is safe to import from any layer and cannot
create an import cycle.
"""

from __future__ import annotations

import logging
import uuid
from enum import Enum
from typing import Any, Mapping

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

__all__ = [
    "ErrorCode",
    "HTTP_STATUS_BY_CODE",
    "AppError",
    "APIError",
    "NotFoundError",
    "ValidationAppError",
    "IllegalTransitionError",
    "PreconditionFailedError",
    "ScopeViolationError",
    "LeaseHeldError",
    "BudgetExhaustedError",
    "LLMUnavailableError",
    "SandboxUnavailableError",
    "BaselineUnavailableError",
    "BaselineUnavailable",
    "IntegrityViolationError",
    "ConflictError",
    "InternalError",
    "IllegalTransition",
    "PreconditionFailed",
    "ScopeViolation",
    "LeaseHeld",
    "BudgetExhausted",
    "LLMUnavailable",
    "SandboxUnavailable",
    "IntegrityViolation",
    "Conflict",
    "get_request_id",
    "error_body",
    "error_response",
    "install_exception_handlers",
    "register_exception_handlers",
    "setup_exception_handlers",
]


# ---------------------------------------------------------------------------
# Codes / status mapping
# ---------------------------------------------------------------------------
class ErrorCode(str, Enum):
    """Stable machine-readable error codes (see ``API.md``)."""

    NOT_FOUND = "NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    ILLEGAL_TRANSITION = "ILLEGAL_TRANSITION"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    SCOPE_VIOLATION = "SCOPE_VIOLATION"
    LEASE_HELD = "LEASE_HELD"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"
    BASELINE_UNAVAILABLE = "BASELINE_UNAVAILABLE"
    INTEGRITY_VIOLATION = "INTEGRITY_VIOLATION"
    CONFLICT = "CONFLICT"
    INTERNAL = "INTERNAL"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return str(self.value)


HTTP_STATUS_BY_CODE: Mapping[ErrorCode, int] = {
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.ILLEGAL_TRANSITION: 409,
    ErrorCode.PRECONDITION_FAILED: 422,
    ErrorCode.SCOPE_VIOLATION: 403,
    ErrorCode.LEASE_HELD: 409,
    ErrorCode.BUDGET_EXHAUSTED: 409,
    ErrorCode.LLM_UNAVAILABLE: 503,
    ErrorCode.SANDBOX_UNAVAILABLE: 503,
    ErrorCode.BASELINE_UNAVAILABLE: 501,
    ErrorCode.INTEGRITY_VIOLATION: 500,
    ErrorCode.CONFLICT: 409,
    ErrorCode.INTERNAL: 500,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class AppError(Exception):
    """Base class for every *expected* application failure.

    ``code`` is an :class:`ErrorCode` (a plain string is accepted and coerced
    when possible so that adapters may raise without importing the enum).
    """

    code: ErrorCode = ErrorCode.INTERNAL
    status_code: int = 500
    message: str = "Internal server error"

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
        code: ErrorCode | str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.message = message or type(self).message
        self.details: dict[str, Any] = dict(details or {})
        if code is not None:
            self.code = _coerce_code(code)
        if status_code is not None:
            self.status_code = int(status_code)
        elif "status_code" not in type(self).__dict__ or status_code is None:
            # Derive the HTTP status from the code unless a subclass fixed it.
            self.status_code = HTTP_STATUS_BY_CODE.get(self.code, self.status_code)
        super().__init__(self.message)

    # -- serialisation ----------------------------------------------------
    def to_dict(self, request_id: str | None = None) -> dict[str, Any]:
        return {
            "error": {
                "code": _as_str(self.code),
                "message": self.message,
                "details": self.details or {},
                "request_id": request_id,
            }
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(code={_as_str(self.code)!r}, message={self.message!r})"


def _as_str(code: ErrorCode | str) -> str:
    return code.value if isinstance(code, ErrorCode) else str(code)


def _coerce_code(code: ErrorCode | str) -> ErrorCode:
    if isinstance(code, ErrorCode):
        return code
    try:
        return ErrorCode(str(code))
    except ValueError:
        return ErrorCode.INTERNAL


class NotFoundError(AppError):
    code = ErrorCode.NOT_FOUND
    status_code = 404
    message = "Resource not found"


class ValidationAppError(AppError):
    code = ErrorCode.VALIDATION_ERROR
    status_code = 422
    message = "Invalid request payload"


class IllegalTransitionError(AppError):
    code = ErrorCode.ILLEGAL_TRANSITION
    status_code = 409
    message = "Illegal state transition"


class PreconditionFailedError(AppError):
    code = ErrorCode.PRECONDITION_FAILED
    status_code = 422
    message = "Intent preconditions are not satisfied"


class ScopeViolationError(AppError):
    code = ErrorCode.SCOPE_VIOLATION
    status_code = 403
    message = "Target is outside the authorised scope"


class LeaseHeldError(AppError):
    code = ErrorCode.LEASE_HELD
    status_code = 409
    message = "Project lease is held by another scheduler"


class BudgetExhaustedError(AppError):
    code = ErrorCode.BUDGET_EXHAUSTED
    status_code = 409
    message = "Round budget exhausted"


class LLMUnavailableError(AppError):
    code = ErrorCode.LLM_UNAVAILABLE
    status_code = 503
    message = "LLM endpoint is unavailable"


class SandboxUnavailableError(AppError):
    code = ErrorCode.SANDBOX_UNAVAILABLE
    status_code = 503
    message = "Docker sandbox is unavailable"


class BaselineUnavailableError(AppError):
    code = ErrorCode.BASELINE_UNAVAILABLE
    status_code = 501
    message = "Baseline is not reproducible in this environment"


class IntegrityViolationError(AppError):
    code = ErrorCode.INTEGRITY_VIOLATION
    status_code = 500
    message = "Fact-Intent DAG invariant violated"


class ConflictError(AppError):
    code = ErrorCode.CONFLICT
    status_code = 409
    message = "Conflicting state"


class InternalError(AppError):
    code = ErrorCode.INTERNAL
    status_code = 500
    message = "Internal server error"


# Aliases used across the code base / documentation (kept deliberately short so
# that call sites can raise ``ScopeViolation`` or ``IllegalTransition`` without a
# dependency on the API layer's naming).
APIError = AppError
ValidationError = ValidationAppError
IllegalTransition = IllegalTransitionError
PreconditionFailed = PreconditionFailedError
ScopeViolation = ScopeViolationError
LeaseHeld = LeaseHeldError
BudgetExhausted = BudgetExhaustedError
LLMUnavailable = LLMUnavailableError
SandboxUnavailable = SandboxUnavailableError
BaselineUnavailable = BaselineUnavailableError
IntegrityViolation = IntegrityViolationError
Conflict = ConflictError
NotFound = NotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_request_id(request: Request | None) -> str:
    """Return the correlation id for ``request`` (generating one if needed)."""

    if request is None:
        return str(uuid.uuid4())
    state = getattr(request, "state", None)
    rid = getattr(state, "request_id", None) if state is not None else None
    if rid:
        return str(rid)
    header = None
    try:
        header = request.headers.get("x-request-id")
    except Exception:  # pragma: no cover - defensive
        header = None
    rid = header or str(uuid.uuid4())
    if state is not None:
        try:
            state.request_id = rid
        except Exception:  # pragma: no cover - defensive
            pass
    return rid


def error_body(
    code: ErrorCode | str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build the canonical error envelope."""

    return {
        "error": {
            "code": _as_str(code),
            "message": message,
            "details": dict(details or {}),
            "request_id": request_id,
        }
    }


def error_response(
    status_code: int,
    code: ErrorCode | str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
    request_id: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    hdrs = dict(headers or {})
    if request_id:
        hdrs.setdefault("X-Request-ID", request_id)
    return JSONResponse(
        status_code=status_code,
        content=error_body(code, message, details=details, request_id=request_id),
        headers=hdrs,
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    rid = get_request_id(request)
    log = logger.warning if exc.status_code < 500 else logger.error
    log(
        "api.error",
        extra={
            "request_id": rid,
            "code": _as_str(exc.code),
            "status": exc.status_code,
            "path": request.url.path,
            "detail": exc.details or {},
        },
    )
    return error_response(
        exc.status_code,
        exc.code,
        exc.message,
        details=exc.details,
        request_id=rid,
    )


async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    rid = get_request_id(request)
    code = ErrorCode.NOT_FOUND if exc.status_code == 404 else (
        ErrorCode.VALIDATION_ERROR if exc.status_code in (400, 422) else ErrorCode.INTERNAL
    )
    detail = exc.detail
    message = detail if isinstance(detail, str) else "HTTP error"
    details = {} if isinstance(detail, str) else {"detail": detail}
    return error_response(
        exc.status_code,
        code,
        message,
        details=details,
        request_id=rid,
        headers=getattr(exc, "headers", None),
    )


async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    rid = get_request_id(request)
    try:
        errors = exc.errors()
    except Exception:  # pragma: no cover - defensive
        errors = []
    # ``errors()`` may contain non JSON-serialisable ctx objects; flatten them.
    cleaned: list[dict[str, Any]] = []
    for err in errors:
        item = {k: v for k, v in dict(err).items() if k != "ctx"}
        ctx = dict(err).get("ctx")
        if ctx:
            item["ctx"] = {k: str(v) for k, v in ctx.items()}
        cleaned.append(item)
    return error_response(
        422,
        ErrorCode.VALIDATION_ERROR,
        "Request validation failed",
        details={"errors": cleaned},
        request_id=rid,
    )


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    rid = get_request_id(request)
    logger.exception(
        "api.unhandled_exception",
        extra={"request_id": rid, "path": request.url.path},
    )
    return error_response(
        500,
        ErrorCode.INTERNAL,
        "Internal server error",
        details={"type": type(exc).__name__},
        request_id=rid,
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Register the canonical error handlers on ``app`` (idempotent)."""

    app.add_exception_handler(AppError, _handle_app_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _handle_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _handle_unexpected)  # type: ignore[arg-type]
    logger.debug("api.error_handlers_installed")


# Backwards/forwards compatible names -- ``main.py`` may use either spelling.
register_exception_handlers = install_exception_handlers
setup_exception_handlers = install_exception_handlers
