"""API 与 SSE 共用的稳定错误契约。"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


ERROR_CODE_BY_STATUS = {
    400: "BAD_REQUEST",
    401: "AUTHENTICATION_REQUIRED",
    403: "ACCESS_DENIED",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    502: "UPSTREAM_ERROR",
    503: "SERVICE_UNAVAILABLE",
    504: "UPSTREAM_TIMEOUT",
}
RETRYABLE_STATUSES = {429, 502, 503, 504}


class APIError(HTTPException):
    """携带稳定机器码的 HTTP 错误；detail 继续兼容现有页面。"""

    def __init__(
        self,
        status_code: int,
        detail: Any,
        error_code: str,
        *,
        retryable: bool = False,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.retryable = retryable


def error_payload(
    detail: Any,
    *,
    status_code: int,
    error_code: str | None = None,
    retryable: bool | None = None,
) -> dict[str, Any]:
    """构建所有 JSON/SSE 错误都遵循的三个公共字段。"""
    return {
        "detail": detail,
        "error_code": error_code or ERROR_CODE_BY_STATUS.get(status_code, "HTTP_ERROR"),
        "retryable": status_code in RETRYABLE_STATUSES if retryable is None else retryable,
    }
