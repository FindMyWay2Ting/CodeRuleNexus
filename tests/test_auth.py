"""用户认证中不依赖数据库的安全契约测试。"""

from hashlib import sha256
import asyncio
import json

import pytest

from app.auth import (
    AuthenticationError,
    AuthenticationMiddleware,
    AuthenticatedPrincipal,
    RateLimitError,
    _validate_registration,
    normalize_email,
    verify_csrf,
)
import app.auth as auth


async def _middleware_request(middleware, path: str, method: str = "GET", headers=None):
    messages = []
    scope = {
        "type": "http", "path": path, "method": method,
        "headers": headers or [(b"host", b"testserver")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await middleware(scope, receive, send)
    start = next(item for item in messages if item["type"] == "http.response.start")
    body = b"".join(item.get("body", b"") for item in messages if item["type"] == "http.response.body")
    return start["status"], json.loads(body)


def _principal(csrf_token: str = "csrf-secret") -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(
        user_id="user-1",
        session_id="session-1",
        email="user@example.com",
        display_name="测试用户",
        department_id=None,
        default_workspace_id="workspace-1",
        csrf_token_hash=sha256(csrf_token.encode("utf-8")).hexdigest(),
    )


def test_email_normalization_is_unicode_and_case_stable():
    assert normalize_email("  USER@Example.COM  ") == "user@example.com"
    assert normalize_email("ｕｓｅｒ@example.com") == "user@example.com"


def test_registration_rejects_weak_password_and_invalid_email():
    with pytest.raises(ValueError, match="有效邮箱"):
        _validate_registration("not-an-email", "LongEnoughPassword!", "测试用户")
    with pytest.raises(ValueError, match="12 到 128"):
        _validate_registration("user@example.com", "too-short", "测试用户")


def test_csrf_requires_matching_cookie_header_and_session_hash():
    principal = _principal()
    assert verify_csrf(principal, "csrf-secret", "csrf-secret") is True
    assert verify_csrf(principal, "csrf-secret", "different") is False
    assert verify_csrf(principal, None, "csrf-secret") is False


def test_audit_failure_does_not_reverse_completed_business_operation(monkeypatch):
    class BrokenConnection:
        def __enter__(self):
            raise RuntimeError("audit database unavailable")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(auth, "connection", lambda: BrokenConnection())

    # 审计是附属写入；失败会记日志，但不能让已提交的删除/上传接口返回假失败。
    auth.record_audit_event("user-1", "document.deleted")


def test_denied_login_records_audit_after_login_transaction_closes(monkeypatch):
    events = []

    class Result:
        def fetchone(self):
            return None

    class LoginConnection:
        def execute(self, *_args, **_kwargs):
            return Result()

        def __enter__(self):
            events.append("transaction_open")
            return self

        def __exit__(self, exc_type, *_args):
            events.append(("transaction_closed", exc_type))
            return False

    monkeypatch.setattr(auth, "connection", lambda: LoginConnection())
    monkeypatch.setattr(auth, "record_audit_event", lambda *_args, **kwargs: events.append(("audit", kwargs["outcome"])))
    monkeypatch.setattr(auth, "_check_login_rate", lambda *_args: None)

    with pytest.raises(AuthenticationError):
        auth.login_user("missing@example.com", "wrong-password")

    assert events == ["transaction_open", ("transaction_closed", None), ("audit", "denied")]


def test_authentication_middleware_rejections_use_stable_error_contract(monkeypatch):
    async def downstream(_scope, _receive, _send):
        raise AssertionError("rejected request must not reach downstream app")

    middleware = AuthenticationMiddleware(downstream)
    monkeypatch.setattr(auth, "authenticate_session", lambda _token: None)
    status, unauthenticated = asyncio.run(_middleware_request(middleware, "/api/auth/me"))
    assert status == 401
    assert unauthenticated == {
        "detail": "需要登录",
        "error_code": "AUTHENTICATION_REQUIRED",
        "retryable": False,
    }

    status, expired = asyncio.run(_middleware_request(
        middleware,
        "/api/auth/me",
        headers=[(b"host", b"testserver"), (b"cookie", b"knowledge_session=expired-token")],
    ))
    assert status == 401
    assert expired["error_code"] == "SESSION_EXPIRED"

    status, bad_origin = asyncio.run(_middleware_request(
        middleware,
        "/api/auth/login",
        "POST",
        [(b"host", b"testserver"), (b"origin", b"https://untrusted.example")],
    ))
    assert status == 403
    assert bad_origin["error_code"] == "ORIGIN_REJECTED"
    assert bad_origin["retryable"] is False


def test_login_rate_limit_has_machine_readable_retry_window(monkeypatch):
    """限流不能与密码错误混为同一个认证异常。"""
    monkeypatch.setattr(auth.time, "monotonic", lambda: 1000.0)
    monkeypatch.setitem(auth._login_attempts, "client:user@example.com", [999.0] * 8)
    with pytest.raises(RateLimitError) as exc_info:
        auth._check_login_rate("client:user@example.com")
    assert exc_info.value.retry_after == 900
