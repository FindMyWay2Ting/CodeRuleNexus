"""发布探针的最小契约测试。"""

import asyncio
import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.testclient import TestClient

from app import main
from app.auth import AuthenticationMiddleware
from app.errors import APIError


def test_health_probes_are_public():
    assert "/api/health/live" in AuthenticationMiddleware.PUBLIC_PATHS
    assert "/api/health/ready" in AuthenticationMiddleware.PUBLIC_PATHS


def test_liveness_does_not_depend_on_database():
    assert main.health_live() == {"status": "ok"}


def test_readiness_checks_database(monkeypatch):
    monkeypatch.setattr(main, "verify_application_schema", lambda: None)
    assert main.health_ready() == {"status": "ready", "database": "ok"}


def test_readiness_hides_database_error(monkeypatch):
    def broken_schema():
        raise RuntimeError("secret connection details")

    monkeypatch.setattr(main, "verify_application_schema", broken_schema)
    with pytest.raises(HTTPException) as exc_info:
        main.health_ready()
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "database unavailable"


def test_unhandled_error_is_stable_json_and_hides_details():
    """未知异常不能退化为前端无法解析的纯文本 Internal Server Error。"""
    request = Request({"type": "http", "method": "GET", "path": "/api/test", "headers": []})
    response = asyncio.run(main.unexpected_error_handler(request, RuntimeError("secret local path")))
    payload = json.loads(response.body)
    assert response.status_code == 500
    assert payload == {
        "detail": "服务内部错误，请稍后重试",
        "error_code": "INTERNAL_ERROR",
        "retryable": True,
    }


def test_http_error_keeps_detail_and_adds_stable_machine_fields():
    request = Request({"type": "http", "method": "GET", "path": "/api/test", "headers": []})
    response = asyncio.run(main.http_error_handler(request, HTTPException(403, "无权访问")))
    assert response.status_code == 403
    assert json.loads(response.body) == {
        "detail": "无权访问",
        "error_code": "ACCESS_DENIED",
        "retryable": False,
    }


def test_error_contract_covers_validation_routing_and_headers():
    """框架生成的 422/404/405 也必须遵守同一 JSON 契约。"""
    test_app = FastAPI()
    test_app.add_exception_handler(StarletteHTTPException, main.http_error_handler)
    test_app.add_exception_handler(RequestValidationError, main.validation_error_handler)

    class Payload(BaseModel):
        value: str

    @test_app.post("/items")
    def create_item(payload: Payload):
        return payload

    @test_app.get("/protected")
    def protected():
        raise HTTPException(401, "需要登录", headers={"WWW-Authenticate": "Session"})

    client = TestClient(test_app)
    validation = client.post("/items", json={})
    missing = client.get("/missing")
    method = client.get("/items")
    protected_response = client.get("/protected")

    assert validation.json() == {
        "detail": "请求参数不完整或格式不正确",
        "error_code": "VALIDATION_ERROR",
        "retryable": False,
    }
    assert missing.json()["error_code"] == "NOT_FOUND"
    assert method.json()["error_code"] == "METHOD_NOT_ALLOWED"
    assert protected_response.json()["error_code"] == "AUTHENTICATION_REQUIRED"
    assert protected_response.headers["www-authenticate"] == "Session"


def test_rate_limit_error_is_retryable_and_preserves_retry_after():
    request = Request({"type": "http", "method": "POST", "path": "/api/auth/login", "headers": []})
    error = APIError(
        429, "登录尝试过多，请稍后再试", "RATE_LIMITED",
        retryable=True, headers={"Retry-After": "900"},
    )
    response = asyncio.run(main.http_error_handler(request, error))
    assert response.status_code == 429
    assert response.headers["retry-after"] == "900"
    assert json.loads(response.body) == {
        "detail": "登录尝试过多，请稍后再试",
        "error_code": "RATE_LIMITED",
        "retryable": True,
    }
