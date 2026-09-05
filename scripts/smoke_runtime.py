"""对正在运行的 MVP 执行不调用模型、不修改业务资源的 HTTP 冒烟验收。"""

from __future__ import annotations

import argparse
from getpass import getpass
from http.cookiejar import CookieJar
import json
import os
import sys
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener


def request_json(opener, base_url: str, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    """发送同源 JSON 请求，并拒绝把 HTML/纯文本错误误当成 API 响应。"""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(f"{base_url.rstrip('/')}{path}", data=body, headers=headers, method=method)
    try:
        with opener.open(request, timeout=15) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw)
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError as parse_error:
            raise RuntimeError(f"{path} returned non-JSON HTTP {exc.code}") from parse_error
        return exc.code, detail


def csrf_token(cookie_jar: CookieJar) -> str:
    """读取可由前端使用的 CSRF Cookie；HttpOnly Session 不会被打印。"""
    cookie_name = os.getenv("CSRF_COOKIE_NAME", "knowledge_csrf")
    for cookie in cookie_jar:
        if cookie.name == cookie_name:
            return cookie.value
    raise RuntimeError("login response did not set the CSRF cookie")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running knowledge-base MVP")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--email", default=os.getenv("KNOWLEDGE_SMOKE_EMAIL"))
    parser.add_argument(
        "--forbidden-workspace-id",
        default=os.getenv("KNOWLEDGE_SMOKE_FORBIDDEN_WORKSPACE_ID"),
        help="an existing workspace UUID that the smoke account is not allowed to access",
    )
    args = parser.parse_args()
    if not args.email:
        parser.error("provide --email or KNOWLEDGE_SMOKE_EMAIL")
    if not args.forbidden_workspace_id:
        parser.error("provide --forbidden-workspace-id or KNOWLEDGE_SMOKE_FORBIDDEN_WORKSPACE_ID")
    try:
        forbidden_workspace_id = str(uuid.UUID(args.forbidden_workspace_id))
    except ValueError:
        parser.error("forbidden workspace id must be a valid UUID")
    parsed_base = urlparse(args.base_url)
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.hostname:
        parser.error("--base-url must be an absolute HTTP(S) URL")
    loopback_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed_base.scheme == "http" and parsed_base.hostname.casefold() not in loopback_hosts:
        parser.error("plain HTTP is allowed only for localhost/loopback addresses")
    password = os.getenv("KNOWLEDGE_SMOKE_PASSWORD") or getpass("Smoke account password: ")
    if not password:
        parser.error("password is required")

    cookies = CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookies))
    checks: list[tuple[str, bool, str]] = []

    try:
        status, live = request_json(opener, args.base_url, "GET", "/api/health/live")
        checks.append(("Liveness", status == 200 and live.get("status") == "ok", f"HTTP {status}"))
        status, ready = request_json(opener, args.base_url, "GET", "/api/health/ready")
        checks.append(("Readiness", status == 200 and ready.get("status") == "ready", f"HTTP {status}"))

        status, login = request_json(
            opener, args.base_url, "POST", "/api/auth/login",
            {"email": args.email, "password": password},
        )
        checks.append(("Login", status == 200 and login.get("status") == "ok", f"HTTP {status}"))
        if status != 200:
            raise RuntimeError("login failed; remaining authenticated checks were skipped")

        status, me = request_json(opener, args.base_url, "GET", "/api/auth/me")
        user = me.get("user", {})
        checks.append(("Current user", status == 200 and user.get("email", "").casefold() == args.email.casefold(), f"HTTP {status}"))

        status, workspace_response = request_json(opener, args.base_url, "GET", "/api/workspaces")
        workspaces = workspace_response.get("items", [])
        workspace_id = workspace_response.get("current_workspace_id")
        checks.append(("Workspace catalog", status == 200 and bool(workspaces) and bool(workspace_id), f"HTTP {status}, count={len(workspaces)}"))
        if not workspace_id:
            raise RuntimeError("authenticated user has no current workspace")
        visible_workspace_ids = {str(item.get("workspace_id") or item.get("id")) for item in workspaces}
        if forbidden_workspace_id in visible_workspace_ids:
            raise RuntimeError("forbidden workspace is visible to the smoke account; use a workspace without membership")

        query = urlencode({"workspace_id": workspace_id})
        status, scoped_health = request_json(opener, args.base_url, "GET", f"/api/health?{query}")
        checks.append(("Workspace scope", status == 200 and scoped_health.get("workspace_id") == workspace_id, f"HTTP {status}"))

        knowledge_query = urlencode({"source_type": "rag", "workspace_id": workspace_id})
        status, knowledge = request_json(opener, args.base_url, "GET", f"/api/knowledge?{knowledge_query}")
        checks.append(("RAG catalog", status == 200 and isinstance(knowledge.get("items"), list), f"HTTP {status}, count={len(knowledge.get('items', []))}"))

        status, projects = request_json(opener, args.base_url, "GET", f"/api/code-wiki/projects?{query}")
        checks.append(("Code Wiki catalog", status == 200 and isinstance(projects.get("items"), list), f"HTTP {status}, count={len(projects.get('items', []))}"))

        # 使用真实存在但未授权的空间，证明拒绝来自成员校验而不是“空间不存在”。
        forbidden_requests = (
            ("Cross-workspace health denied", f"/api/health?{urlencode({'workspace_id': forbidden_workspace_id})}"),
            ("Cross-workspace RAG denied", f"/api/knowledge?{urlencode({'source_type': 'rag', 'workspace_id': forbidden_workspace_id})}"),
            ("Cross-workspace Code Wiki denied", f"/api/code-wiki/projects?{urlencode({'workspace_id': forbidden_workspace_id})}"),
        )
        for name, path in forbidden_requests:
            status, _ = request_json(opener, args.base_url, "GET", path)
            checks.append((name, status == 403, f"HTTP {status}"))

        logout_request = Request(
            f"{args.base_url.rstrip('/')}/api/auth/logout", data=b"{}", method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json", "X-CSRF-Token": csrf_token(cookies)},
        )
        with opener.open(logout_request, timeout=15) as response:
            logout = json.loads(response.read().decode("utf-8"))
            checks.append(("Logout", response.status == 200 and logout.get("status") == "ok", f"HTTP {response.status}"))

        status, _ = request_json(opener, args.base_url, "GET", "/api/auth/me")
        checks.append(("Session revoked", status == 401, f"HTTP {status}"))
    except (URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        checks.append(("Runtime", False, str(exc)))

    print("Knowledge Base Runtime Smoke Test")
    print("=" * 36)
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'FAIL'}] {name}: {detail}")
    failures = sum(not passed for _, passed, _ in checks)
    print("=" * 36)
    print("All checks passed" if failures == 0 else f"{failures} check(s) failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
