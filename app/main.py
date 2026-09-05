from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import re
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
import bleach
import json
import logging
import markdown
from time import perf_counter
from typing import Literal
from pydantic import BaseModel

from .config import settings
from .bootstrap import verify_application_schema
from .access_control import (
    AccessScope, WorkspaceNameConflict, accept_workspace_invitation, can_manage_document, can_read_document,
    create_workspace, create_workspace_invitation, leave_workspace,
    list_user_workspaces, list_workspace_members, remove_workspace_member, require_workspace_role,
    resolve_scope, revoke_workspace_invitation, set_default_workspace, transfer_workspace_ownership,
    update_workspace_member_role,
)
from .auth import (
    AuthenticationError, AuthenticationMiddleware, HandoverError, RateLimitError, RegistrationConflict, SessionBundle,
    accept_data_handover, change_password, create_data_handover, current_principal, login_user, register_user,
    record_audit_event, revoke_all_sessions, revoke_session, session_is_active,
)
from .agent_loop import AgentLoopState
from .adaptive_retrieval import AdaptiveRetrievalState
from .knowledge_agent import (
    KnowledgeAgentState,
    KnowledgeDecision,
    KnowledgeEvidence,
    KnowledgeToolResult,
    render_claim_contract_answer,
    validate_answer_contract,
)
from .code_agent import (
    CODE_AGENT_TOOLS,
    CitationRegistry,
    analysis_message_for_tool,
    analysis_result_message,
    assistant_message_dict,
    build_code_agent_messages,
    execute_code_agent_tool,
    tool_result_message,
)
from .code_wiki import (
    DEFAULT_REPOSITORY_ROOT,
    CodeImportValidationError,
    IGNORED_DIRS,
    get_code_overview,
    get_code_symbol,
    delete_code_projects,
    import_and_scan_github_repository,
    RepositoryImportBusy,
    list_code_files,
    list_code_architecture,
    list_code_architecture_links,
    list_code_config_facts,
    list_code_projects,
    managed_local_repository_path,
    managed_code_project_id,
    existing_managed_project_id,
    normalize_uploaded_path,
    persist_scan,
    read_code_source,
    repository_import_lock,
    repository_lock_resource,
    scan_project,
    search_code_symbols,
    trace_code_call_chain,
)
from .db import (
    connection,
    delete_document,
    hybrid_search,
    expand_context,
    invalidate_document,
    list_document_revisions,
    list_knowledge,
    restore_document,
)
from .ingestion import ingest_file
from .errors import APIError, error_payload
from .llm import answer, decide_next_knowledge_action, embed, grade_rag_evidence, plan_knowledge_query, rerank, rewrite_rag_queries, stream_answer, stream_code_agent_completion, stream_code_investigation_note


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时只检查 schema；迁移必须由独立初始化命令完成。"""
    verify_application_schema()
    yield


app = FastAPI(title="Two-Level Knowledge Base MVP", version="0.1.0", lifespan=lifespan)
app.add_middleware(AuthenticationMiddleware)
app.mount("/picture", StaticFiles(directory="picture"), name="picture")
logger = logging.getLogger("knowledge.observability")


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """为业务 HTTP 错误补充稳定机器字段，同时保留 detail 前端契约。"""
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(
            exc.detail,
            status_code=exc.status_code,
            error_code=getattr(exc, "error_code", None),
            retryable=getattr(exc, "retryable", None),
        ),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, _exc: RequestValidationError) -> JSONResponse:
    """请求体错误不回显用户输入，尤其避免密码出现在调试信息中。"""
    return JSONResponse(
        status_code=422,
        content=error_payload("请求参数不完整或格式不正确", status_code=422),
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """保证未处理异常始终返回 JSON；内部细节只进入服务日志。"""
    logger.exception("unhandled_request_error method=%s path=%s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=error_payload(
            "服务内部错误，请稍后重试",
            status_code=500,
            error_code="INTERNAL_ERROR",
            retryable=True,
        ),
    )


def _cost(tokens: int, price_per_1k: float) -> float:
    """按配置单价估算费用；单价为 0 时保留 0，表示尚未配置计费信息。"""
    return round(tokens / 1000 * price_per_1k, 8)


def _sse_event(event: str, payload: dict) -> str:
    """把一个事件编码成浏览器可解析的 SSE 文本；JSON 保证中文和结构字段不丢失。"""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _raise_child_agent_error(payload: dict, fallback: str) -> None:
    """保留嵌套 Agent 的权限撤销语义，其他子任务错误仍按执行失败处理。"""
    message = str(payload.get("message") or fallback)
    if payload.get("error_code") == "AUTHORIZATION_REVOKED":
        raise PermissionError(message)
    raise RuntimeError(message)


def _with_agentic_dispatch_trace(
    response: StreamingResponse,
    decision: KnowledgeDecision,
    *,
    planner_fallback: bool,
) -> StreamingResponse:
    """在下层执行器事件前追加顶层路由决定，不复制或篡改下层结果。"""
    async def traced_iterator():
        route = decision.route or "rag"
        yield _sse_event("step", {
            "step": "knowledge_planner", "status": "completed",
            "message": f"Agent 已选择 {route} 路径",
            "metrics": {"route": route, "planner_fallback": planner_fallback},
        })
        yield _sse_event("model", {
            "id": "knowledge_planner", "role": "Knowledge Agent · 顶层规划",
            "phase": "investigation", "status": "started",
        })
        summary = (
            f"我把问题识别为「{decision.intent or '知识查询'}」，决定使用 {route}。"
            f"{('需要核验：' + '；'.join(decision.sub_questions) + '。') if decision.sub_questions else ''}"
            f"{decision.reason}"
        )
        for offset in range(0, len(summary), 4):
            yield _sse_event("model", {
                "id": "knowledge_planner", "role": "Knowledge Agent · 顶层规划",
                "phase": "investigation", "status": "streaming", "delta": summary[offset:offset + 4],
            })
        yield _sse_event("model", {
            "id": "knowledge_planner", "role": "Knowledge Agent · 顶层规划",
            "phase": "investigation", "status": "completed",
        })
        async for chunk in response.body_iterator:
            yield chunk

    return StreamingResponse(
        traced_iterator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _public_agent_trace(event_id: str, role: str, message: str, chunk_size: int = 4):
    """把可由执行状态核验的 Agent 调查说明拆成增量事件，不冒充模型私有思维链。"""
    yield _sse_event("model", {
        "id": event_id, "role": role, "phase": "investigation", "status": "started",
    })
    for offset in range(0, len(message), chunk_size):
        yield _sse_event("model", {
            "id": event_id, "role": role, "phase": "investigation", "status": "streaming",
            "delta": message[offset:offset + chunk_size],
        })
    yield _sse_event("model", {
        "id": event_id, "role": role, "phase": "investigation", "status": "completed",
    })


def _next_stream_item(iterator):
    """在线程中安全读取同步模型流；用 None 表示结束，避免传播 StopIteration。"""
    return next(iterator, None)


def render_answer_markdown(answer_text: str) -> str:
    """将模型 Markdown 转为安全 HTML，只允许回答页面需要的标签。"""
    html = markdown.markdown(
        answer_text,
        extensions=["extra", "sane_lists", "nl2br"],
    )
    return bleach.clean(
        html,
        tags={
            "p", "br", "strong", "em", "del", "blockquote", "code", "pre",
            "ol", "ul", "li", "h1", "h2", "h3", "h4", "hr", "a",
            "table", "thead", "tbody", "tr", "th", "td",
        },
        attributes={"a": ["href", "title", "target", "rel"]},
        protocols={"http", "https"},
        strip=True,
    )


class ChatRequest(BaseModel):
    message: str
    # 前端请求的空间只能通过服务端成员关系校验后生效。
    workspace_id: str | None = None


class KnowledgeStreamRequest(BaseModel):
    """统一知识入口的请求契约；模式是服务端约束，不是模型提示。"""

    message: str
    mode: Literal["rag", "codewiki", "hybrid", "auto"] = "auto"
    project_id: str | None = None
    workspace_id: str | None = None


class CodeAgentRequest(BaseModel):
    project_id: str
    message: str
    # 顶层 Knowledge Agent 可下发剩余预算；独立 Code Wiki 请求仍使用服务默认值。
    max_rounds: int | None = None
    max_tool_calls: int | None = None
    evidence_only: bool = False
    workspace_id: str | None = None


class PathRequest(BaseModel):
    path: str
    workspace_id: str | None = None


class GithubRepositoryRequest(BaseModel):
    repository_url: str
    workspace_id: str | None = None


class DeleteCodeProjectsRequest(BaseModel):
    project_ids: list[str]
    workspace_id: str | None = None


class InvalidateRequest(BaseModel):
    reason: str | None = None
    workspace_id: str | None = None


class CreateWorkspaceRequest(BaseModel):
    """创建空间时只接收显示名称；owner 必须由服务端身份决定。"""

    workspace_name: str


class DefaultWorkspaceRequest(BaseModel):
    """用户主动切换后的默认空间，仍需服务端成员关系校验。"""

    workspace_id: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str
    bootstrap_token: str | None = None
    handover_token: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class CreateDataHandoverRequest(BaseModel):
    """交接码绑定接收邮箱，并要求原员工重新输入密码确认。"""

    target_email: str
    current_password: str


class AcceptDataHandoverRequest(BaseModel):
    """接收人必须已登录，服务端还会校验令牌绑定邮箱。"""

    token: str


class WorkspaceInvitationRequest(BaseModel):
    email: str
    role: Literal["admin", "editor", "viewer"] = "editor"


class AcceptInvitationRequest(BaseModel):
    token: str


class WorkspaceMemberRoleRequest(BaseModel):
    role: Literal["admin", "editor", "viewer"]


class TransferOwnershipRequest(BaseModel):
    target_user_id: str


def _request_metadata(request: Request) -> tuple[str | None, str | None]:
    return (request.client.host if request.client else None, request.headers.get("user-agent"))


def _set_session_cookies(response: Response, bundle: SessionBundle) -> None:
    """Session 不对 JavaScript 开放；CSRF Token 可读但不能单独用于认证。"""
    max_age = max(1, int((bundle.expires_at - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(
        settings.session_cookie_name, bundle.session_token, max_age=max_age, httponly=True,
        secure=settings.session_cookie_secure, samesite="lax", path="/",
    )
    response.set_cookie(
        settings.csrf_cookie_name, bundle.csrf_token, max_age=max_age, httponly=False,
        secure=settings.session_cookie_secure, samesite="lax", path="/",
    )


def _clear_session_cookies(response: Response) -> None:
    response.delete_cookie(settings.session_cookie_name, path="/")
    response.delete_cookie(settings.csrf_cookie_name, path="/")


def _principal_payload() -> dict:
    principal = current_principal()
    return {
        "user_id": principal.user_id, "email": principal.email,
        "display_name": principal.display_name, "department_id": principal.department_id,
        "default_workspace_id": principal.default_workspace_id,
    }


@app.post("/api/auth/register", status_code=201)
def register(request_body: RegisterRequest, request: Request, response: Response) -> dict:
    try:
        bundle = register_user(
            request_body.email, request_body.password, request_body.display_name, *_request_metadata(request),
            bootstrap_token=request_body.bootstrap_token,
            handover_token=request_body.handover_token,
        )
    except RegistrationConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except RateLimitError as exc:
        raise APIError(
            429, str(exc), "RATE_LIMITED", retryable=True,
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except HandoverError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    _set_session_cookies(response, bundle)
    return {"status": "ok", "user": {
        "user_id": bundle.principal.user_id, "email": bundle.principal.email,
        "display_name": bundle.principal.display_name, "department_id": bundle.principal.department_id,
        "default_workspace_id": bundle.principal.default_workspace_id,
    }}


@app.post("/api/auth/login")
def login(request_body: LoginRequest, request: Request, response: Response) -> dict:
    try:
        bundle = login_user(request_body.email, request_body.password, *_request_metadata(request))
    except RateLimitError as exc:
        raise APIError(
            429, str(exc), "RATE_LIMITED", retryable=True,
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except AuthenticationError as exc:
        raise APIError(401, str(exc), "INVALID_CREDENTIALS") from exc
    _set_session_cookies(response, bundle)
    return {"status": "ok", "user": {
        "user_id": bundle.principal.user_id, "email": bundle.principal.email,
        "display_name": bundle.principal.display_name, "department_id": bundle.principal.department_id,
        "default_workspace_id": bundle.principal.default_workspace_id,
    }}


@app.get("/api/auth/me")
def me() -> dict:
    return {"user": _principal_payload()}


@app.post("/api/auth/logout")
def logout(response: Response) -> dict:
    principal = current_principal()
    revoke_session(principal.session_id, principal.user_id)
    _clear_session_cookies(response)
    return {"status": "ok"}


@app.post("/api/auth/logout-all")
def logout_all(response: Response) -> dict:
    principal = current_principal()
    revoke_all_sessions(principal.user_id)
    _clear_session_cookies(response)
    return {"status": "ok"}


@app.post("/api/auth/change-password")
def update_password(request_body: ChangePasswordRequest, request: Request, response: Response) -> dict:
    principal = current_principal()
    try:
        bundle = change_password(
            principal.user_id, request_body.current_password, request_body.new_password,
            *_request_metadata(request),
        )
    except AuthenticationError as exc:
        raise APIError(401, str(exc), "INVALID_CURRENT_PASSWORD") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    _set_session_cookies(response, bundle)
    return {"status": "ok"}


@app.post("/api/auth/data-handovers", status_code=201)
def create_handover(request_body: CreateDataHandoverRequest) -> dict:
    """由离职员工生成 24 小时有效、仅指定员工可使用的一次性交接码。"""
    principal = current_principal()
    try:
        handover = create_data_handover(
            principal.user_id, request_body.target_email, request_body.current_password,
        )
    except AuthenticationError as exc:
        raise APIError(401, str(exc), "INVALID_CURRENT_PASSWORD") from exc
    except HandoverError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"status": "ok", "handover": handover}


@app.post("/api/auth/data-handovers/accept")
def accept_handover(request_body: AcceptDataHandoverRequest) -> dict:
    """接收数据并立即撤销原账号全部会话；整个迁移在一个数据库事务中完成。"""
    principal = current_principal()
    try:
        summary = accept_data_handover(principal.user_id, request_body.token)
    except HandoverError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"status": "ok", "handover": summary}


@app.get("/api/health")
def health(workspace_id: str | None = Query(default=None)) -> dict:
    scope = request_scope(workspace_id) if workspace_id else request_scope()
    return {"status": "ok", "workspace_id": scope.workspace_id, "workspace_name": scope.workspace_name}


@app.get("/api/health/live")
def health_live() -> dict:
    """公开存活探针：只确认 Web 进程可以响应，不读取业务数据。"""
    return {"status": "ok"}


@app.get("/api/health/ready")
def health_ready() -> dict:
    """公开就绪探针：验证 PostgreSQL 和目标 schema，不暴露业务数据。"""
    try:
        verify_application_schema()
    except Exception as exc:
        logger.warning("readiness_check_failed", exc_info=exc)
        raise HTTPException(503, "database unavailable") from exc
    return {"status": "ready", "database": "ok"}


@app.get("/api/workspaces")
def workspaces() -> dict:
    """返回当前登录用户已加入的空间，作为前端切换的授权候选集。"""
    principal = current_principal()
    items = list_user_workspaces(principal.user_id)
    current_workspace_id = principal.default_workspace_id or (items[0]["workspace_id"] if items else None)
    return {"items": items, "current_workspace_id": current_workspace_id}


@app.post("/api/workspaces", status_code=201)
def create_workspace_endpoint(request: CreateWorkspaceRequest) -> dict:
    """创建独立工作空间，并将当前用户原子登记为 owner。"""
    try:
        workspace = create_workspace(request.workspace_name, current_principal().user_id)
    except WorkspaceNameConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    record_audit_event(
        current_principal().user_id, "workspace.created", workspace_id=workspace["workspace_id"],
        object_type="workspace", object_id=workspace["workspace_id"],
    )
    return {"status": "ok", "workspace": workspace}


@app.patch("/api/users/me/default-workspace")
def change_default_workspace(request: DefaultWorkspaceRequest) -> dict:
    """持久化空间切换，使下次登录仍进入同一授权空间。"""
    principal = current_principal()
    try:
        workspace = set_default_workspace(principal.user_id, request.workspace_id.strip())
    except PermissionError as exc:
        raise APIError(403, str(exc), "WORKSPACE_ACCESS_DENIED") from exc
    return {"status": "ok", "workspace": workspace}


@app.get("/api/workspaces/{workspace_id}/members")
def workspace_members(workspace_id: str) -> dict:
    try:
        return list_workspace_members(request_scope(workspace_id))
    except PermissionError as exc:
        raise APIError(403, str(exc), "INSUFFICIENT_WORKSPACE_ROLE") from exc


@app.post("/api/workspaces/{workspace_id}/invitations", status_code=201)
def invite_workspace_member(workspace_id: str, request: WorkspaceInvitationRequest) -> dict:
    try:
        invitation = create_workspace_invitation(request_scope(workspace_id), request.email, request.role)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    record_audit_event(
        current_principal().user_id, "workspace.invitation_created", workspace_id=workspace_id,
        object_type="workspace_invitation", object_id=invitation["invitation_id"],
        details={"email": invitation["email"], "role": invitation["role"]},
    )
    return {"status": "ok", "invitation": invitation}


@app.delete("/api/workspaces/{workspace_id}/invitations/{invitation_id}")
def cancel_workspace_invitation(workspace_id: str, invitation_id: str) -> dict:
    try:
        found = revoke_workspace_invitation(request_scope(workspace_id), parse_document_id(invitation_id))
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not found:
        raise HTTPException(404, "invitation not found")
    record_audit_event(
        current_principal().user_id, "workspace.invitation_revoked", workspace_id=workspace_id,
        object_type="workspace_invitation", object_id=invitation_id,
    )
    return {"status": "ok"}


@app.post("/api/invitations/accept")
def accept_invitation(request: AcceptInvitationRequest) -> dict:
    principal = current_principal()
    try:
        workspace = accept_workspace_invitation(request.token, principal.user_id, principal.email)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    record_audit_event(
        principal.user_id, "workspace.invitation_accepted", workspace_id=workspace["workspace_id"],
        object_type="workspace", object_id=workspace["workspace_id"],
        details={"role": workspace["role"]},
    )
    return {"status": "ok", "workspace": workspace}


@app.patch("/api/workspaces/{workspace_id}/members/{user_id}")
def change_workspace_member_role(workspace_id: str, user_id: str, request: WorkspaceMemberRoleRequest) -> dict:
    try:
        found = update_workspace_member_role(request_scope(workspace_id), user_id, request.role)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not found:
        raise HTTPException(404, "member not found or role cannot be changed")
    record_audit_event(
        current_principal().user_id, "workspace.member_role_changed", workspace_id=workspace_id,
        object_type="user", object_id=user_id, details={"role": request.role},
    )
    return {"status": "ok", "role": request.role}


@app.delete("/api/workspaces/{workspace_id}/members/{user_id}")
def delete_workspace_member(workspace_id: str, user_id: str) -> dict:
    try:
        found = remove_workspace_member(request_scope(workspace_id), user_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not found:
        raise HTTPException(404, "member not found or cannot be removed")
    record_audit_event(
        current_principal().user_id, "workspace.member_removed", workspace_id=workspace_id,
        object_type="user", object_id=user_id,
    )
    return {"status": "ok"}


@app.post("/api/workspaces/{workspace_id}/leave")
def leave_current_workspace(workspace_id: str) -> dict:
    principal = current_principal()
    try:
        leave_workspace(request_scope(workspace_id))
    except (PermissionError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    record_audit_event(
        principal.user_id, "workspace.left", workspace_id=workspace_id,
        object_type="workspace", object_id=workspace_id,
    )
    return {"status": "ok"}


@app.post("/api/workspaces/{workspace_id}/transfer-ownership")
def transfer_current_workspace(workspace_id: str, request: TransferOwnershipRequest) -> dict:
    try:
        transfer_workspace_ownership(request_scope(workspace_id), request.target_user_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    record_audit_event(
        current_principal().user_id, "workspace.ownership_transferred", workspace_id=workspace_id,
        object_type="user", object_id=request.target_user_id,
    )
    return {"status": "ok", "owner_user_id": request.target_user_id}


def request_scope(workspace_id: str | None = None) -> AccessScope:
    """统一解析请求空间；所有检索和代码项目接口都应先经过这里。"""
    try:
        principal = current_principal()
        return resolve_scope(workspace_id, principal.user_id)
    except PermissionError as exc:
        raise APIError(403, str(exc), "WORKSPACE_ACCESS_DENIED") from exc


def refresh_request_scope(previous: AccessScope) -> AccessScope:
    """为长时间 Agent 请求重新计算授权，成员或会话撤销后立即停止。"""
    try:
        principal = current_principal()
    except AuthenticationError:
        # 仅供直接调用执行器的单元测试；真实 HTTP 始终由认证中间件设置身份。
        return previous
    if not session_is_active(principal):
        raise PermissionError("登录会话已失效")
    return resolve_scope(previous.workspace_id, principal.user_id)


def refresh_scope_for_cached_rag(previous: AccessScope) -> AccessScope:
    """刷新授权并拒绝继续使用部门边界发生变化前缓存的 RAG 证据。"""
    current = refresh_request_scope(previous)
    if current.user_id != previous.user_id or current.workspace_id != previous.workspace_id:
        raise PermissionError("请求身份或工作空间已变化")
    if current.department_id != previous.department_id:
        raise PermissionError("用户部门已变化，请重新发起查询")
    return current


async def refresh_answer_authorization(
    previous: AccessScope,
    last_checked: float,
    *,
    required_project_ids: set[str] | None = None,
    require_same_department: bool = False,
    force: bool = False,
) -> tuple[AccessScope, float]:
    """答案流期间定期续租授权，避免撤权后继续发送后续 Token。"""
    now = perf_counter()
    if not force and now - last_checked < 0.5:
        return previous, last_checked
    current = await asyncio.to_thread(refresh_request_scope, previous)
    if current.user_id != previous.user_id or current.workspace_id != previous.workspace_id:
        raise PermissionError("请求身份或工作空间已变化")
    if require_same_department and current.department_id != previous.department_id:
        raise PermissionError("用户部门已变化，请重新发起查询")
    required = required_project_ids or set()
    if not required.issubset(set(current.allowed_project_ids)):
        raise PermissionError("代码项目授权已在回答期间撤销")
    return current, now


def require_workspace_write(scope: AccessScope) -> None:
    """上传、扫描和共享知识变更只允许 editor 及以上角色。"""
    try:
        require_workspace_role(scope, "owner", "admin", "editor")
    except PermissionError as exc:
        raise APIError(403, str(exc), "INSUFFICIENT_WORKSPACE_ROLE") from exc


@app.get("/api/knowledge")
def knowledge(
    source_type: str | None = Query(default=None),
    scope_type: str | None = Query(default=None),
    workspace_id: str | None = Query(default=None),
) -> dict:
    """返回知识管理页面所需的文档级列表，可按 Wiki/RAG 过滤。"""
    if source_type not in {None, "wiki", "rag"}:
        raise HTTPException(400, "source_type must be wiki or rag")
    if scope_type not in {None, "personal", "department", "workspace"}:
        raise HTTPException(400, "scope_type must be personal, department or workspace")
    scope = request_scope(workspace_id)
    return {
        "items": list_knowledge(
            source_type,
            scope_type,
            scope.user_id,
            scope.department_id,
            scope.workspace_id,
        ),
        "source_type": source_type or "all",
        "workspace_id": scope.workspace_id,
    }


@app.get("/api/knowledge/{document_id}/revisions")
def revisions(document_id: str, workspace_id: str | None = Query(default=None)) -> dict:
    """返回指定文档的修订历史；workspace 校验由数据库查询统一完成。"""
    normalized_id = parse_document_id(document_id)
    scope = request_scope(workspace_id)
    if not can_read_document(normalized_id, scope):
        raise HTTPException(404, "document not found")
    items = list_document_revisions(normalized_id, scope.workspace_id)
    if not items:
        raise HTTPException(404, "document not found or has no revision history")
    return {"document_id": normalized_id, "items": items}


def parse_document_id(document_id: str) -> str:
    """提前校验 UUID，避免把数据库类型错误直接暴露给 API 调用方。"""
    import uuid

    try:
        return str(uuid.UUID(document_id))
    except ValueError as exc:
        raise HTTPException(400, "document_id must be a valid UUID") from exc


@app.post("/api/knowledge/{document_id}/invalidate")
def invalidate(document_id: str, request: InvalidateRequest) -> dict:
    """标记文档失效；文档仍保留在数据库中，但不会再进入检索结果。"""
    normalized_id = parse_document_id(document_id)
    scope = request_scope(request.workspace_id)
    if not can_manage_document(normalized_id, scope):
        raise HTTPException(404, "document not found or not manageable")
    if not invalidate_document(normalized_id, request.reason, scope.workspace_id):
        raise HTTPException(404, "active document not found")
    record_audit_event(
        scope.user_id, "document.invalidated", workspace_id=scope.workspace_id,
        object_type="document", object_id=normalized_id,
        details={"reason_supplied": bool(request.reason)},
    )
    return {"status": "ok", "document_id": normalized_id, "document_status": "invalid"}


@app.post("/api/knowledge/{document_id}/restore")
def restore(document_id: str, workspace_id: str | None = Query(default=None)) -> dict:
    """重新生效文档，让该文档的全部分块恢复参与检索。"""
    normalized_id = parse_document_id(document_id)
    scope = request_scope(workspace_id)
    if not can_manage_document(normalized_id, scope):
        raise HTTPException(404, "document not found or not manageable")
    if not restore_document(normalized_id, scope.workspace_id):
        raise HTTPException(404, "invalid document not found")
    record_audit_event(
        scope.user_id, "document.restored", workspace_id=scope.workspace_id,
        object_type="document", object_id=normalized_id,
    )
    return {"status": "ok", "document_id": normalized_id, "document_status": "active"}


@app.delete("/api/knowledge/{document_id}")
def delete(document_id: str, workspace_id: str | None = Query(default=None)) -> dict:
    """永久删除文档和关联分块。"""
    normalized_id = parse_document_id(document_id)
    scope = request_scope(workspace_id)
    if not can_manage_document(normalized_id, scope):
        raise HTTPException(404, "document not found or not manageable")
    if not delete_document(normalized_id, scope.workspace_id):
        raise HTTPException(404, "document not found")
    record_audit_event(
        scope.user_id, "document.deleted", workspace_id=scope.workspace_id,
        object_type="document", object_id=normalized_id,
    )
    return {"status": "ok", "document_id": normalized_id, "deleted": True}


@app.post("/api/ingest/upload")
async def upload(
    file: UploadFile = File(...),
    # 普通知识上传只负责 RAG 文档；代码项目必须走 Code Wiki 扫描入口。
    source_type: str = Form("rag"),
    scope_type: str = Form("personal"),
    department_id: str | None = Form(None),
    workspace_name: str | None = Form(None),
    workspace_id: str | None = Form(None),
) -> dict:
    """上传文档；标题取文件名，作者和归属由服务端认证身份与空间关系派生。"""
    if source_type != "rag":
        raise HTTPException(400, "普通知识上传仅支持 RAG 文档；代码请使用 Code Wiki 项目扫描")
    scope = request_scope(workspace_id)
    require_workspace_write(scope)
    if scope_type not in {"personal", "department", "workspace"}:
        raise HTTPException(400, "scope_type must be personal, department or workspace")
    if scope_type == "department" and not scope.department_id:
        raise HTTPException(403, "当前用户尚未分配部门，不能导入部门知识")
    # 部门归属必须由登录用户关系派生，不能信任浏览器提交的 department_id。
    department_id = scope.department_id if scope_type == "department" else None
    # workspace_id 是权限键；名称只保存服务端生成的展示快照。
    workspace_name = scope.workspace_name if scope_type == "workspace" else None
    # 每次上传使用独立目录，避免不同用户或空间的同名文件发生并发覆盖。
    upload_root = Path("uploads") / scope.workspace_id / scope.user_id / uuid.uuid4().hex
    target = upload_root / Path(file.filename or "upload.txt").name
    try:
        target.parent.mkdir(parents=True, exist_ok=False)
        total_bytes = 0
        with target.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > 50 * 1024 * 1024:
                    raise HTTPException(413, "单个知识文档不能超过 50 MB")
                output.write(chunk)
        logical_source_path = f"upload://rag/{scope_type}/{target.name}"
        document_id, count = ingest_file(
            target,
            "rag",
            scope_type,
            department_id.strip() if department_id else None,
            None,
            workspace_name.strip() if workspace_name else None,
            scope.workspace_id,
            user_id=scope.user_id,
            display_name=current_principal().display_name,
            source_path=logical_source_path,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("document_ingestion_failed")
        raise HTTPException(500, "文档导入失败，请检查文件格式或服务日志") from exc
    finally:
        # 上传目录只是解析暂存区；内容和向量提交数据库后不再保留原文件副本。
        shutil.rmtree(upload_root, ignore_errors=True)
    record_audit_event(
        scope.user_id, "document.ingested", workspace_id=scope.workspace_id,
        object_type="document", object_id=str(document_id),
        details={"scope_type": scope_type, "chunk_count": count, "source_name": target.name},
    )
    return {"status": "ok", "document_id": document_id, "source": target.name, "chunks": count}


@app.post("/api/code-wiki/scan")
def scan_code_project(request: PathRequest) -> dict:
    """扫描本地项目；Tree-sitter 建立结构，Go SCIP 可选增强语义，全程不调用 LLM。"""
    if not settings.allow_server_path_scan:
        raise HTTPException(403, "服务器路径扫描已关闭，请使用本地项目上传")
    root = Path(request.path).expanduser()
    if not root.is_dir():
        raise HTTPException(404, "project path does not exist or is not a directory")
    try:
        scope = request_scope(request.workspace_id)
        require_workspace_write(scope)
        scan = scan_project(str(root), scope.workspace_id)
        scan.update({"workspace_id": scope.workspace_id, "owner_user_id": scope.user_id, "created_by_user_id": scope.user_id})
        result = persist_scan(scan)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("code_wiki_scan_failed path=%s", root)
        raise HTTPException(500, "代码项目扫描失败，请检查服务日志") from exc
    record_audit_event(
        scope.user_id, "code_project.scanned", workspace_id=scope.workspace_id,
        object_type="code_project", object_id=str(result["project_id"]),
        details={"source_type": "server_path", "file_count": result.get("file_count", 0)},
    )
    return {"status": "ok", **result}


@app.post("/api/code-wiki/import/github")
def import_github_code_project(request: GithubRepositoryRequest) -> dict:
    """通过不可变 Commit 快照导入 GitHub，HTTP 层只负责权限和错误映射。"""
    scope = request_scope(request.workspace_id)
    require_workspace_write(scope)
    try:
        result = import_and_scan_github_repository(
            request.repository_url, scope.workspace_id, scope.user_id,
        )
    except CodeImportValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RepositoryImportBusy as exc:
        raise HTTPException(409, "该仓库正在导入，请稍后重试") from exc
    except RuntimeError as exc:
        logger.exception("code_wiki_github_import_failed")
        raise HTTPException(502, "GitHub 仓库拉取或扫描失败，请检查仓库地址和服务日志") from exc
    except Exception as exc:
        logger.exception("code_wiki_github_import_failed")
        raise HTTPException(500, "代码项目导入失败，请稍后重试") from exc
    record_audit_event(
        scope.user_id, "code_project.imported", workspace_id=scope.workspace_id,
        object_type="code_project", object_id=str(result["project_id"]),
        details={"source_type": "github", "import_action": result["import_action"]},
    )
    return {"status": "ok", **result}


@app.post("/api/code-wiki/import/local")
async def import_local_code_project(files: list[UploadFile] = File(...), workspace_id: str | None = Form(None)) -> dict:
    """接收浏览器选择的完整项目目录，保存托管副本后建立 Code Wiki 索引。"""
    max_files = 10_000
    max_file_bytes = 20 * 1024 * 1024
    max_total_bytes = 300 * 1024 * 1024
    if not files or len(files) > max_files:
        raise HTTPException(422, f"请选择项目目录，文件总数不能超过 {max_files} 个")

    managed_root = (DEFAULT_REPOSITORY_ROOT / "local").resolve()
    managed_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".upload-", dir=managed_root))
    staging_project = staging_root / "repository"
    staging_project.mkdir()
    project_name: str | None = None
    total_bytes = 0
    saved_files = 0
    seen_paths: set[str] = set()

    # 直接调用该函数的单元测试会保留 FastAPI 的 Form 默认对象，此时按默认空间处理。
    scope = request_scope(workspace_id if isinstance(workspace_id, str) else None)
    require_workspace_write(scope)
    try:
        for upload in files:
            current_project, relative_path = normalize_uploaded_path(upload.filename or "")
            if project_name is None:
                project_name = current_project
            elif current_project != project_name:
                raise CodeImportValidationError("一次只能上传一个项目目录")
            if any(part.lower() in IGNORED_DIRS for part in relative_path.parts):
                continue
            relative_key = relative_path.as_posix()
            if relative_key in seen_paths:
                raise CodeImportValidationError(f"项目中存在重复路径：{relative_key}")
            seen_paths.add(relative_key)
            target_file = staging_project.joinpath(*relative_path.parts)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            file_bytes = 0
            with target_file.open("wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    file_bytes += len(chunk)
                    total_bytes += len(chunk)
                    if file_bytes > max_file_bytes:
                        raise CodeImportValidationError(f"单个文件不能超过 20 MB：{relative_key}")
                    if total_bytes > max_total_bytes:
                        raise CodeImportValidationError("项目上传总大小不能超过 300 MB")
                    output.write(chunk)
            saved_files += 1

        if not project_name or saved_files == 0:
            raise CodeImportValidationError("所选目录没有可扫描文件")

        identity_path = managed_local_repository_path(
            project_name, DEFAULT_REPOSITORY_ROOT, scope.workspace_id,
        )
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        repository_key = f"local:{identity_path.name.casefold()}"
        project_id = existing_managed_project_id(scope.workspace_id, repository_key) or managed_code_project_id(
            repository_key, scope.workspace_id,
        )
        with repository_import_lock(
            scope.workspace_id, repository_lock_resource(repository_key, project_id),
        ):
            # 先在 staging 完成扫描，项目的当前版本在 persist_scan 提交前保持不变。
            scan = await asyncio.to_thread(scan_project, str(staging_project), scope.workspace_id)
            commit_hash = str(scan["commit_hash"])
            safe_commit = re.sub(r"[^A-Za-z0-9._-]", "-", commit_hash)[:80]
            snapshot_family = identity_path.parent / ".snapshots" / identity_path.name
            had_snapshot = snapshot_family.exists() and any(snapshot_family.iterdir())
            snapshot_path = snapshot_family / safe_commit
            snapshot_family.mkdir(parents=True, exist_ok=True)
            promoted_here = False
            if snapshot_path.exists():
                shutil.rmtree(staging_project, ignore_errors=True)
            else:
                staging_project.replace(snapshot_path)
                promoted_here = True
            with connection() as conn:
                existed_in_database = conn.execute(
                    "SELECT 1 FROM code_projects WHERE project_id = %s AND workspace_id = %s",
                    (project_id, scope.workspace_id),
                ).fetchone() is not None
            existed = existed_in_database or identity_path.exists() or had_snapshot
            scan.update({
                "project_id": project_id,
                "root_path": str(snapshot_path.resolve()),
                "workspace_id": scope.workspace_id,
                "owner_user_id": scope.user_id,
                "created_by_user_id": scope.user_id,
            })
            scan["project_name"] = project_name
            scan["source"] = {
                "type": "local_upload",
                "project_name": project_name,
                "uploaded_files": saved_files,
                "repository_key": repository_key,
            }
            try:
                result = await asyncio.to_thread(persist_scan, scan)
            except Exception:
                if promoted_here and snapshot_path.exists():
                    shutil.rmtree(snapshot_path, ignore_errors=True)
                raise
        record_audit_event(
            scope.user_id, "code_project.imported", workspace_id=scope.workspace_id,
            object_type="code_project", object_id=str(result["project_id"]),
            details={"source_type": "local_upload", "import_action": "updated" if existed else "uploaded"},
        )
        return {
            "status": "ok",
            "import_action": "updated" if existed else "uploaded",
            "uploaded_files": saved_files,
            **result,
        }
    except CodeImportValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("code_wiki_local_upload_failed project=%s", project_name or "unknown")
        raise HTTPException(500, "本地项目上传或扫描失败，请检查服务日志") from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        for upload in files:
            await upload.close()


@app.get("/api/code-wiki/projects")
def code_projects(workspace_id: str | None = Query(default=None)) -> dict:
    """列出已经进入独立 Code Wiki 事实层的项目。"""
    scope = request_scope(workspace_id)
    return {"items": list_code_projects(scope.workspace_id, scope.user_id), "workspace_id": scope.workspace_id}


@app.delete("/api/code-wiki/projects")
def delete_code_projects_endpoint(request: DeleteCodeProjectsRequest) -> dict:
    """批量删除代码项目；项目 ID 先规范化，避免把任意字符串传入数据库。"""
    if not request.project_ids or len(request.project_ids) > 100:
        raise HTTPException(422, "请选择 1 到 100 个代码项目")
    try:
        project_ids = [str(parse_document_id(project_id)) for project_id in request.project_ids]
        scope = request_scope(request.workspace_id)
        result = delete_code_projects(
            project_ids, scope.workspace_id, scope.user_id, scope.workspace_role,
        )
        record_audit_event(
            scope.user_id, "code_project.deleted", workspace_id=scope.workspace_id,
            object_type="code_project_batch",
            details={
                "requested_count": len(project_ids),
                "deleted_count": len(result.get("deleted", [])),
                "deleted_project_ids": list(result.get("deleted", [])),
            },
        )
        return {"status": "ok", **result}
    except ValueError as exc:
        raise HTTPException(422, "项目 ID 格式不正确") from exc


@app.get("/api/code-wiki/projects/{project_id}/overview")
def code_project_overview(project_id: str, workspace_id: str | None = Query(default=None)) -> dict:
    """返回项目组件证据和文件/符号/关系数量，作为架构理解层的输入。"""
    normalized_id = parse_document_id(project_id)
    scope = request_scope(workspace_id)
    if normalized_id not in scope.allowed_project_ids:
        raise HTTPException(404, "code project not found")
    result = get_code_overview(normalized_id, scope.workspace_id)
    if not result:
        raise HTTPException(404, "code project not found")
    return result


@app.get("/api/code-wiki/projects/{project_id}/files")
def code_project_files(project_id: str, workspace_id: str | None = Query(default=None)) -> dict:
    """返回项目当前 Commit 的文件、语言、行数、符号数和关系数。"""
    normalized_id = parse_document_id(project_id)
    if normalized_id not in request_scope(workspace_id).allowed_project_ids:
        raise HTTPException(404, "code project not found")
    return {"items": list_code_files(normalized_id)}


@app.get("/api/code-wiki/projects/{project_id}/architecture")
def code_project_architecture(project_id: str, workspace_id: str | None = Query(default=None)) -> dict:
    """返回当前 Commit 的组件、消息资源、数据库资源和下游服务证据。"""
    normalized_id = parse_document_id(project_id)
    if normalized_id not in request_scope(workspace_id).allowed_project_ids:
        raise HTTPException(404, "code project not found")
    return {
        "items": list_code_architecture(normalized_id),
        "links": list_code_architecture_links(normalized_id),
    }


@app.get("/api/code-wiki/projects/{project_id}/config-facts")
def code_project_config_facts(project_id: str, q: str = Query(default=""), limit: int = Query(500, ge=1, le=1000), workspace_id: str | None = Query(default=None)) -> dict:
    """返回当前 Commit 的通用配置事实，供页面和 Agent 共同查看。"""
    normalized_id = parse_document_id(project_id)
    if normalized_id not in request_scope(workspace_id).allowed_project_ids:
        raise HTTPException(404, "code project not found")
    return list_code_config_facts(normalized_id, q.strip(), limit)


@app.get("/api/code-wiki/projects/{project_id}/source")
def code_project_source(
    project_id: str,
    path: str = Query(..., min_length=1),
    start_line: int = Query(1, ge=1),
    end_line: int | None = Query(default=None, ge=1),
    workspace_id: str | None = Query(default=None),
) -> dict:
    """读取数据库已登记文件的源码片段，单次最多返回 200 行。"""
    normalized_id = parse_document_id(project_id)
    if normalized_id not in request_scope(workspace_id).allowed_project_ids:
        raise HTTPException(404, "code project not found")
    try:
        result = read_code_source(normalized_id, path, start_line, end_line)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if not result:
        raise HTTPException(404, "code source file not found in current project commit")
    return result


@app.get("/api/code-wiki/symbols")
def code_symbol_search(
    project_id: str = Query(...),
    q: str = Query(default=""),
    file_path: str | None = Query(default=None),
    symbol_kind: str | None = Query(default=None),
    limit: int = Query(100, ge=1, le=500),
    workspace_id: str | None = Query(default=None),
) -> dict:
    """搜索当前 Commit 的符号，也可按文件和符号类型筛选。"""
    normalized_id = parse_document_id(project_id)
    if normalized_id not in request_scope(workspace_id).allowed_project_ids:
        raise HTTPException(404, "code project not found")
    return {"items": search_code_symbols(
        normalized_id,
        q.strip(),
        limit,
        file_path.strip() if file_path else None,
        symbol_kind.strip() if symbol_kind else None,
    )}


@app.get("/api/code-wiki/symbols/{symbol_id}")
def code_symbol_detail(symbol_id: str, workspace_id: str | None = Query(default=None)) -> dict:
    """返回符号定位、出站关系以及引用/实现该符号的入站关系。"""
    normalized_id = parse_document_id(symbol_id)
    result = get_code_symbol(normalized_id)
    if not result:
        raise HTTPException(404, "code symbol not found")
    if str(result.get("project_id")) not in request_scope(workspace_id).allowed_project_ids:
        raise HTTPException(404, "code symbol not found")
    return result


@app.get("/api/code-wiki/symbols/{symbol_id}/call-chain")
def code_symbol_call_chain(
    symbol_id: str,
    max_depth: int = Query(4, ge=1, le=8),
    max_nodes: int = Query(80, ge=2, le=200),
    workspace_id: str | None = Query(default=None),
) -> dict:
    """沿已解析调用边返回有界调用图、未解析边、环路和节点架构证据。"""
    normalized_id = parse_document_id(symbol_id)
    symbol = get_code_symbol(normalized_id)
    if not symbol or str(symbol.get("project_id")) not in request_scope(workspace_id).allowed_project_ids:
        raise HTTPException(404, "code symbol not found")
    result = trace_code_call_chain(normalized_id, max_depth, max_nodes)
    if not result:
        raise HTTPException(404, "code symbol not found in current project commit")
    return result


async def _code_wiki_evidence_stream(request: CodeAgentRequest) -> StreamingResponse:
    """内部 Code Wiki 取证器；公共回答必须再经过顶层 Answer Gate。"""
    if not request.message.strip():
        raise HTTPException(400, "message is required")
    project_id = parse_document_id(request.project_id)
    scope = request_scope(request.workspace_id)
    if project_id not in scope.allowed_project_ids:
        raise HTTPException(404, "code project not found in current workspace")
    try:
        messages, project = build_code_agent_messages(project_id, request.message.strip(), scope.workspace_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    async def event_generator():
        active_scope = scope
        citations = CitationRegistry()
        started_at = perf_counter()
        # 请求字段供顶层控制器下发“更小的剩余预算”，不能借此突破服务端上限。
        max_rounds = min(request.max_rounds or settings.code_agent_max_rounds, settings.code_agent_max_rounds)
        max_tool_calls = min(request.max_tool_calls or settings.code_agent_max_tool_calls, settings.code_agent_max_tool_calls)
        loop_state = AgentLoopState(max_rounds=max_rounds, max_tool_calls=max_tool_calls)
        tool_labels = {"list_project_files": "查看项目文件资产", "search_symbols": "搜索符号", "get_symbol": "读取符号详情", "read_source": "读取源码或配置", "trace_call_path": "追踪调用链", "find_architecture": "查询架构事实", "find_config_facts": "查询通用配置事实"}

        async def stream_model_call(event_id: str, role: str, phase: str, model_messages: list[dict], tools: list[dict] | None, holder: dict):
            """把模型真实文本增量转发给前端，并保留完整工具调用供 Loop Engine 执行。"""
            visible = phase != "decision"
            if visible:
                yield _sse_event("model", {"id": event_id, "role": role, "phase": phase, "status": "started"})
            iterator = iter(stream_code_agent_completion(model_messages, tools))
            message = None
            while True:
                item = await asyncio.to_thread(_next_stream_item, iterator)
                if item is None:
                    break
                if item["type"] == "delta":
                    if visible:
                        yield _sse_event("model", {
                            "id": event_id, "role": role, "phase": phase, "status": "streaming", "delta": item["content"],
                        })
                elif item["type"] == "done":
                    message = item["message"]
            if message is None:
                raise RuntimeError("代码 Agent 模型流未正常结束")
            if visible:
                yield _sse_event("model", {"id": event_id, "role": role, "phase": phase, "status": "completed"})
            holder["message"] = message

        def result_summary(name: str, result: dict) -> str:
            """把工具结果转为用户可理解的事实摘要，而不是展示原始 JSON。"""
            if result.get("error"):
                return f"{tool_labels.get(name, name)}没有得到结果：{result['error']}"
            if name == "list_project_files":
                coverage = result.get("coverage", {})
                return f"发现 {coverage.get('total', 0)} 个文件：已解析 {coverage.get('parsed', 0)} 个、部分解析 {coverage.get('partial', 0)} 个、未分类 {coverage.get('unclassified', 0)} 个"
            if name == "search_symbols":
                items = result.get("items", [])
                names = "、".join(item.get("name", "") for item in items[:3])
                return f"搜索到 {len(items)} 个符号" + (f"，包括 {names}" if names else "")
            if name == "get_symbol":
                return f"已定位 {result.get('qualified_name', result.get('name', '目标符号'))}，读取到 {len(result.get('outgoing', []))} 条出站关系和 {len(result.get('incoming', []))} 条入站关系"
            if name == "read_source":
                return f"已读取 {result.get('path', '源码')} 第 {result.get('start_line', '?')}-{result.get('end_line', '?')} 行"
            if name == "trace_call_path":
                return f"调用链得到 {len(result.get('nodes', []))} 个节点、{len(result.get('edges', []))} 条已解析调用，另有 {len(result.get('unresolved_edges', []))} 条未解析边"
            if name == "find_architecture":
                return f"找到 {len(result.get('facts', []))} 条架构事实和 {len(result.get('links', []))} 条配置调用关联"
            if name == "find_config_facts":
                return f"找到 {result.get('count', 0)} 条配置事实"
            return "工具已返回结果"
        yield _sse_event("step", {
            "step": "project_scope", "status": "completed",
            "message": f"已锁定项目 {project['project_name']}",
            "metrics": {"commit": project["commit"][:12]},
        })
        yield _sse_event("analysis", {
            "id": "analysis_scope", "status": "completed", "done": True,
            "message": "我已经锁定项目和当前 Commit，后续只在这个版本的证据范围内回答",
        })
        try:
            answer_text = ""
            investigation_evidence_sufficient = None
            evidence_payloads: list[dict] = []
            round_index = 0
            while True:
                active_scope = refresh_request_scope(active_scope)
                if project_id not in active_scope.allowed_project_ids:
                    raise PermissionError("代码项目授权已在执行期间撤销")
                yield _sse_event("step", {
                    "step": f"agent_round_{round_index + 1}", "status": "started",
                    "message": "我会根据刚拿到的证据，决定下一步查什么",
                })
                if loop_state.tool_calls == 0:
                    round_analysis = "我先检查当前项目的文件资产，确认后续应该查询源码、配置还是架构关系"
                elif loop_state.unchanged_rounds:
                    round_analysis = "上一轮没有产生新的有效证据，我需要改变查询策略；如果仍然没有进展就停止探索"
                else:
                    round_analysis = "我已经获得新的证据，先判断它是否覆盖问题；如果还有缺口，只补充最小范围的信息"
                for character in round_analysis:
                    yield _sse_event("analysis", {
                        "id": f"analysis_round_{round_index + 1}",
                        "status": "started", "done": False, "delta": character,
                    })
                    await asyncio.sleep(0.012)
                yield _sse_event("analysis", {
                    "id": f"analysis_round_{round_index + 1}",
                    "status": "completed", "done": True, "message": round_analysis,
                })
                model_event_id = f"code_agent_round_{round_index + 1}"
                model_holder = {}
                async for event in stream_model_call(
                    model_event_id,
                    f"代码 Agent · 第 {round_index + 1} 轮",
                    "decision",
                    messages,
                    CODE_AGENT_TOOLS,
                    model_holder,
                ):
                    yield event
                message = model_holder["message"]
                yield _sse_event("model", {
                    "id": model_event_id,
                    "role": f"代码 Agent · 第 {round_index + 1} 轮",
                    "phase": "investigation",
                    "status": "started",
                })
                note_iterator = iter(stream_code_investigation_note(
                    request.message.strip(),
                    project["project_name"],
                    project["commit"],
                    message.tool_calls,
                    messages,
                ))
                while True:
                    note_delta = await asyncio.to_thread(_next_stream_item, note_iterator)
                    if note_delta is None:
                        break
                    yield _sse_event("model", {
                        "id": model_event_id,
                        "role": f"代码 Agent · 第 {round_index + 1} 轮",
                        "phase": "investigation",
                        "status": "streaming",
                        "delta": note_delta,
                    })
                yield _sse_event("model", {
                    "id": model_event_id,
                    "role": f"代码 Agent · 第 {round_index + 1} 轮",
                    "phase": "investigation",
                    "status": "completed",
                })
                messages.append(assistant_message_dict(message))
                if not message.tool_calls:
                    # 兼容未调用 finish_investigation 的模型：无工具响应只代表调查结束，最终答案仍走独立阶段。
                    transition = loop_state.finish("agent_completed", round_index)
                    yield _sse_event("step", {
                        "step": f"agent_round_{round_index + 1}", "status": "completed",
                        "message": "Agent 已完成证据整理",
                        "metrics": {"tool_calls": transition.tool_calls, "citations": len(citations.items)},
                    })
                    break

                finish_call = next(
                    (call for call in message.tool_calls if call.function.name == "finish_investigation"),
                    None,
                )
                if finish_call is not None:
                    try:
                        finish_arguments = json.loads(finish_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        finish_arguments = {}
                    investigation_evidence_sufficient = bool(finish_arguments.get("evidence_sufficient", False))
                    for pending_call in message.tool_calls:
                        if pending_call.id == finish_call.id:
                            control_result = {
                                "status": "accepted",
                                "reason": str(finish_arguments.get("reason") or "Agent 已结束调查")[:500],
                                "evidence_sufficient": bool(finish_arguments.get("evidence_sufficient", False)),
                            }
                        else:
                            control_result = {
                                "status": "skipped",
                                "reason": "finish_investigation was selected in the same response",
                            }
                        messages.append(tool_result_message(
                            pending_call.id,
                            pending_call.function.name,
                            control_result,
                        ))
                    transition = loop_state.finish("agent_completed", round_index)
                    yield _sse_event("step", {
                        "step": f"agent_round_{round_index + 1}", "status": "completed",
                        "message": "Agent 已自主结束调查，开始组织最终回答",
                        "metrics": {"tool_calls": transition.tool_calls, "citations": len(citations.items)},
                    })
                    break

                yield _sse_event("step", {
                    "step": f"agent_round_{round_index + 1}", "status": "completed",
                    "message": f"Agent 选择了 {len(message.tool_calls)} 个受限工具",
                    "metrics": {"round": round_index + 1},
                })
                executed_calls = 0
                blocked_reasons: list[str] = []
                for call in message.tool_calls:
                    name = call.function.name
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    allowed, blocked_reason = loop_state.admit(name, arguments)
                    if not allowed:
                        result = {"error": f"tool call blocked by loop guard: {blocked_reason}", "stop_reason": blocked_reason}
                        blocked_reasons.append(blocked_reason or "tool_call_blocked")
                    else:
                        executed_calls += 1
                        analysis_id = f"analysis_tool_{loop_state.tool_calls}"
                        analysis_text = analysis_message_for_tool(name, arguments)
                        for character in analysis_text:
                            yield _sse_event("analysis", {
                                "id": analysis_id, "status": "started", "done": False,
                                "delta": character, "next_tool": name,
                            })
                            await asyncio.sleep(0.012)
                        yield _sse_event("analysis", {
                            "id": analysis_id, "status": "completed", "done": True,
                            "message": analysis_text, "next_tool": name,
                        })
                        yield _sse_event("tool", {
                            "tool": name, "status": "started",
                            "message": f"我调用{tool_labels.get(name, name)}，确认相关代码证据",
                        })
                        result = await asyncio.to_thread(
                            execute_code_agent_tool, project_id, name, arguments, citations,
                            expected_commit=project["commit"],
                        )
                        evidence_payloads.append({"capability": name, "arguments": arguments, "result": result})
                        result_count = len(result.get("items", result.get("nodes", result.get("facts", [])))) if isinstance(result, dict) else 0
                        yield _sse_event("tool", {
                            "tool": name, "status": "completed",
                            "message": f"调用{tool_labels.get(name, name)}得到：{result_summary(name, result)}",
                            "metrics": {"result_count": result_count, "citations": len(citations.items)},
                        })
                        loop_state.record(name, arguments, result)
                        result_analysis_id = f"analysis_result_{loop_state.tool_calls}"
                        result_analysis = analysis_result_message(name, result)
                        for character in result_analysis:
                            yield _sse_event("analysis", {
                                "id": result_analysis_id, "status": "started", "done": False,
                                "delta": character,
                            })
                            await asyncio.sleep(0.012)
                        yield _sse_event("analysis", {
                            "id": result_analysis_id, "status": "completed", "done": True,
                            "message": result_analysis,
                        })
                    messages.append(tool_result_message(call.id, name, result))
                # Engine 统一处理重复请求、无进展、工具预算和轮次预算；主循环只消费状态转移。
                transition = loop_state.evaluate_round(round_index, executed_calls, blocked_reasons)
                if not transition.should_continue:
                    yield _sse_event("step", {
                        "step": f"agent_round_{round_index + 1}", "status": "completed",
                        "message": f"Loop Engine 已停止：{transition.reason}",
                        "metrics": {"tool_calls": transition.tool_calls, "unchanged_rounds": transition.unchanged_rounds},
                    })
                    break
                round_index += 1

            if not request.evidence_only:
                # 独立 Code Wiki 页面仍自行生成答案；顶层 Agent 调用时只返回证据，避免重复生成。
                active_scope = refresh_request_scope(active_scope)
                if project_id not in active_scope.allowed_project_ids:
                    raise PermissionError("代码项目授权已在回答前撤销")
                if not investigation_evidence_sufficient or not citations.items:
                    answer_text = "当前代码调查没有形成足够的可定位证据，无法可靠回答该问题。"
                else:
                    messages.append({"role": "system", "content": f"调查阶段已经结束，原因是 {loop_state.stop_reason or 'agent_completed'}。现在仅根据已有证据生成最终回答，不得继续调用工具，并明确未确认部分。"})
                    final_holder = {}
                    answer_auth_checked_at = perf_counter()
                    async for event in stream_model_call(
                        "code_agent_final", "代码 Agent · 最终回答", "answer", messages, None, final_holder,
                    ):
                        active_scope, answer_auth_checked_at = await refresh_answer_authorization(
                            active_scope,
                            answer_auth_checked_at,
                            required_project_ids={project_id},
                        )
                        yield event
                    active_scope, answer_auth_checked_at = await refresh_answer_authorization(
                        active_scope,
                        answer_auth_checked_at,
                        required_project_ids={project_id},
                        force=True,
                    )
                    answer_text = final_holder["message"].content or "证据不足，无法生成代码结论。"

            total_ms = round((perf_counter() - started_at) * 1000, 2)
            yield _sse_event("done", {
                "route": "code_wiki_agent",
                "project": project,
                "answer": answer_text,
                "answer_html": render_answer_markdown(answer_text),
                "sources": citations.items if investigation_evidence_sufficient else [],
                "evidence_payloads": evidence_payloads if request.evidence_only else [],
                "observability": {"round_limit": max_rounds, "completed_rounds": loop_state.completed_rounds, "tool_call_limit": max_tool_calls, "tool_calls": loop_state.tool_calls, "citations": len(citations.items), "evidence_sufficient": investigation_evidence_sufficient, "stop_reason": loop_state.stop_reason, "unchanged_rounds": loop_state.unchanged_rounds, "total_ms": total_ms},
            })
        except PermissionError:
            logger.info("code_wiki_agent_authorization_revoked project_id=%s", project_id)
            yield _sse_event("error", {
                "message": "登录或项目权限已在执行期间发生变化，请重新发起查询",
                "error_code": "AUTHORIZATION_REVOKED", "retryable": False,
            })
        except Exception as exc:
            logger.exception("code_wiki_agent_failed project_id=%s", project_id)
            yield _sse_event("error", {
                "message": "代码 Wiki Agent 执行失败，请稍后重试",
                "error_code": "CODE_AGENT_FAILED", "retryable": True,
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def route_question(message: str) -> str:
    """MVP 初版路由器：用规则判断 Wiki、RAG 或 Hybrid，后续可替换成模型路由。"""
    lower = message.lower()
    hybrid_terms = ("是否符合", "规范", "为什么", "问题", "架构", "符合")
    wiki_terms = ("服务", "项目", "代码", "topic", "框架", "依赖", "调用")
    rag_terms = (
        "制度", "流程", "标准", "规定", "sop", "文档", "权限", "mvp",
        "上传", "表格", "评测", "切分", "知识库", "content_hash", "embedding",
    )
    if any(term in message for term in hybrid_terms) and any(term in lower for term in ("服务", "项目", "代码", "kafka", "redis")):
        return "hybrid"
    if any(term in lower for term in rag_terms):
        return "rag"
    if any(term in lower for term in wiki_terms):
        return "wiki"
    return "hybrid"


def _legacy_chat(request: ChatRequest) -> dict:
    """保留旧实现用于回归对照；它不再注册为公共回答接口。"""
    if not request.message.strip():
        raise HTTPException(400, "message is required")
    scope = request_scope(request.workspace_id)
    started_at = perf_counter()
    stage_started = started_at
    stage = "route"
    route = route_question(request.message)
    source_types = {"wiki": ["wiki"], "rag": ["rag"], "hybrid": ["wiki", "rag"]}[route]
    telemetry = {
        "route": route,
        "source_types": source_types,
        "candidate_count": 0,
        "rerank_input_count": 0,
        "rerank_output_count": 0,
        "reranker_input_tokens": 0,
        "reranker_fallback": False,
        "reranker_fallback_reason": None,
        "context_count": 0,
        "parent_context_count": 0,
        "context_tokens": 0,
        "embedding_tokens": 0,
        "chat_input_tokens": 0,
        "chat_output_tokens": 0,
        "estimated_cost": 0.0,
        "stage_ms": {},
    }
    try:
        # 先扩大向量召回范围，再交给 Reranker 精排，避免只在很小的候选集里排序。
        stage = "embedding"
        stage_started = perf_counter()
        query_embedding, embedding_usage = embed(request.message, return_usage=True)
        telemetry["embedding_tokens"] = embedding_usage.get("prompt_tokens", 0)
        telemetry["stage_ms"]["embedding"] = round((perf_counter() - stage_started) * 1000, 2)

        stage = "hybrid_retrieval"
        stage_started = perf_counter()
        candidates = hybrid_search(
            query_embedding,
            request.message,
            source_types,
            settings.reranker_candidate_limit,
            scope.user_id,
            scope.department_id,
            scope.workspace_id,
        )
        telemetry["candidate_count"] = len(candidates)
        telemetry["reranker_input_tokens"] = sum(max(1, len(str(item.get("content", ""))) // 2) for item in candidates)
        telemetry["stage_ms"]["hybrid_retrieval"] = round((perf_counter() - stage_started) * 1000, 2)

        stage = "reranker"
        stage_started = perf_counter()
        telemetry["rerank_input_count"] = len(candidates)
        try:
            results = rerank(request.message, candidates, settings.max_context_chunks)
            retrieval_mode = "vector+rerank"
        except Exception as exc:
            # Reranker 是精排增强层，短暂不可用时保留向量召回，避免整个问答服务不可用。
            results = candidates[: settings.max_context_chunks]
            retrieval_mode = "vector_fallback"
            telemetry["reranker_fallback"] = True
            telemetry["reranker_fallback_reason"] = type(exc).__name__
        telemetry["rerank_output_count"] = len(results)
        telemetry["stage_ms"]["reranker"] = round((perf_counter() - stage_started) * 1000, 2)
        if not results:
            telemetry["total_ms"] = round((perf_counter() - started_at) * 1000, 2)
            logger.info("chat_observability %s", json.dumps(telemetry, ensure_ascii=False))
            return {"route": route, "retrieval_mode": retrieval_mode, "answer": "当前知识库没有检索到相关内容，请先导入项目或文档。", "sources": [], "observability": telemetry}
        # 命中子块后扩展父块；若父块超出预算则保留命中的子块，避免上下文失控。
        stage = "context_builder"
        stage_started = perf_counter()
        context_results = expand_context(
            results,
            settings.max_context_tokens,
            scope.workspace_id,
            scope.user_id,
            scope.department_id,
        )
        if not context_results:
            context_results = results
        telemetry["context_count"] = len(context_results)
        telemetry["parent_context_count"] = sum(item.get("context_level") == "parent" for item in context_results)
        telemetry["context_tokens"] = sum(item.get("context_tokens", 0) for item in context_results)
        telemetry["stage_ms"]["context_builder"] = round((perf_counter() - stage_started) * 1000, 2)
        retrieval_mode = f"{retrieval_mode}+parent-context"
        # 每段证据带有编号和来源引用，模型回答时可以据此回指原始知识。
        evidence = "\n\n".join(
            f"[{index + 1}] 类型={item['source_type']} 来源={item['source_ref']}\n{item['content']}"
            for index, item in enumerate(context_results)
        )
        stage = "chat_llm"
        stage_started = perf_counter()
        response, chat_usage = answer(request.message, evidence, return_usage=True)
        telemetry["chat_input_tokens"] = chat_usage.get("prompt_tokens", 0)
        telemetry["chat_output_tokens"] = chat_usage.get("completion_tokens", 0)
        telemetry["stage_ms"]["chat_llm"] = round((perf_counter() - stage_started) * 1000, 2)
        telemetry["estimated_cost"] = round(
            _cost(telemetry["embedding_tokens"], settings.embedding_input_cost_per_1k_tokens)
            + _cost(telemetry["chat_input_tokens"], settings.chat_input_cost_per_1k_tokens)
            + _cost(telemetry["chat_output_tokens"], settings.chat_output_cost_per_1k_tokens)
            + _cost(telemetry["reranker_input_tokens"], settings.reranker_cost_per_1k_tokens),
            8,
        )
    except Exception as exc:
        telemetry["failed_stage"] = stage
        telemetry["error_type"] = type(exc).__name__
        telemetry["total_ms"] = round((perf_counter() - started_at) * 1000, 2)
        logger.exception("chat_observability_failed %s", json.dumps(telemetry, ensure_ascii=False))
        raise HTTPException(502, "知识查询服务暂时不可用，请稍后重试") from exc
    telemetry["total_ms"] = round((perf_counter() - started_at) * 1000, 2)
    logger.info("chat_observability %s", json.dumps(telemetry, ensure_ascii=False))
    sources = [
        {
            "id": index + 1,
            "document_id": str(item["document_id"]),
            "type": item["source_type"],
            "name": item["source_name"],
            "ref": item["source_ref"],
            "score": round(float(item.get("score", item.get("vector_score", 0.0))), 4),
            "vector_score": round(float(item.get("vector_score", item.get("score", 0.0))), 4),
            "keyword_score": round(float(item.get("keyword_score", 0.0)), 4),
            "fused_score": round(float(item.get("fused_score", 0.0)), 6),
            "rerank_score": round(float(item.get("rerank_score", 0.0)), 4),
            "context_level": item.get("context_level", "child"),
            "matched_ref": item.get("matched_ref", item.get("source_ref")),
        }
        for index, item in enumerate(context_results)
    ]
    return {"route": route, "retrieval_mode": retrieval_mode, "context_tokens": telemetry["context_tokens"], "answer": response, "answer_html": render_answer_markdown(response), "sources": sources, "observability": telemetry}


async def _legacy_chat_stream(request: ChatRequest) -> StreamingResponse:
    """保留旧实现用于回归对照；它不再注册为公共回答接口。"""
    if not request.message.strip():
        raise HTTPException(400, "message is required")
    scope = request_scope(request.workspace_id)

    async def event_generator():
        active_scope = scope
        started_at = perf_counter()
        route = route_question(request.message)
        source_types = {"wiki": ["wiki"], "rag": ["rag"], "hybrid": ["wiki", "rag"]}[route]
        telemetry = {
            "route": route,
            "source_types": source_types,
            "candidate_count": 0,
            "rerank_input_count": 0,
            "rerank_output_count": 0,
            "reranker_input_tokens": 0,
            "reranker_fallback": False,
            "reranker_fallback_reason": None,
            "context_count": 0,
            "parent_context_count": 0,
            "context_tokens": 0,
            "embedding_tokens": 0,
            "chat_input_tokens": 0,
            "chat_output_tokens": 0,
            "estimated_cost": 0.0,
            "replan_count": 0,
            "planner_fallback": False,
            "evidence_grader_fallback": False,
            "evidence_decision": None,
            "stage_ms": {},
        }
        agent_state = KnowledgeAgentState(request.message)
        try:
            # 当前接口是文档检索入口，因此 Planner 只能选择 RAG；跨源 route 将由统一入口开放。
            initial_decision = await asyncio.to_thread(plan_knowledge_query, request.message, {"rag"})
        except Exception:
            telemetry["planner_fallback"] = True
            initial_decision = KnowledgeDecision(
                next_action="retrieve_rag", route="rag", intent="document_question",
                sub_questions=(request.message.strip(),), reason="planner_fallback",
            )
        agent_state.apply_decision(initial_decision)
        route = agent_state.route or "rag"
        source_types = ["rag"]
        telemetry["route"] = route
        telemetry["source_types"] = source_types
        agent_state.plan.append({"objective": "先执行原问题检索；证据不足时根据 Evidence Grader 重规划"})
        plan_state = AdaptiveRetrievalState(request.message)
        current_plan = plan_state.create()

        async def step_started(step: str, message: str):
            yield _sse_event("step", {"step": step, "status": "started", "message": message})

        async def step_completed(step: str, message: str, metrics: dict | None = None):
            payload = {"step": step, "status": "completed", "message": message}
            if metrics:
                payload["metrics"] = metrics
            yield _sse_event("step", payload)

        yield _sse_event("step", {"step": "route", "status": "completed", "message": f"已选择 {route} 检索", "metrics": {"route": route}})
        yield _sse_event("step", {"step": "plan", "status": "completed", "message": "已建立本次问题的检索计划：先检索原问题，证据不足时再改变查询表达", "metrics": {"plan_steps": len(plan_state.steps)}})
        question_summary = " ".join(request.message.split())[:160]
        for event in _public_agent_trace(
            "rag_intent", "RAG Agent · 问题理解",
            f"用户问的是「{question_summary}」。我先在 {route} 范围内用原问题执行混合检索；如果没有候选，再改写查询。",
        ):
            yield event
        try:
            stage = "embedding"
            yield _sse_event("step", {"step": "embedding", "status": "started", "message": f"正在为计划步骤生成查询向量：{current_plan.query}"})
            stage_started = perf_counter()
            query_embedding, embedding_usage = await asyncio.to_thread(embed, current_plan.query, True)
            telemetry["embedding_tokens"] += embedding_usage.get("prompt_tokens", 0)
            telemetry["stage_ms"]["embedding"] = round((perf_counter() - stage_started) * 1000, 2)
            yield _sse_event("step", {"step": "embedding", "status": "completed", "message": "查询向量已生成", "metrics": {"tokens": telemetry["embedding_tokens"], "duration_ms": telemetry["stage_ms"]["embedding"]}})

            stage = "hybrid_retrieval"
            yield _sse_event("step", {"step": "hybrid_retrieval", "status": "started", "message": "正在执行向量和关键词混合检索"})
            stage_started = perf_counter()
            candidates = await asyncio.to_thread(
                hybrid_search,
                query_embedding,
                current_plan.query,
                source_types,
                settings.reranker_candidate_limit,
                active_scope.user_id,
                active_scope.department_id,
                active_scope.workspace_id,
            )
            agent_state.add_action("retrieve_rag", query=current_plan.query, candidate_count=len(candidates))
            telemetry["candidate_count"] = len(candidates)
            telemetry["reranker_input_tokens"] = sum(max(1, len(str(item.get("content", ""))) // 2) for item in candidates)
            telemetry["stage_ms"]["hybrid_retrieval"] = round((perf_counter() - stage_started) * 1000, 2)
            yield _sse_event("step", {"step": "hybrid_retrieval", "status": "completed", "message": f"已召回 {len(candidates)} 个候选", "metrics": {"candidate_count": len(candidates), "duration_ms": telemetry["stage_ms"]["hybrid_retrieval"]}})
            plan_state.complete(current_plan, len(candidates), "首轮检索完成")
            if candidates:
                agent_state.add_evidence(candidates)
            yield _sse_event("step", {"step": "plan_execute", "status": "completed", "message": "首轮检索已完成，正在检查证据是否足够", "metrics": {"plan_steps": len(plan_state.steps), "candidate_count": len(candidates)}})
            retrieval_next = "下一步对候选进行相关性精排。" if candidates else "当前没有候选，下一步调用查询规划模型改写检索表达。"
            for event in _public_agent_trace(
                "rag_retrieval_1", "RAG Agent · 首轮检索",
                f"我用「{current_plan.query[:120]}」查询了向量索引和关键词索引，融合后得到 {len(candidates)} 个候选。{retrieval_next}",
            ):
                yield event

            # 只有首轮没有候选时才重规划，避免无意义地增加模型 API 和数据库负担。
            if not candidates:
                yield _sse_event("model", {"id": "rag_planner", "role": "RAG Agent · 查询规划模型", "phase": "investigation", "status": "started"})
                try:
                    rewritten_queries = await asyncio.to_thread(
                        rewrite_rag_queries, request.message, current_plan.query, "no_candidates",
                    )
                except Exception:
                    rewritten_queries = []
                    telemetry["planner_fallback"] = True
                planner_message = (
                    f"首轮查询没有命中。模型给出的候选查询是：{'；'.join(rewritten_queries)}。我将选择一个未执行过的查询重试。"
                    if rewritten_queries else
                    "首轮查询没有命中，查询规划模型未返回有效改写。我将使用本地通用改写策略继续。"
                )
                for offset in range(0, len(planner_message), 4):
                    yield _sse_event("model", {
                        "id": "rag_planner", "role": "RAG Agent · 查询规划模型", "phase": "investigation",
                        "status": "streaming", "delta": planner_message[offset:offset + 4],
                    })
                yield _sse_event("model", {"id": "rag_planner", "role": "RAG Agent · 查询规划模型", "phase": "investigation", "status": "completed"})
                next_plan = plan_state.replan_with_queries(rewritten_queries) or plan_state.replan()
                if next_plan:
                    current_plan = next_plan
                    telemetry["replan_count"] = plan_state.replan_count
                    yield _sse_event("step", {"step": "replan", "status": "completed", "message": "首轮没有命中证据，改变查询表达后重新检索", "metrics": {"replan_count": plan_state.replan_count}})
                    stage = "replan_embedding"
                    retry_embedding, retry_usage = await asyncio.to_thread(embed, current_plan.query, True)
                    telemetry["embedding_tokens"] += retry_usage.get("prompt_tokens", 0)
                    retry_candidates = await asyncio.to_thread(
                        hybrid_search, retry_embedding, current_plan.query, source_types,
                        settings.reranker_candidate_limit, active_scope.user_id,
                        active_scope.department_id,
                        active_scope.workspace_id,
                    )
                    candidates = retry_candidates
                    agent_state.add_action("retrieve_rag", query=current_plan.query, candidate_count=len(candidates), replanned=True)
                    if candidates:
                        agent_state.add_evidence(candidates)
                    plan_state.complete(current_plan, len(candidates), "重规划检索完成")
                    telemetry["candidate_count"] = len(candidates)
                    telemetry["reranker_input_tokens"] = sum(max(1, len(str(item.get("content", ""))) // 2) for item in candidates)
                    yield _sse_event("step", {"step": "replan_execute", "status": "completed", "message": f"重规划后召回 {len(candidates)} 个候选", "metrics": {"candidate_count": len(candidates)}})
                    retry_next = "下一步进行相关性精排。" if candidates else "改写后仍没有候选，现有知识不足以回答。"
                    for event in _public_agent_trace(
                        "rag_retrieval_2", "RAG Agent · 重规划检索",
                        f"我改用「{current_plan.query[:120]}」再次执行混合检索，得到 {len(candidates)} 个候选。{retry_next}",
                    ):
                        yield event

            stage = "reranker"
            yield _sse_event("step", {"step": "reranker", "status": "started", "message": "正在进行相关性精排"})
            stage_started = perf_counter()
            telemetry["rerank_input_count"] = len(candidates)
            try:
                results = await asyncio.to_thread(rerank, request.message, candidates, settings.max_context_chunks)
                retrieval_mode = "vector+rerank"
                rerank_message = f"精排完成，保留 {len(results)} 个结果"
            except Exception as exc:
                results = candidates[: settings.max_context_chunks]
                retrieval_mode = "vector_fallback"
                telemetry["reranker_fallback"] = True
                telemetry["reranker_fallback_reason"] = type(exc).__name__
                rerank_message = f"精排不可用，已回退初召回：{type(exc).__name__}"
            telemetry["rerank_output_count"] = len(results)
            telemetry["stage_ms"]["reranker"] = round((perf_counter() - stage_started) * 1000, 2)
            yield _sse_event("step", {"step": "reranker", "status": "completed", "message": rerank_message, "metrics": {"input_count": len(candidates), "output_count": len(results), "fallback": telemetry["reranker_fallback"], "duration_ms": telemetry["stage_ms"]["reranker"]}})
            rerank_detail = (
                f"Reranker 不可用，我已回退使用初召回排序，保留 {len(results)} 条证据。"
                if telemetry["reranker_fallback"] else
                f"我对 {len(candidates)} 个候选执行相关性精排，保留 {len(results)} 条证据。"
            )
            rerank_next = "下一步扩展父块和相邻上下文。" if results else "没有可用证据，本轮将拒绝生成无依据答案。"
            for event in _public_agent_trace(
                "rag_rerank", "RAG Agent · 证据筛选", f"{rerank_detail}{rerank_next}",
            ):
                yield event

            if not results:
                agent_state.mark_refused("no_retrievable_evidence")
                telemetry["total_ms"] = round((perf_counter() - started_at) * 1000, 2)
                response = {"route": route, "retrieval_mode": retrieval_mode, "answer": "当前知识库没有检索到相关内容，请先导入项目或文档。", "answer_html": render_answer_markdown("当前知识库没有检索到相关内容，请先导入项目或文档。"), "sources": [], "observability": telemetry}
                yield _sse_event("done", response)
                return

            # 检索和外部精排可能耗时；构建上下文前重新确认用户仍属于该空间。
            active_scope = refresh_scope_for_cached_rag(active_scope)
            stage = "context_builder"
            yield _sse_event("step", {"step": "context_builder", "status": "started", "message": "正在构建回答上下文"})
            stage_started = perf_counter()
            context_results = await asyncio.to_thread(
                expand_context, results, settings.max_context_tokens,
                active_scope.workspace_id, active_scope.user_id, active_scope.department_id,
            )
            if not context_results:
                context_results = results
            telemetry["context_count"] = len(context_results)
            telemetry["parent_context_count"] = sum(item.get("context_level") == "parent" for item in context_results)
            telemetry["context_tokens"] = sum(item.get("context_tokens", 0) for item in context_results)
            telemetry["stage_ms"]["context_builder"] = round((perf_counter() - stage_started) * 1000, 2)
            retrieval_mode = f"{retrieval_mode}+parent-context"
            yield _sse_event("step", {"step": "context_builder", "status": "completed", "message": f"已构建 {len(context_results)} 段上下文", "metrics": {"parent_count": telemetry["parent_context_count"], "context_tokens": telemetry["context_tokens"], "duration_ms": telemetry["stage_ms"]["context_builder"]}})
            for event in _public_agent_trace(
                "rag_context", "RAG Agent · 上下文构建",
                f"我将精排证据扩展为 {len(context_results)} 段上下文，其中包含 {telemetry['parent_context_count']} 个父块，共约 {telemetry['context_tokens']} tokens。下一步评估这些证据能否直接支撑回答。",
            ):
                yield event

            stage = "evidence_grader"
            yield _sse_event("model", {"id": "rag_evidence_grader", "role": "Knowledge Agent · 证据评估", "phase": "investigation", "status": "started"})
            try:
                evidence_decision = await asyncio.to_thread(
                    grade_rag_evidence,
                    request.message,
                    context_results,
                    plan_state.replan_count < plan_state.max_replans,
                )
            except Exception:
                telemetry["evidence_grader_fallback"] = True
                evidence_decision = KnowledgeDecision(
                    next_action="refuse",
                    reason="evidence_grader_unavailable",
                )
            agent_state.apply_decision(evidence_decision)
            telemetry["evidence_decision"] = evidence_decision.next_action
            grader_message = (
                f"证据评估决定：{evidence_decision.next_action}。"
                f"{('仍缺少：' + '；'.join(evidence_decision.missing_information) + '。') if evidence_decision.missing_information else ''}"
                f"{evidence_decision.reason}"
            )
            for offset in range(0, len(grader_message), 4):
                yield _sse_event("model", {
                    "id": "rag_evidence_grader", "role": "Knowledge Agent · 证据评估", "phase": "investigation",
                    "status": "streaming", "delta": grader_message[offset:offset + 4],
                })
            yield _sse_event("model", {"id": "rag_evidence_grader", "role": "Knowledge Agent · 证据评估", "phase": "investigation", "status": "completed"})

            if evidence_decision.next_action == "replan":
                next_plan = plan_state.replan_with_queries(list(evidence_decision.queries)) or plan_state.replan()
                if next_plan is None:
                    agent_state.apply_decision(KnowledgeDecision(next_action="refuse", reason="replan_exhausted"))
                else:
                    current_plan = next_plan
                    telemetry["replan_count"] = plan_state.replan_count
                    stage = "evidence_replan"
                    retry_embedding, retry_usage = await asyncio.to_thread(embed, current_plan.query, True)
                    telemetry["embedding_tokens"] += retry_usage.get("prompt_tokens", 0)
                    retry_candidates = await asyncio.to_thread(
                        hybrid_search, retry_embedding, current_plan.query, source_types,
                        settings.reranker_candidate_limit, active_scope.user_id,
                        active_scope.department_id,
                        active_scope.workspace_id,
                    )
                    agent_state.add_action("retrieve_rag", query=current_plan.query, candidate_count=len(retry_candidates), evidence_replan=True)
                    try:
                        retry_results = await asyncio.to_thread(
                            rerank, request.message, retry_candidates, settings.max_context_chunks,
                        )
                    except Exception:
                        telemetry["reranker_fallback"] = True
                        retry_results = retry_candidates[:settings.max_context_chunks]
                    retry_context = await asyncio.to_thread(
                        expand_context, retry_results, settings.max_context_tokens,
                        active_scope.workspace_id, active_scope.user_id, active_scope.department_id,
                    )
                    context_results = retry_context or retry_results
                    plan_state.complete(current_plan, len(retry_candidates), "证据评估后重规划检索完成")
                    telemetry["candidate_count"] = len(retry_candidates)
                    telemetry["rerank_input_count"] = len(retry_candidates)
                    telemetry["rerank_output_count"] = len(retry_results)
                    telemetry["context_count"] = len(context_results)
                    telemetry["parent_context_count"] = sum(item.get("context_level") == "parent" for item in context_results)
                    telemetry["context_tokens"] = sum(item.get("context_tokens", 0) for item in context_results)
                    if context_results:
                        agent_state.add_evidence(context_results)
                    for event in _public_agent_trace(
                        "rag_evidence_replan", "Knowledge Agent · 补充检索",
                        f"证据评估发现缺口，我改用「{current_plan.query[:120]}」补充检索，得到 {len(retry_candidates)} 个候选并构建 {len(context_results)} 段上下文。下一步重新评估证据。",
                    ):
                        yield event
                    try:
                        final_decision = await asyncio.to_thread(
                            grade_rag_evidence, request.message, context_results, False,
                        )
                    except Exception:
                        telemetry["evidence_grader_fallback"] = True
                        final_decision = KnowledgeDecision(
                            next_action="refuse",
                            reason="evidence_grader_unavailable_after_replan",
                        )
                    agent_state.apply_decision(final_decision)
                    telemetry["evidence_decision"] = final_decision.next_action

            if agent_state.refused or not agent_state.answer_ready:
                telemetry["total_ms"] = round((perf_counter() - started_at) * 1000, 2)
                refusal = "当前检索到的证据不足以可靠回答这个问题。"
                if agent_state.missing_information:
                    refusal += "仍缺少：" + "；".join(agent_state.missing_information) + "。"
                yield _sse_event("done", {
                    "route": route, "retrieval_mode": retrieval_mode, "answer": refusal,
                    "answer_html": render_answer_markdown(refusal), "sources": [],
                    "observability": {**telemetry, "agent_state": agent_state.snapshot()},
                })
                return

            # 最终模型调用前再次校验，撤权后不得继续向客户端发送业务答案和来源。
            active_scope = refresh_scope_for_cached_rag(active_scope)
            evidence = "\n\n".join(
                f"[{index + 1}] 类型={item['source_type']} 来源={item['source_ref']}\n{item['content']}"
                for index, item in enumerate(context_results)
            )
            stage = "chat_llm"
            yield _sse_event("step", {"step": "chat_llm", "status": "started", "message": "正在基于证据生成回答"})
            yield _sse_event("model", {"id": "rag_answer", "role": "回答模型", "phase": "answer", "status": "started"})
            stage_started = perf_counter()
            response_parts: list[str] = []
            chat_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            answer_auth_checked_at = perf_counter()
            stream = stream_answer(request.message, evidence)
            while True:
                item = await asyncio.to_thread(_next_stream_item, stream)
                if item is None:
                    break
                active_scope, answer_auth_checked_at = await refresh_answer_authorization(
                    active_scope,
                    answer_auth_checked_at,
                    require_same_department=True,
                )
                if item["type"] == "delta":
                    response_parts.append(item["content"])
                    yield _sse_event("model", {"id": "rag_answer", "role": "回答模型", "phase": "answer", "status": "streaming", "delta": item["content"]})
                elif item["type"] == "usage":
                    chat_usage = item
            active_scope, answer_auth_checked_at = await refresh_answer_authorization(
                active_scope,
                answer_auth_checked_at,
                require_same_department=True,
                force=True,
            )
            response_text = "".join(response_parts) or "无法生成回答。"
            yield _sse_event("model", {"id": "rag_answer", "role": "回答模型", "phase": "answer", "status": "completed"})
            telemetry["chat_input_tokens"] = chat_usage.get("prompt_tokens", 0)
            telemetry["chat_output_tokens"] = chat_usage.get("completion_tokens", 0)
            telemetry["stage_ms"]["chat_llm"] = round((perf_counter() - stage_started) * 1000, 2)
            telemetry["estimated_cost"] = round(
                _cost(telemetry["embedding_tokens"], settings.embedding_input_cost_per_1k_tokens)
                + _cost(telemetry["chat_input_tokens"], settings.chat_input_cost_per_1k_tokens)
                + _cost(telemetry["chat_output_tokens"], settings.chat_output_cost_per_1k_tokens)
                + _cost(telemetry["reranker_input_tokens"], settings.reranker_cost_per_1k_tokens),
                8,
            )
            telemetry["total_ms"] = round((perf_counter() - started_at) * 1000, 2)
            yield _sse_event("step", {"step": "chat_llm", "status": "completed", "message": "回答已生成", "metrics": {"output_tokens": telemetry["chat_output_tokens"], "duration_ms": telemetry["stage_ms"]["chat_llm"]}})
            sources = [
                {
                    "id": index + 1,
                    "document_id": str(item["document_id"]),
                    "type": item["source_type"],
                    "name": item["source_name"],
                    "ref": item["source_ref"],
                    "score": round(float(item.get("score", item.get("vector_score", 0.0))), 4),
                    "vector_score": round(float(item.get("vector_score", item.get("score", 0.0))), 4),
                    "keyword_score": round(float(item.get("keyword_score", 0.0)), 4),
                    "fused_score": round(float(item.get("fused_score", 0.0)), 6),
                    "rerank_score": round(float(item.get("rerank_score", 0.0)), 4),
                    "context_level": item.get("context_level", "child"),
                    "matched_ref": item.get("matched_ref", item.get("source_ref")),
                }
                for index, item in enumerate(context_results)
            ]
            yield _sse_event("done", {"route": route, "retrieval_mode": retrieval_mode, "context_tokens": telemetry["context_tokens"], "answer": response_text, "answer_html": render_answer_markdown(response_text), "sources": sources, "observability": {**telemetry, "agent_state": agent_state.snapshot()}})
        except PermissionError:
            logger.info("rag_stream_authorization_revoked")
            yield _sse_event("error", {
                "message": "登录或知识访问范围已在执行期间发生变化，请重新发起查询",
                "error_code": "AUTHORIZATION_REVOKED", "retryable": False,
            })
        except Exception as exc:
            telemetry["failed_stage"] = stage
            telemetry["error_type"] = type(exc).__name__
            telemetry["total_ms"] = round((perf_counter() - started_at) * 1000, 2)
            logger.exception("chat_stream_observability_failed %s", json.dumps(telemetry, ensure_ascii=False))
            yield _sse_event("error", {
                "message": "知识查询服务暂时不可用，请稍后重试",
                "error_code": "RAG_QUERY_FAILED", "retryable": True,
                "observability": {key: value for key, value in telemetry.items() if key != "error_type"},
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _retrieve_rag_capability(query: str, tool_call_id: str, scope: AccessScope | None = None) -> tuple[KnowledgeToolResult, list[dict]]:
    """只获取文档证据，不生成答案；供顶层 Agent 和后续兼容接口复用。"""
    started_at = perf_counter()
    embedding, embedding_usage = await asyncio.to_thread(embed, query, True)
    resolved_scope = scope or request_scope()
    candidates = await asyncio.to_thread(
        hybrid_search, embedding, query, ["rag"], settings.reranker_candidate_limit,
        resolved_scope.user_id, resolved_scope.department_id, resolved_scope.workspace_id,
    )
    reranker_fallback = False
    try:
        ranked = await asyncio.to_thread(rerank, query, candidates, settings.max_context_chunks)
    except Exception:
        ranked = candidates[:settings.max_context_chunks]
        reranker_fallback = True
    context = await asyncio.to_thread(
        expand_context, ranked, settings.max_context_tokens,
        resolved_scope.workspace_id, resolved_scope.user_id, resolved_scope.department_id,
    )
    context = context or ranked
    evidence = tuple(
        KnowledgeEvidence(
            evidence_id=f"{tool_call_id}-E{index + 1}", source_type="rag",
            content_or_fact=str(item.get("content") or ""),
            locator=str(item.get("source_ref") or "") or None,
            document_id=str(item.get("document_id") or "") or None,
            evidence_kind="document_chunk",
            confidence=float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
        )
        for index, item in enumerate(context)
    )
    result = KnowledgeToolResult(
        tool_call_id=tool_call_id, capability="retrieve_rag",
        status="success" if evidence else "empty",
        result={"candidate_count": len(candidates), "context_count": len(context)},
        evidence=evidence,
        warnings=("reranker_fallback",) if reranker_fallback else (),
        usage={
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
            "embedding_tokens": embedding_usage.get("prompt_tokens", 0),
        },
        provenance={"source_system": "postgresql+pgvector"},
    )
    sources = [
        {
            "id": item.evidence_id, "document_id": item.document_id, "type": "rag",
            "name": raw.get("source_name", "RAG 文档"), "ref": item.locator,
            "score": item.confidence, "citation": f"[{item.evidence_id}]",
        }
        for item, raw in zip(evidence, context)
    ]
    return result, sources


def _split_code_payloads_by_citation(payloads: list[dict]) -> dict[str, list[str]]:
    """按事实对象中的引用字段拆分代码证据，避免一个大结果错误绑定给所有引用。"""
    mapped: dict[str, list[str]] = {}

    def visit(value) -> None:
        if isinstance(value, dict):
            citation_ids = set()
            for key, field_value in value.items():
                if "citation" in str(key).casefold():
                    citation_ids.update(re.findall(r"\[(C\d+)\]", str(field_value)))
            if citation_ids:
                compact = json.dumps(value, ensure_ascii=False, default=str)[:4000]
                for citation_id in citation_ids:
                    mapped.setdefault(citation_id, []).append(compact)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for payload in payloads:
        visit(payload.get("result", payload))
    return mapped


async def agentic_knowledge_stream(
    message: str,
    project_id: str | None,
    scope: AccessScope | None = None,
    capability_mode: str = "auto",
) -> StreamingResponse:
    """运行唯一顶层动态循环；显式模式只限制能力，不绕过统一 Answer Gate。"""
    if capability_mode not in {"auto", "rag", "codewiki", "hybrid"}:
        raise ValueError("unsupported knowledge capability mode")

    async def event_generator():
        resolved_scope = scope or request_scope()
        state = KnowledgeAgentState(
            message, project_id=project_id, cross_project_policy="candidate_set",
            allowed_project_ids=set(resolved_scope.allowed_project_ids),
            authorized_projects=list(resolved_scope.authorized_projects),
        )
        if capability_mode == "codewiki":
            state.set_scope("strict_project", "none")
        sources: list[dict] = []
        available_routes: set = (
            {"rag"} if capability_mode == "rag" else
            {"wiki"} if capability_mode == "codewiki" else
            {"hybrid"} if capability_mode == "hybrid" else
            {"rag"}
        )
        if capability_mode == "auto" and (project_id or resolved_scope.allowed_project_ids):
            available_routes.update({"wiki", "hybrid"})
        investigated_project_ids: set[str] = set()

        async def checkpoint_investigation_authorization(
            additional_project_ids: set[str] | None = None,
        ) -> None:
            """公开调查事件发送前复核当前证据仍属于有效授权范围。"""
            nonlocal resolved_scope
            evidence_project_ids = {
                str(item.get("project_id"))
                for item in state.evidence
                if item.get("source_type") == "code_wiki" and item.get("project_id")
            }
            evidence_project_ids.update(investigated_project_ids)
            evidence_project_ids.update(additional_project_ids or set())
            has_rag_evidence = any(item.get("source_type") == "rag" for item in state.evidence)
            resolved_scope, _ = await refresh_answer_authorization(
                resolved_scope,
                0.0,
                required_project_ids=evidence_project_ids,
                require_same_department=has_rag_evidence,
                force=True,
            )
            state.allowed_project_ids.intersection_update(resolved_scope.allowed_project_ids)
            state.authorized_projects = [
                item for item in resolved_scope.authorized_projects
                if item.get("project_id") in state.allowed_project_ids
            ]
            if project_id and project_id not in state.allowed_project_ids:
                raise PermissionError("selected project authorization revoked during investigation")
        try:
            while not state.answer_ready and not state.refused:
                # Agent 可能运行数十秒，每轮重新查询会话、空间成员和项目 ACL，
                # 不能把请求开始时的授权候选集合当作全程有效。
                previous_scope = resolved_scope
                resolved_scope = refresh_request_scope(resolved_scope)
                if resolved_scope.department_id != previous_scope.department_id and any(
                    item.get("source_type") == "rag" for item in state.evidence
                ):
                    state.mark_refused("department_changed_during_rag_investigation")
                    break
                state.allowed_project_ids.intersection_update(resolved_scope.allowed_project_ids)
                state.authorized_projects = [
                    item for item in resolved_scope.authorized_projects
                    if item.get("project_id") in state.allowed_project_ids
                ]
                if project_id and project_id not in state.allowed_project_ids:
                    state.mark_refused("authorization_revoked_during_request")
                    break
                # 旧证据进入下一轮 Planner Prompt 之前必须再次校验其项目/部门授权；
                # 不能只在 Planner 返回后拦截浏览器事件，否则外部模型已看见旧证据。
                await checkpoint_investigation_authorization()
                state.begin_round()
                if state.refused:
                    break

                source_actions = set()
                if capability_mode in {"auto", "rag", "hybrid"}:
                    source_actions.add("retrieve_rag")
                if capability_mode in {"auto", "codewiki", "hybrid"} and (project_id or state.allowed_project_ids):
                    source_actions.add("query_code_wiki")
                if state.scope_mode == "strict_project" or capability_mode == "codewiki":
                    source_actions.discard("retrieve_rag")
                if state.used_tool_calls >= state.max_tool_calls:
                    source_actions.clear()
                # 一次空查询不等于路径耗尽；每类能力允许模型改变查询再补证一次。
                remaining_paths = {action for action in source_actions if state.action_attempt_count(action) < 2}
                allowed_actions = set(remaining_paths) | {"replan"}
                if state.evidence:
                    allowed_actions.add("answer")
                if not remaining_paths:
                    allowed_actions.add("refuse")

                try:
                    decision = await asyncio.to_thread(
                        decide_next_knowledge_action,
                        message,
                        state.snapshot(),
                        allowed_actions,
                        available_routes,
                    )
                except Exception:
                    # Planner 暂时不可用时只执行一个仍合法且未尝试的能力，不扩大项目范围。
                    deterministic_code_target = project_id or (
                        next(iter(state.allowed_project_ids)) if len(state.allowed_project_ids) == 1 else None
                    )
                    fallback_action = "query_code_wiki" if deterministic_code_target and "query_code_wiki" in remaining_paths else (
                        "retrieve_rag" if "retrieve_rag" in remaining_paths else "refuse"
                    )
                    fallback_route = "wiki" if fallback_action == "query_code_wiki" else "rag" if fallback_action == "retrieve_rag" else None
                    decision = KnowledgeDecision(
                        next_action=fallback_action, route=fallback_route,
                        intent="planner_fallback", queries=(message,),
                        missing_information=("顶层决策模型暂时不可用",),
                        reason="top_level_decision_fallback",
                        public_update="顶层决策模型暂时不可用，我只执行当前授权范围内尚未尝试的知识能力。",
                        target_project_id=deterministic_code_target if fallback_action == "query_code_wiki" else None,
                    )
                state.apply_scope_suggestion(decision)
                if capability_mode == "codewiki":
                    state.set_scope("strict_project", "none")
                elif capability_mode == "hybrid":
                    # 显式 Hybrid 的产品契约要求两侧证据；模型可以调整查询，
                    # 但不能通过 strict_project 建议关闭文档检索能力。
                    state.set_scope("soft_project", "rag_allowed")
                decision_target_project_id = decision.target_project_id or project_id
                if decision.next_action == "query_code_wiki":
                    if decision_target_project_id not in state.allowed_project_ids:
                        state.replan(
                            "project_not_in_authorized_candidate_set",
                            ["只能查询当前空间内已授权的代码项目"],
                        )
                        continue
                    investigated_project_ids.add(str(decision_target_project_id))
                update = decision.public_update or decision.reason or f"下一步执行 {decision.next_action}"
                for event in _public_agent_trace(
                    f"knowledge_round_{state.used_rounds}",
                    f"Knowledge Agent · 第 {state.used_rounds} 轮", update,
                ):
                    await checkpoint_investigation_authorization(
                        {str(decision_target_project_id)}
                        if decision.next_action == "query_code_wiki" and decision_target_project_id else None
                    )
                    yield event

                if decision.next_action == "answer":
                    if capability_mode == "hybrid":
                        successful_capabilities = {
                            str(result.get("capability"))
                            for result in state.tool_results if result.get("status") == "success"
                        }
                        if not {"retrieve_rag", "query_code_wiki"}.issubset(successful_capabilities):
                            state.replan("hybrid_evidence_incomplete", ["需要同时获得文档和代码证据"])
                            continue
                    if state.can_answer(decision):
                        state.apply_decision(decision)
                        break
                    state.replan("answer_gate_rejected", ["回答引用未覆盖当前有效证据"])
                    continue
                if decision.next_action == "refuse":
                    if remaining_paths:
                        state.replan("refuse_gate_rejected", ["仍有合法且未尝试的知识能力"])
                        continue
                    state.apply_decision(decision)
                    break
                if decision.next_action == "replan":
                    state.replan(decision.reason or "model_requested_replan", list(decision.missing_information))
                    continue
                if state.scope_mode == "strict_project" and decision.next_action == "retrieve_rag":
                    state.replan("scope_action_mismatch", ["严格项目代码问题不能用文档证据替代代码事实"])
                    continue

                target_project_id = decision_target_project_id
                query = decision.queries[0] if decision.queries else message
                signature = f"{decision.next_action}:{target_project_id or '-'}:{query.strip()[:300]}"
                try:
                    state.admit_action(
                        decision.next_action, signature,
                        # 代码动作的目标项目可以是锚点，也可以是授权候选集合中的其他项目；
                        # 上面的 target_project_id 校验已经替代了“必须有锚点项目”的旧限制。
                        requires_project=False,
                    )
                except ValueError:
                    state.replan("duplicate_action_rejected", ["需要改变查询表达或知识来源"])
                    continue

                if decision.next_action == "retrieve_rag":
                    yield _sse_event("step", {"step": "agentic_rag", "status": "started", "message": "正在检索文档证据"})
                    if scope is None:
                        # 保留旧的二参数适配形态，便于独立测试替换 RAG 子执行器。
                        result, rag_sources = await _retrieve_rag_capability(query, f"rag-{state.used_tool_calls}")
                    else:
                        result, rag_sources = await _retrieve_rag_capability(query, f"rag-{state.used_tool_calls}", resolved_scope)
                    state.record_tool_result(result)
                    sources.extend(rag_sources)
                    await checkpoint_investigation_authorization()
                    yield _sse_event("step", {"step": "agentic_rag", "status": "completed", "message": f"文档检索返回 {len(result.evidence)} 条证据", "metrics": result.usage})
                    continue

                # Code Wiki 作为证据子执行器运行，不生成或发送子层最终答案。
                remaining_tools = state.max_tool_calls - state.used_tool_calls
                # 至少保留一轮给顶层 Observe/Answer Gate，子 Agent 不能耗尽整个请求轮次。
                remaining_rounds = state.max_rounds - state.used_rounds - 1
                if remaining_tools <= 0 or remaining_rounds <= 0:
                    state.mark_refused("code_agent_budget_unavailable")
                    break
                response = await _code_wiki_evidence_stream(CodeAgentRequest(
                    project_id=target_project_id, message=query, evidence_only=True, workspace_id=resolved_scope.workspace_id,
                    max_rounds=remaining_rounds, max_tool_calls=remaining_tools,
                ))
                buffer = ""
                code_done = None
                async for raw in response.body_iterator:
                    buffer += raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    while "\n\n" in buffer:
                        block, buffer = buffer.split("\n\n", 1)
                        event_name = "message"
                        data = None
                        for line in block.splitlines():
                            if line.startswith("event:"):
                                event_name = line[6:].strip()
                            elif line.startswith("data:"):
                                try:
                                    data = json.loads(line[5:].strip())
                                except json.JSONDecodeError:
                                    data = None
                        if event_name == "done" and data:
                            code_done = data
                        elif event_name == "error" and data:
                            _raise_child_agent_error(data, "Code Wiki evidence adapter failed")
                        elif event_name in {"step", "model", "analysis"} and data:
                            # 子 Agent 的 yield 仍只是内存事件；只有顶层重新鉴权后
                            # 才能把源码路径、符号名和调查摘要发送给浏览器。
                            await checkpoint_investigation_authorization({str(target_project_id)})
                            yield _sse_event(event_name, data)
                if not code_done:
                    raise RuntimeError("Code Wiki evidence adapter returned no result")
                observability = code_done.get("observability", {})
                state.record_child_tool_calls(int(observability.get("tool_calls", 0) or 0))
                state.record_child_rounds(int(observability.get("completed_rounds", 0) or 0))
                commit = str(code_done.get("project", {}).get("commit") or "")
                state.commit_id = commit or state.commit_id
                source_by_id = {str(item.get("id")): item for item in code_done.get("sources", []) if item.get("id")}
                code_namespace = f"code-{state.action_attempt_count('query_code_wiki')}"
                payload_by_citation = _split_code_payloads_by_citation(code_done.get("evidence_payloads", []))
                code_evidence_items = []
                remaining_chars = 24000
                for citation_id, payloads in payload_by_citation.items():
                    source = source_by_id.get(citation_id)
                    if not source or not commit or not source.get("path") or remaining_chars <= 0:
                        continue
                    content = re.sub(
                        r"\[(C\d+)\]", lambda match: f"[{code_namespace}-{match.group(1)}]",
                        "\n".join(payloads),
                    )[:min(4000, remaining_chars)]
                    remaining_chars -= len(content)
                    code_evidence_items.append(KnowledgeEvidence(
                        evidence_id=f"{code_namespace}-{citation_id}", source_type="code_wiki", content_or_fact=content,
                        locator=f"{source.get('path')}:{source.get('line', 1)}", project_id=target_project_id,
                        commit_id=commit, evidence_kind="source_citation", derivation="code_wiki_evidence_adapter",
                    ))
                code_evidence = tuple(code_evidence_items)
                status = "success" if code_evidence and observability.get("evidence_sufficient") is True else ("partial" if code_evidence else "empty")
                state.record_tool_result(KnowledgeToolResult(
                    tool_call_id=f"code-{state.used_tool_calls}", capability="query_code_wiki",
                    status=status, result={"commit": commit, "payload_count": len(code_done.get("evidence_payloads", []))},
                    evidence=code_evidence,
                    warnings=("code_investigation_incomplete",) if status != "success" else (),
                    usage={"tool_calls": observability.get("tool_calls", 0), "completed_rounds": observability.get("completed_rounds", 0)},
                    provenance={"source_system": "code_wiki", "commit": commit},
                ))
                sources.extend([
                    {**source, "id": f"{code_namespace}-{source.get('id')}", "document_id": target_project_id,
                     "type": "wiki", "commit": commit,
                     "ref": f"{source.get('path')}:{source.get('line', 1)}",
                     "citation": f"[{code_namespace}-{source.get('id')}]"}
                    for source in code_done.get("sources", [])
                ])

            approved_evidence_ids: set[str] = set()
            approved_claims = ()
            approved_code_project_ids: set[str] = set()
            approved_has_rag = False
            if state.answer_ready:
                gate_claims = state.approved_claims()
                gate_evidence_ids = {
                    evidence_id for claim in gate_claims for evidence_id in claim.evidence_ids
                }
                # Answer Gate 到实际模型调用之间仍可能发生撤权，因此在发送首个答案
                # Token 前最后刷新一次作用域，并校验已经获批的内存证据。
                final_scope = refresh_request_scope(resolved_scope)
                current_project_ids = set(final_scope.allowed_project_ids)
                approved_evidence = [
                    item for item in state.evidence if item.get("evidence_id") in gate_evidence_ids
                ]
                rag_scope_changed = (
                    final_scope.department_id != resolved_scope.department_id
                    and any(item.get("source_type") == "rag" for item in approved_evidence)
                )
                revoked_code_evidence = any(
                    item.get("source_type") == "code_wiki"
                    and item.get("project_id") not in current_project_ids
                    for item in approved_evidence
                )
                approved_code_project_ids = {
                    str(item.get("project_id"))
                    for item in approved_evidence
                    if item.get("source_type") == "code_wiki" and item.get("project_id")
                }
                approved_has_rag = any(item.get("source_type") == "rag" for item in approved_evidence)
                state.allowed_project_ids.intersection_update(current_project_ids)
                if rag_scope_changed or revoked_code_evidence:
                    state.mark_refused("authorization_changed_before_final_answer")
                    answer_text = "回答生成前授权范围发生变化，请重新发起查询。"
                else:
                    resolved_scope = final_scope
                    evidence_text, approved_claims, approved_evidence_ids = state.bounded_answer_material(
                        settings.max_context_tokens * 4
                    )
                if state.answer_ready and not approved_claims:
                    state.mark_refused("approved_evidence_exceeds_context_budget")
                    answer_text = "现有证据超过本次上下文预算，无法在不截断依据的情况下可靠回答。"
                elif state.answer_ready:
                    yield _sse_event("model", {"id": "knowledge_final", "role": "Knowledge Agent · 最终回答", "phase": "answer", "status": "started"})
                    parts = []
                    answer_auth_checked_at = perf_counter()
                    stream = stream_answer(message, evidence_text, [
                        {"text": claim.text, "evidence_ids": list(claim.evidence_ids)}
                        for claim in approved_claims
                    ])
                    while True:
                        item = await asyncio.to_thread(_next_stream_item, stream)
                        if item is None:
                            break
                        resolved_scope, answer_auth_checked_at = await refresh_answer_authorization(
                            resolved_scope,
                            answer_auth_checked_at,
                            required_project_ids=approved_code_project_ids,
                            require_same_department=approved_has_rag,
                        )
                        if item["type"] == "delta":
                            parts.append(item["content"])
                    resolved_scope, answer_auth_checked_at = await refresh_answer_authorization(
                        resolved_scope,
                        answer_auth_checked_at,
                        required_project_ids=approved_code_project_ids,
                        require_same_department=approved_has_rag,
                        force=True,
                    )
                    generated_answer = "".join(parts) or "无法生成回答。"
                    contract_valid, contract_reason = validate_answer_contract(
                        generated_answer,
                        approved_claims,
                        approved_evidence_ids,
                    )
                    answer_text = generated_answer if contract_valid else render_claim_contract_answer(approved_claims)
                    state.add_action(
                        "assess_evidence",
                        stage="final_answer_contract",
                        status="validated" if contract_valid else "repaired",
                        reason=contract_reason,
                    )
                    # 完整答案通过引用契约后才对外发送；避免客户端短暂看到未获批内容。
                    for offset in range(0, len(answer_text), 12):
                        resolved_scope, answer_auth_checked_at = await refresh_answer_authorization(
                            resolved_scope,
                            answer_auth_checked_at,
                            required_project_ids=approved_code_project_ids,
                            require_same_department=approved_has_rag,
                            force=True,
                        )
                        yield _sse_event("model", {
                            "id": "knowledge_final", "role": "Knowledge Agent · 最终回答",
                            "phase": "answer", "status": "streaming", "delta": answer_text[offset:offset + 12],
                        })
                    yield _sse_event("model", {"id": "knowledge_final", "role": "Knowledge Agent · 最终回答", "phase": "answer", "status": "completed"})
            else:
                answer_text = "当前授权范围和已执行能力仍不足以可靠回答该问题。"
                if state.missing_information:
                    answer_text += "仍缺少：" + "；".join(state.missing_information) + "。"
            # done 即使是拒答也会携带状态摘要，必须覆盖本次实际调查过的项目。
            await checkpoint_investigation_authorization(approved_code_project_ids)
            yield _sse_event("done", {
                "route": "agentic", "answer": answer_text, "answer_html": render_answer_markdown(answer_text),
                "sources": [source for source in sources if source.get("id") in approved_evidence_ids] if state.answer_ready else [],
                "observability": {"agent_state": state.public_snapshot()},
            })
        except PermissionError:
            logger.info("agentic_knowledge_authorization_revoked")
            yield _sse_event("error", {
                "message": "登录或知识访问范围已在执行期间发生变化，请重新发起查询",
                "error_code": "AUTHORIZATION_REVOKED", "retryable": False,
            })
        except Exception as exc:
            logger.exception("agentic_knowledge_stream_failed")
            yield _sse_event("error", {
                "message": "统一知识 Agent 执行失败，请稍后重试",
                "error_code": "KNOWLEDGE_AGENT_FAILED", "retryable": True,
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


async def hybrid_knowledge_stream(message: str, project_id: str, scope: AccessScope | None = None) -> StreamingResponse:
    """执行一轮跨源证据汇总；两类证据先独立获取，最后只生成一次答案。"""
    async def event_generator():
        started_at = perf_counter()
        citations = CitationRegistry()
        resolved_scope = scope or request_scope()
        state = KnowledgeAgentState(message, project_id=project_id, cross_project_policy="candidate_set", allowed_project_ids=set(resolved_scope.allowed_project_ids))
        state.apply_decision(KnowledgeDecision(
            next_action="retrieve_rag", route="hybrid", intent="cross_source_question",
            sub_questions=("文档依据", "项目代码事实"), reason="planner_selected_hybrid",
        ))
        yield _sse_event("step", {"step": "route", "status": "completed", "message": "已选择 Hybrid：分别查询文档和代码事实"})
        yield _sse_event("model", {"id": "hybrid_plan", "role": "Knowledge Agent · 跨源规划", "phase": "investigation", "status": "started"})
        plan_text = "当前问题同时需要通用知识依据和项目代码事实；我先分别检索两类来源，再交给同一个回答模型合并。"
        for offset in range(0, len(plan_text), 4):
            yield _sse_event("model", {"id": "hybrid_plan", "role": "Knowledge Agent · 跨源规划", "phase": "investigation", "status": "streaming", "delta": plan_text[offset:offset + 4]})
        yield _sse_event("model", {"id": "hybrid_plan", "role": "Knowledge Agent · 跨源规划", "phase": "investigation", "status": "completed"})
        try:
            # 文档侧只检索 RAG，避免把 Code Wiki 的事实表误当普通文档。
            yield _sse_event("step", {"step": "rag_retrieval", "status": "started", "message": "正在检索文档知识"})
            state.begin_round()
            state.admit_action("retrieve_rag", f"rag:{message.strip()[:300]}")
            embedding, embedding_usage = await asyncio.to_thread(embed, message, True)
            rag_candidates = await asyncio.to_thread(
                hybrid_search, embedding, message, ["rag"], settings.reranker_candidate_limit,
                resolved_scope.user_id, resolved_scope.department_id, resolved_scope.workspace_id,
            )
            try:
                rag_results = await asyncio.to_thread(rerank, message, rag_candidates, settings.max_context_chunks)
            except Exception:
                rag_results = rag_candidates[:settings.max_context_chunks]
            rag_context = await asyncio.to_thread(
                expand_context, rag_results, settings.max_context_tokens,
                resolved_scope.workspace_id, resolved_scope.user_id, resolved_scope.department_id,
            )
            rag_context = rag_context or rag_results
            rag_evidence = tuple(
                KnowledgeEvidence(
                    evidence_id=f"rag-{index + 1}", source_type="rag",
                    content_or_fact=str(item.get("content", "")),
                    locator=str(item.get("source_ref") or "") or None,
                    document_id=str(item.get("document_id") or "") or None,
                    evidence_kind="document_chunk",
                    confidence=float(item.get("score", 0.0) or 0.0),
                )
                for index, item in enumerate(rag_context)
            )
            state.record_tool_result(KnowledgeToolResult(
                tool_call_id=f"rag-{state.used_tool_calls}", capability="search_rag",
                status="success" if rag_context else "empty", result={"count": len(rag_context)},
                evidence=rag_evidence,
                usage={"candidate_count": len(rag_candidates), "context_count": len(rag_context)},
                provenance={"source_system": "postgresql+pgvector"},
            ))
            yield _sse_event("step", {"step": "rag_retrieval", "status": "completed", "message": f"文档侧得到 {len(rag_context)} 段上下文", "metrics": {"candidate_count": len(rag_candidates), "context_tokens": sum(item.get("context_tokens", 0) for item in rag_context)}})

            # Hybrid 的代码分支复用完整 Code Wiki Agent，使其按问题自行选择符号、源码和配置工具。
            yield _sse_event("step", {"step": "code_retrieval", "status": "started", "message": "Code Wiki Agent 正在调查项目代码"})
            state.begin_round()
            state.admit_action("query_code_wiki", f"codewiki:{project_id}:{message.strip()[:300]}", requires_project=True)
            remaining_tool_budget = state.max_tool_calls - state.used_tool_calls
            remaining_round_budget = state.max_rounds - state.used_rounds
            if remaining_tool_budget <= 0 or remaining_round_budget <= 0:
                state.mark_refused("no_child_agent_budget")
                raise RuntimeError("no budget remains for Code Wiki Agent")
            code_response = await _code_wiki_evidence_stream(CodeAgentRequest(
                project_id=project_id, message=message,
                max_rounds=remaining_round_budget, max_tool_calls=remaining_tool_budget,
                workspace_id=resolved_scope.workspace_id,
            ))
            code_done = None
            async for raw_chunk in code_response.body_iterator:
                text = raw_chunk.decode("utf-8") if isinstance(raw_chunk, bytes) else raw_chunk
                event_name = ""
                data = None
                for line in text.splitlines():
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            data = None
                if event_name == "done" and data:
                    code_done = data
                elif event_name == "error" and data:
                    _raise_child_agent_error(data, "Code Wiki Agent failed")
                elif event_name == "model" and data and data.get("phase") == "investigation":
                    # 前缀化调用 ID，避免与顶层和 RAG 模型事件冲突。
                    data = {**data, "id": f"hybrid_code_{data.get('id', 'investigation')}"}
                    yield _sse_event("model", data)
                elif event_name == "step" and data:
                    data = {**data, "step": f"hybrid_code_{data.get('step', 'step')}"}
                    yield _sse_event("step", data)
            if not code_done:
                raise RuntimeError("Code Wiki Agent did not return a final result")
            code_commit = code_done.get("project", {}).get("commit")
            code_sources = []
            for index, source in enumerate(code_done.get("sources", []), start=1):
                code_sources.append({
                    "id": source.get("id", f"C{index}"), "document_id": project_id, "type": "wiki",
                    "name": code_done.get("project", {}).get("project_name", "Code Wiki"),
                    "commit": code_commit,
                    "path": source.get("path"), "line": source.get("line", 1),
                    "label": source.get("label"), "symbol_id": source.get("symbol_id"),
                    "ref": f"{source.get('path', 'source')}:{source.get('line', 1)}",
                    "citation": f"[{source.get('id', 'C' + str(index))}]",
                })
            code_summary = code_done.get("answer") or "代码侧未形成可验证结论。"
            code_evidence_sufficient = code_done.get("observability", {}).get("evidence_sufficient")
            code_tool_calls = int(code_done.get("observability", {}).get("tool_calls", 0) or 0)
            state.record_child_tool_calls(code_tool_calls)
            state.record_child_rounds(int(code_done.get("observability", {}).get("completed_rounds", 0) or 0))
            if code_commit and code_sources and code_evidence_sufficient is True:
                state.commit_id = str(code_commit)
                code_evidence = tuple(
                    KnowledgeEvidence(
                        evidence_id=f"code-{index + 1}", source_type="code_wiki",
                        content_or_fact=str(source.get("label") or code_summary),
                        locator=str(source.get("ref") or "") or None,
                        project_id=project_id, commit_id=str(code_commit),
                        evidence_kind="source_citation", derivation="code_wiki_agent",
                    )
                    for index, source in enumerate(code_sources)
                )
                state.record_tool_result(KnowledgeToolResult(
                    tool_call_id=f"codewiki-{state.used_tool_calls}", capability="query_codewiki",
                    status="success", result={"summary": code_summary, "commit": code_commit},
                    evidence=code_evidence,
                    usage={"tool_calls": code_tool_calls},
                    provenance={"source_system": "code_wiki", "commit": code_commit},
                ))
            else:
                state.record_tool_result(KnowledgeToolResult(
                    tool_call_id=f"codewiki-{state.used_tool_calls}", capability="query_codewiki",
                    status="partial" if code_commit or code_sources else "empty",
                    result={"summary": code_summary, "commit": code_commit},
                    warnings=("代码证据未同时具备 Commit 和可定位引用",),
                    provenance={"source_system": "code_wiki", "commit": code_commit},
                ))
            yield _sse_event("step", {"step": "code_retrieval", "status": "completed", "message": f"Code Wiki Agent 完成调查，得到 {len(code_sources)} 条源码引用", "metrics": {"tool_calls": code_tool_calls, "citation_count": len(code_sources)}})

            evidence_parts = []
            sources = []
            for item in rag_context:
                citation = f"[R{len(sources) + 1}]"
                evidence_parts.append(f"{citation} 类型=RAG 来源={item.get('source_ref')}\n{item.get('content', '')}")
                sources.append({"id": len(sources) + 1, "document_id": str(item["document_id"]), "type": item["source_type"], "name": item["source_name"], "ref": item["source_ref"], "score": round(float(item.get("score", item.get("vector_score", 0.0))), 4), "citation": citation})
            if not rag_context or not code_commit or not code_sources or code_evidence_sufficient is not True:
                state.mark_refused("incomplete_cross_source_evidence")
                missing = []
                if not rag_context:
                    missing.append("RAG 文档依据")
                if not code_commit:
                    missing.append("代码版本 Commit")
                if not code_sources or code_evidence_sufficient is not True:
                    missing.append("可验证的代码引用")
                refusal = "跨源证据不足，无法可靠完成联合判断。缺少：" + "、".join(missing) + "。"
                yield _sse_event("done", {"route": "hybrid", "retrieval_mode": "rag+code-agent", "answer": refusal, "answer_html": render_answer_markdown(refusal), "sources": sources + code_sources, "observability": {"agent_state": state.snapshot(), "code_commit": code_commit, "total_ms": round((perf_counter() - started_at) * 1000, 2)}})
                return
            code_citation_index = " ".join(f"{item['citation']}={item['ref']}" for item in code_sources)
            evidence_parts.append(f"[CODE] 类型=Code Wiki Agent 调查结论 Commit={code_commit}\n{code_summary}\n引用索引：{code_citation_index}")
            sources.extend(code_sources)
            if not evidence_parts:
                state.mark_refused("no_cross_source_evidence")
                yield _sse_event("done", {"route": "hybrid", "answer": "当前没有足够的文档或代码证据回答该问题。", "answer_html": render_answer_markdown("当前没有足够的文档或代码证据回答该问题。"), "sources": [], "observability": {"agent_state": state.snapshot(), "total_ms": round((perf_counter() - started_at) * 1000, 2)}})
                return

            yield _sse_event("step", {"step": "hybrid_answer", "status": "started", "message": "两类证据已合并，正在生成最终回答"})
            yield _sse_event("model", {"id": "hybrid_answer", "role": "Knowledge Agent · 综合回答", "phase": "answer", "status": "started"})
            parts = []
            usage = {}
            stream = stream_answer(message, "\n\n".join(evidence_parts))
            while True:
                item = await asyncio.to_thread(_next_stream_item, stream)
                if item is None:
                    break
                if item["type"] == "delta":
                    parts.append(item["content"])
                    yield _sse_event("model", {"id": "hybrid_answer", "role": "Knowledge Agent · 综合回答", "phase": "answer", "status": "streaming", "delta": item["content"]})
                elif item["type"] == "usage":
                    usage = item
            response = "".join(parts) or "无法生成回答。"
            yield _sse_event("model", {"id": "hybrid_answer", "role": "Knowledge Agent · 综合回答", "phase": "answer", "status": "completed"})
            state.mark_answer_ready()
            yield _sse_event("done", {"route": "hybrid", "retrieval_mode": "rag+code-agent", "answer": response, "answer_html": render_answer_markdown(response), "sources": sources, "observability": {"rag_candidates": len(rag_candidates), "rag_context_count": len(rag_context), "code_tool_calls": code_done.get("observability", {}).get("tool_calls", 0), "code_citation_count": len(code_sources), "embedding_tokens": embedding_usage.get("prompt_tokens", 0), "chat_output_tokens": usage.get("completion_tokens", 0), "agent_state": state.snapshot(), "total_ms": round((perf_counter() - started_at) * 1000, 2)}})
        except PermissionError:
            logger.info("hybrid_knowledge_authorization_revoked")
            yield _sse_event("error", {
                "message": "登录或知识访问范围已在执行期间发生变化，请重新发起查询",
                "error_code": "AUTHORIZATION_REVOKED", "retryable": False,
            })
        except Exception as exc:
            logger.exception("hybrid_knowledge_stream_failed")
            yield _sse_event("error", {
                "message": "混合知识查询失败，请稍后重试",
                "error_code": "HYBRID_QUERY_FAILED", "retryable": True,
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


async def _collect_knowledge_stream_result(response: StreamingResponse) -> dict:
    """为旧同步接口消费统一 SSE；只返回通过 Answer Gate 的 done 结果。"""
    buffer = ""
    async for raw_chunk in response.body_iterator:
        buffer += raw_chunk.decode("utf-8") if isinstance(raw_chunk, bytes) else raw_chunk
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            event_name = "message"
            data = None
            for line in block.splitlines():
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        data = None
            if event_name == "done" and isinstance(data, dict):
                return data
            if event_name == "error" and isinstance(data, dict):
                status_code = 403 if data.get("error_code") == "AUTHORIZATION_REVOKED" else 502
                raise HTTPException(status_code, data.get("message") or "knowledge agent failed")
    raise HTTPException(502, "knowledge agent returned no approved answer")


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    """兼容旧同步 RAG 客户端，但回答统一经过顶层 Agent 和 Answer Gate。"""
    if not request.message.strip():
        raise HTTPException(400, "message is required")
    scope = request_scope(request.workspace_id)
    response = await agentic_knowledge_stream(
        request.message.strip(), None, scope=scope, capability_mode="rag",
    )
    return await _collect_knowledge_stream_result(response)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """兼容旧 SSE RAG 客户端，直接复用统一 Agent 的受控输出。"""
    if not request.message.strip():
        raise HTTPException(400, "message is required")
    scope = request_scope(request.workspace_id)
    return await agentic_knowledge_stream(
        request.message.strip(), None, scope=scope, capability_mode="rag",
    )


@app.post("/api/code-wiki/agent/stream")
async def code_wiki_agent_stream(request: CodeAgentRequest) -> StreamingResponse:
    """兼容旧客户端，但回答统一交给顶层 Agent 和 Answer Gate。"""
    if not request.message.strip():
        raise HTTPException(400, "message is required")
    project_id = parse_document_id(request.project_id)
    scope = request_scope(request.workspace_id)
    if project_id not in scope.allowed_project_ids or not get_code_overview(project_id, scope.workspace_id):
        raise HTTPException(404, "code project not found in current workspace")
    return await agentic_knowledge_stream(
        request.message.strip(), project_id, scope=scope, capability_mode="codewiki",
    )


@app.post("/api/knowledge/stream")
async def knowledge_stream(request: KnowledgeStreamRequest) -> StreamingResponse:
    """统一知识入口：约束来源后复用 RAG 或 Code Wiki 执行器。"""
    if not request.message.strip():
        raise HTTPException(400, "message is required")

    scope = request_scope(request.workspace_id)
    project_id = None
    if request.project_id:
        try:
            project_id = str(parse_document_id(request.project_id))
        except (TypeError, ValueError):
            raise HTTPException(400, "invalid project_id") from None
        # 项目 ID 只是客户端选择，最终必须属于当前用户在当前空间的授权候选集。
        if project_id not in scope.allowed_project_ids or not get_code_overview(project_id, scope.workspace_id):
            raise HTTPException(404, "code project not found in current workspace")

    if request.mode == "codewiki" and not project_id:
        raise HTTPException(400, "project_id is required for codewiki mode")
    if request.mode == "hybrid" and not project_id:
        raise HTTPException(400, "project_id is required for hybrid mode")
    if request.mode == "codewiki":
        return await agentic_knowledge_stream(
            request.message, project_id, scope=scope, capability_mode="codewiki",
        )
    if request.mode == "rag":
        return await agentic_knowledge_stream(
            request.message, None, scope=scope, capability_mode="rag",
        )
    if request.mode == "hybrid":
        return await agentic_knowledge_stream(
            request.message, project_id, scope=scope, capability_mode="hybrid",
        )

    # auto 是唯一顶层动态循环；显式模式保留为单能力调试入口。
    return await agentic_knowledge_stream(request.message, project_id, scope=scope, capability_mode="auto")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")
