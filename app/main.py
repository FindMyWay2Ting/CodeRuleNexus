from pathlib import Path
import asyncio
import re
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import bleach
import json
import logging
import markdown
from time import perf_counter
from pydantic import BaseModel

from .config import settings
from .code_wiki import (
    DEFAULT_REPOSITORY_ROOT,
    IGNORED_DIRS,
    get_code_overview,
    get_code_symbol,
    import_github_repository,
    initialize_code_wiki,
    list_code_files,
    list_code_projects,
    managed_local_repository_path,
    normalize_uploaded_path,
    persist_scan,
    scan_project,
    search_code_symbols,
)
from .db import (
    delete_document,
    hybrid_search,
    expand_context,
    initialize,
    invalidate_document,
    list_document_revisions,
    list_knowledge,
    restore_document,
)
from .ingestion import ingest_file, ingest_path
from .llm import answer, embed, rerank

app = FastAPI(title="Two-Level Knowledge Base MVP", version="0.1.0")
logger = logging.getLogger("knowledge.observability")


def _cost(tokens: int, price_per_1k: float) -> float:
    """按配置单价估算费用；单价为 0 时保留 0，表示尚未配置计费信息。"""
    return round(tokens / 1000 * price_per_1k, 8)


def _sse_event(event: str, payload: dict) -> str:
    """把一个事件编码成浏览器可解析的 SSE 文本；JSON 保证中文和结构字段不丢失。"""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


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


class PathRequest(BaseModel):
    path: str


class GithubRepositoryRequest(BaseModel):
    repository_url: str


class InvalidateRequest(BaseModel):
    reason: str | None = None


@app.on_event("startup")
def startup() -> None:
    """服务启动时初始化数据库和兼容迁移。"""
    initialize()
    # Code Wiki 使用独立事实表，避免继续把代码关系塞进文档 RAG 分块。
    initialize_code_wiki()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "workspace_id": settings.workspace_id}


@app.get("/api/knowledge")
def knowledge(
    source_type: str | None = Query(default=None),
    scope_type: str | None = Query(default=None),
) -> dict:
    """返回知识管理页面所需的文档级列表，可按 Wiki/RAG 过滤。"""
    if source_type not in {None, "wiki", "rag"}:
        raise HTTPException(400, "source_type must be wiki or rag")
    if scope_type not in {None, "personal", "department", "workspace"}:
        raise HTTPException(400, "scope_type must be personal, department or workspace")
    return {
        "items": list_knowledge(
            source_type,
            scope_type,
            settings.current_user_id,
            settings.current_department_id,
        ),
        "source_type": source_type or "all",
    }


@app.get("/api/knowledge/{document_id}/revisions")
def revisions(document_id: str) -> dict:
    """返回指定文档的修订历史；workspace 校验由数据库查询统一完成。"""
    normalized_id = parse_document_id(document_id)
    items = list_document_revisions(normalized_id)
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
    if not invalidate_document(normalized_id, request.reason):
        raise HTTPException(404, "active document not found")
    return {"status": "ok", "document_id": normalized_id, "document_status": "invalid"}


@app.post("/api/knowledge/{document_id}/restore")
def restore(document_id: str) -> dict:
    """重新生效文档，让该文档的全部分块恢复参与检索。"""
    normalized_id = parse_document_id(document_id)
    if not restore_document(normalized_id):
        raise HTTPException(404, "invalid document not found")
    return {"status": "ok", "document_id": normalized_id, "document_status": "active"}


@app.delete("/api/knowledge/{document_id}")
def delete(document_id: str) -> dict:
    """永久删除文档和关联分块。"""
    normalized_id = parse_document_id(document_id)
    if not delete_document(normalized_id):
        raise HTTPException(404, "document not found")
    return {"status": "ok", "document_id": normalized_id, "deleted": True}


@app.post("/api/ingest/upload")
async def upload(
    file: UploadFile = File(...),
    # 这些值来自前端 FormData，必须显式声明为 Form 字段，不能当作 query 参数读取。
    source_type: str = Form("rag"),
    scope_type: str = Form("personal"),
    department_id: str | None = Form(None),
    project_name: str | None = Form(None),
    workspace_name: str | None = Form(None),
) -> dict:
    """上传文档；标题自动取文件名，作者暂时固定为 User，归属范围由表单选择。"""
    if source_type not in {"wiki", "rag"}:
        raise HTTPException(400, "source_type must be wiki or rag")
    # Wiki 以项目名组织，不参与个人/部门/工作空间三档业务归属选择；统一落在当前工作空间。
    if source_type == "wiki":
        scope_type = "workspace"
        workspace_name = settings.current_workspace_name
    if scope_type not in {"personal", "department", "workspace"}:
        raise HTTPException(400, "scope_type must be personal, department or workspace")
    if scope_type == "department" and not (department_id and department_id.strip()):
        raise HTTPException(400, "department_id is required for department scope")
    if scope_type == "department" and department_id.strip() != settings.current_department_id:
        raise HTTPException(403, "department_id must match the current configured department")
    if source_type == "wiki" and not (project_name and project_name.strip()):
        raise HTTPException(400, "project_name is required for Wiki")
    if source_type == "rag" and project_name and project_name.strip():
        raise HTTPException(400, "project_name is only valid for Wiki")
    if scope_type == "workspace" and not (workspace_name and workspace_name.strip()):
        raise HTTPException(400, "workspace_name is required for workspace scope")
    if scope_type == "workspace" and workspace_name.strip() != settings.current_workspace_name:
        raise HTTPException(403, "workspace_name must match the current configured workspace")
    if scope_type != "workspace" and workspace_name and workspace_name.strip():
        raise HTTPException(400, "workspace_name is only valid for workspace scope")
    target = Path("uploads") / Path(file.filename or "upload.txt").name
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(await file.read())
    try:
        document_id, count = ingest_file(
            target,
            source_type,
            scope_type,
            department_id.strip() if department_id else None,
            project_name.strip() if project_name else None,
            workspace_name.strip() if workspace_name else None,
        )
    except Exception as exc:
        raise HTTPException(500, f"ingestion failed: {exc}") from exc
    return {"status": "ok", "document_id": document_id, "source": target.name, "chunks": count}


@app.post("/api/scan/local")
def scan_local(request: PathRequest) -> dict:
    """扫描本地目录并将代码类文本作为 Wiki 知识导入。"""
    root = Path(request.path).expanduser()
    if not root.exists():
        raise HTTPException(404, "path does not exist")
    try:
        count = ingest_path(root, "wiki")
    except Exception as exc:
        raise HTTPException(500, f"scan failed: {exc}") from exc
    return {"status": "ok", "path": str(root), "chunks": count}


@app.post("/api/code-wiki/scan")
def scan_code_project(request: PathRequest) -> dict:
    """扫描本地项目；Tree-sitter 建立结构，Go SCIP 可选增强语义，全程不调用 LLM。"""
    root = Path(request.path).expanduser()
    if not root.is_dir():
        raise HTTPException(404, "project path does not exist or is not a directory")
    try:
        result = persist_scan(scan_project(str(root)))
    except Exception as exc:
        logger.exception("code_wiki_scan_failed path=%s", root)
        raise HTTPException(500, f"code wiki scan failed: {exc}") from exc
    return {"status": "ok", **result}


@app.post("/api/code-wiki/import/github")
def import_github_code_project(request: GithubRepositoryRequest) -> dict:
    """拉取公开 GitHub 仓库并复用本地 Code Wiki 扫描链路。"""
    try:
        imported = import_github_repository(request.repository_url)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    try:
        scan = scan_project(imported["path"])
        scan["source"] = {
            "type": "github",
            "repository_url": imported["repository_url"],
            "owner": imported["owner"],
            "repository": imported["repository"],
        }
        result = persist_scan(scan)
    except Exception as exc:
        logger.exception("code_wiki_github_scan_failed repository=%s", imported["repository_url"])
        raise HTTPException(500, f"GitHub 仓库已拉取，但代码扫描失败：{exc}") from exc
    return {"status": "ok", "import_action": imported["action"], **result}


@app.post("/api/code-wiki/import/local")
async def import_local_code_project(files: list[UploadFile] = File(...)) -> dict:
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

    try:
        for upload in files:
            current_project, relative_path = normalize_uploaded_path(upload.filename or "")
            if project_name is None:
                project_name = current_project
            elif current_project != project_name:
                raise ValueError("一次只能上传一个项目目录")
            if any(part.lower() in IGNORED_DIRS for part in relative_path.parts):
                continue
            relative_key = relative_path.as_posix()
            if relative_key in seen_paths:
                raise ValueError(f"项目中存在重复路径：{relative_key}")
            seen_paths.add(relative_key)
            target_file = staging_project.joinpath(*relative_path.parts)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            file_bytes = 0
            with target_file.open("wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    file_bytes += len(chunk)
                    total_bytes += len(chunk)
                    if file_bytes > max_file_bytes:
                        raise ValueError(f"单个文件不能超过 20 MB：{relative_key}")
                    if total_bytes > max_total_bytes:
                        raise ValueError("项目上传总大小不能超过 300 MB")
                    output.write(chunk)
            saved_files += 1

        if not project_name or saved_files == 0:
            raise ValueError("所选目录没有可扫描文件")

        target = managed_local_repository_path(project_name, DEFAULT_REPOSITORY_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
        existed = target.exists()
        installed = False
        try:
            if existed:
                target.replace(backup)
            staging_project.replace(target)
            installed = True
            scan = scan_project(str(target))
            scan["project_name"] = project_name
            scan["source"] = {
                "type": "local_upload",
                "project_name": project_name,
                "uploaded_files": saved_files,
            }
            result = persist_scan(scan)
        except Exception:
            if installed and target.exists():
                shutil.rmtree(target, ignore_errors=True)
            if backup.exists():
                backup.replace(target)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
        return {
            "status": "ok",
            "import_action": "updated" if existed else "uploaded",
            "uploaded_files": saved_files,
            **result,
        }
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("code_wiki_local_upload_failed project=%s", project_name or "unknown")
        raise HTTPException(500, f"本地项目上传或扫描失败：{exc}") from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        for upload in files:
            await upload.close()


@app.get("/api/code-wiki/projects")
def code_projects() -> dict:
    """列出已经进入独立 Code Wiki 事实层的项目。"""
    return {"items": list_code_projects()}


@app.get("/api/code-wiki/projects/{project_id}/overview")
def code_project_overview(project_id: str) -> dict:
    """返回项目组件证据和文件/符号/关系数量，作为架构理解层的输入。"""
    normalized_id = parse_document_id(project_id)
    result = get_code_overview(normalized_id)
    if not result:
        raise HTTPException(404, "code project not found")
    return result


@app.get("/api/code-wiki/projects/{project_id}/files")
def code_project_files(project_id: str) -> dict:
    """返回项目当前 Commit 的文件、语言、行数、符号数和关系数。"""
    normalized_id = parse_document_id(project_id)
    return {"items": list_code_files(normalized_id)}


@app.get("/api/code-wiki/symbols")
def code_symbol_search(
    project_id: str = Query(...),
    q: str = Query(default=""),
    file_path: str | None = Query(default=None),
    symbol_kind: str | None = Query(default=None),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """搜索当前 Commit 的符号，也可按文件和符号类型筛选。"""
    normalized_id = parse_document_id(project_id)
    return {"items": search_code_symbols(
        normalized_id,
        q.strip(),
        limit,
        file_path.strip() if file_path else None,
        symbol_kind.strip() if symbol_kind else None,
    )}


@app.get("/api/code-wiki/symbols/{symbol_id}")
def code_symbol_detail(symbol_id: str) -> dict:
    """返回符号定位、出站关系以及引用/实现该符号的入站关系。"""
    normalized_id = parse_document_id(symbol_id)
    result = get_code_symbol(normalized_id)
    if not result:
        raise HTTPException(404, "code symbol not found")
    return result


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


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    """执行问题路由、向量检索、证据拼接和带引用的模型回答。"""
    if not request.message.strip():
        raise HTTPException(400, "message is required")
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
            settings.current_user_id,
            settings.current_department_id,
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
        context_results = expand_context(results, settings.max_context_tokens)
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
        raise HTTPException(502, f"knowledge query failed: {exc}") from exc
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


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """通过 SSE 实时发送可验证的处理步骤，最后发送完整答案和证据。"""
    if not request.message.strip():
        raise HTTPException(400, "message is required")

    async def event_generator():
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
            "stage_ms": {},
        }

        async def step_started(step: str, message: str):
            yield _sse_event("step", {"step": step, "status": "started", "message": message})

        async def step_completed(step: str, message: str, metrics: dict | None = None):
            payload = {"step": step, "status": "completed", "message": message}
            if metrics:
                payload["metrics"] = metrics
            yield _sse_event("step", payload)

        yield _sse_event("step", {"step": "route", "status": "completed", "message": f"已选择 {route} 检索", "metrics": {"route": route}})
        try:
            stage = "embedding"
            yield _sse_event("step", {"step": "embedding", "status": "started", "message": "正在生成查询向量"})
            stage_started = perf_counter()
            query_embedding, embedding_usage = await asyncio.to_thread(embed, request.message, True)
            telemetry["embedding_tokens"] = embedding_usage.get("prompt_tokens", 0)
            telemetry["stage_ms"]["embedding"] = round((perf_counter() - stage_started) * 1000, 2)
            yield _sse_event("step", {"step": "embedding", "status": "completed", "message": "查询向量已生成", "metrics": {"tokens": telemetry["embedding_tokens"], "duration_ms": telemetry["stage_ms"]["embedding"]}})

            stage = "hybrid_retrieval"
            yield _sse_event("step", {"step": "hybrid_retrieval", "status": "started", "message": "正在执行向量和关键词混合检索"})
            stage_started = perf_counter()
            candidates = await asyncio.to_thread(
                hybrid_search,
                query_embedding,
                request.message,
                source_types,
                settings.reranker_candidate_limit,
                settings.current_user_id,
                settings.current_department_id,
            )
            telemetry["candidate_count"] = len(candidates)
            telemetry["reranker_input_tokens"] = sum(max(1, len(str(item.get("content", ""))) // 2) for item in candidates)
            telemetry["stage_ms"]["hybrid_retrieval"] = round((perf_counter() - stage_started) * 1000, 2)
            yield _sse_event("step", {"step": "hybrid_retrieval", "status": "completed", "message": f"已召回 {len(candidates)} 个候选", "metrics": {"candidate_count": len(candidates), "duration_ms": telemetry["stage_ms"]["hybrid_retrieval"]}})

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

            if not results:
                telemetry["total_ms"] = round((perf_counter() - started_at) * 1000, 2)
                response = {"route": route, "retrieval_mode": retrieval_mode, "answer": "当前知识库没有检索到相关内容，请先导入项目或文档。", "answer_html": render_answer_markdown("当前知识库没有检索到相关内容，请先导入项目或文档。"), "sources": [], "observability": telemetry}
                yield _sse_event("done", response)
                return

            stage = "context_builder"
            yield _sse_event("step", {"step": "context_builder", "status": "started", "message": "正在构建回答上下文"})
            stage_started = perf_counter()
            context_results = await asyncio.to_thread(expand_context, results, settings.max_context_tokens)
            if not context_results:
                context_results = results
            telemetry["context_count"] = len(context_results)
            telemetry["parent_context_count"] = sum(item.get("context_level") == "parent" for item in context_results)
            telemetry["context_tokens"] = sum(item.get("context_tokens", 0) for item in context_results)
            telemetry["stage_ms"]["context_builder"] = round((perf_counter() - stage_started) * 1000, 2)
            retrieval_mode = f"{retrieval_mode}+parent-context"
            yield _sse_event("step", {"step": "context_builder", "status": "completed", "message": f"已构建 {len(context_results)} 段上下文", "metrics": {"parent_count": telemetry["parent_context_count"], "context_tokens": telemetry["context_tokens"], "duration_ms": telemetry["stage_ms"]["context_builder"]}})

            evidence = "\n\n".join(
                f"[{index + 1}] 类型={item['source_type']} 来源={item['source_ref']}\n{item['content']}"
                for index, item in enumerate(context_results)
            )
            stage = "chat_llm"
            yield _sse_event("step", {"step": "chat_llm", "status": "started", "message": "正在基于证据生成回答"})
            stage_started = perf_counter()
            response_text, chat_usage = await asyncio.to_thread(answer, request.message, evidence, True)
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
            yield _sse_event("done", {"route": route, "retrieval_mode": retrieval_mode, "context_tokens": telemetry["context_tokens"], "answer": response_text, "answer_html": render_answer_markdown(response_text), "sources": sources, "observability": telemetry})
        except Exception as exc:
            telemetry["failed_stage"] = stage
            telemetry["error_type"] = type(exc).__name__
            telemetry["total_ms"] = round((perf_counter() - started_at) * 1000, 2)
            logger.exception("chat_stream_observability_failed %s", json.dumps(telemetry, ensure_ascii=False))
            yield _sse_event("error", {"message": f"knowledge query failed: {exc}", "observability": telemetry})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")
