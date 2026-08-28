from pathlib import Path
import asyncio

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
    get_code_overview,
    get_code_symbol,
    initialize_code_wiki,
    list_code_projects,
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
    """扫描本地项目并写入代码事实层；当前不调用 LLM，也不生成推测性 Wiki。"""
    root = Path(request.path).expanduser()
    if not root.is_dir():
        raise HTTPException(404, "project path does not exist or is not a directory")
    try:
        result = persist_scan(scan_project(str(root)))
    except Exception as exc:
        logger.exception("code_wiki_scan_failed path=%s", root)
        raise HTTPException(500, f"code wiki scan failed: {exc}") from exc
    return {"status": "ok", **result}


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


@app.get("/api/code-wiki/symbols")
def code_symbol_search(
    project_id: str = Query(...),
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    """按名称、限定名或文件路径搜索代码符号。"""
    normalized_id = parse_document_id(project_id)
    return {"items": search_code_symbols(normalized_id, q.strip(), limit)}


@app.get("/api/code-wiki/symbols/{symbol_id}")
def code_symbol_detail(symbol_id: str) -> dict:
    """返回符号定位及其 imports/calls 关系，后续直接作为 Wiki Agent 工具。"""
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
