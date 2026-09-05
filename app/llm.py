from openai import OpenAI
import httpx
import json
from types import SimpleNamespace

from .config import settings
from .knowledge_agent import AgentAction, KnowledgeDecision, Route


def _client(api_base: str, api_key: str, timeout: float = 60.0) -> OpenAI:
    """创建 OpenAI-compatible 客户端，聊天和 Embedding 可使用不同配置。"""
    if not api_base or not api_key:
        raise RuntimeError("LLM API configuration is incomplete")
    # 外部模型服务不可控，客户端必须有上限，避免一次 Agent 请求长期占住 SSE 连接。
    return OpenAI(base_url=api_base, api_key=api_key, timeout=timeout, max_retries=0)


def _usage_value(usage, name: str) -> int:
    """兼容 OpenAI SDK 的对象 usage 和部分兼容服务返回的字典 usage。"""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(name, 0) or 0)
    return int(getattr(usage, name, 0) or 0)


def _json_object(text: str) -> dict:
    """解析兼容模型常见的纯 JSON 或 fenced JSON 输出。"""
    content = (text or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")
    return payload


def plan_knowledge_query(
    question: str,
    available_routes: set[Route],
    *,
    project_selected: bool = False,
    cross_project_policy: str = "deny",
) -> KnowledgeDecision:
    """让顶层 Planner 在服务端允许的知识源内生成结构化初始计划。"""
    routes = sorted(available_routes)
    response = _client(settings.llm_api_base, settings.llm_api_key, timeout=45.0).chat.completions.create(
        model=settings.chat_model,
        temperature=0.0,
        extra_body={"enable_thinking": False},
        messages=[
            {
                "role": "system",
                "content": (
                    "你是企业知识平台的顶层 Planner。只输出 JSON 对象，不输出解释。"
                    "在服务端允许的知识源内选择 route，并拆出最多 5 个可验证子问题。"
                    "询问制度、规范、说明文档或业务知识时选择 rag；询问选定项目的源码、函数、类、"
                    "调用、依赖、配置文件或实现细节时选择 wiki；只有问题明确要求把项目实现与制度、"
                    "规范或文档要求进行比较、核验或联合解释时才选择 hybrid。"
                    "route=rag 时 next_action=retrieve_rag；route=wiki 时 next_action=query_code_wiki；"
                    "route=hybrid 时优先 next_action=retrieve_rag，后续由执行器继续补代码证据。"
                    "先判断查询范围：代码事实使用 strict_project；项目背景、规范和文档解释使用 soft_project；"
                    "只有问题明确要求跨项目比较且服务端授权时使用 candidate_projects。用户选中的项目是代码查询的硬边界，"
                    "不是整个问题的唯一知识来源。项目内证据不足时不能直接声称不存在，必须指出缺口并建议合法的下一步。"
                    "JSON 字段：route,intent,sub_questions,next_action,reason,scope_mode,expansion_required,claims。"
                ),
            },
            {"role": "user", "content": f"可用知识源：{routes}\n已选择项目：{project_selected}\n跨项目策略：{cross_project_policy}\n用户问题：{question}"},
        ],
    )
    payload = _json_object(response.choices[0].message.content or "{}")
    return KnowledgeDecision.from_payload(
        payload,
        allowed_actions={"retrieve_rag", "query_code_wiki"},
        allowed_routes=available_routes,
    )


def decide_next_knowledge_action(
    question: str,
    state: dict,
    allowed_actions: set[AgentAction],
    available_routes: set[Route],
) -> KnowledgeDecision:
    """根据当前全局证据状态选择一个原子动作，不预设知识源调用顺序。"""
    capability_catalog = {
        "retrieve_rag": "检索已授权的制度、规范、说明和业务文档，返回可定位文档片段",
        "query_code_wiki": "调查服务端锁定的当前项目和 Commit，返回源码、配置和关系证据",
        "answer": "现有证据已覆盖核心问题时结束调查",
        "refuse": "所有合法补证路径均不可用或已耗尽时拒答",
        "replan": "现有动作参数不合适，需要改变查询表达，但本动作本身不执行检索",
    }
    compact_state = {
        "scope": state.get("scope"),
        "query_scope": state.get("query_scope"),
        "executed_actions": state.get("executed_actions", [])[-8:],
        "tool_results": state.get("tool_results", [])[-6:],
        "evidence_count": state.get("evidence_count", 0),
        "evidence": [
            {
                "evidence_id": item.get("evidence_id"), "source_type": item.get("source_type"),
                "locator": item.get("locator"), "content": str(item.get("content_or_fact") or "")[:600],
                "project_id": item.get("project_id"), "commit_id": item.get("commit_id"),
            }
            for item in state.get("evidence", [])[-20:]
        ],
        "missing_information": state.get("missing_information", []),
        "budget": state.get("budget"),
    }
    response = _client(settings.llm_api_base, settings.llm_api_key, timeout=45.0).chat.completions.create(
        model=settings.chat_model,
        temperature=0.0,
        extra_body={"enable_thinking": False},
        messages=[
            {
                "role": "system",
                "content": (
                    "你是企业知识平台的顶层 Knowledge Agent。只输出 JSON，不回答用户问题。"
                    "你不是一次性路由器；每轮根据问题、服务端授权范围、完整能力目录、历史动作、"
                    "工具状态、已有证据、信息缺口和剩余预算，选择一个最能减少证据缺口的原子动作。"
                    "不要假设任何固定工具顺序。用户选择的项目是代码证据的硬边界，但不是整个问题的唯一来源。"
                    "当前项目查询为空只表示本次未命中，不能证明事实不存在；若问题允许文档解释且 RAG 尚未尝试，"
                    "应建议 retrieve_rag。不能建议未列入能力目录或未授权的项目。"
                    "answer 必须提供 supporting_evidence_ids，并将每条 claims 写成"
                    "{text,evidence_ids} 与证据逐条绑定；没有有效证据不得 answer。"
                    "refuse 仅在所有合法补证动作已耗尽时使用，并列出 missing_information。"
                    "public_update 只写可公开核验的已知事实、缺口和下一步，不输出隐藏推理。"
                    "需要查询代码时可填写 target_project_id；它必须来自当前状态中的 allowed_project_ids，"
                    "没有锚点项目时，根据 authorized_projects 中的项目名、语言、组件和 Commit 选择目标项目，"
                    "不能凭空创造项目 ID。项目目录只是发现信息，不是回答用户问题的证据。"
                    "字段：next_action,route,intent,scope_mode,expansion_required,queries,claims,"
                    "supporting_evidence_ids,missing_information,reason,public_update,target_project_id。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n允许动作：{sorted(allowed_actions)}\n"
                    f"可用路由：{sorted(available_routes)}\n能力目录："
                    f"{json.dumps({key: capability_catalog[key] for key in allowed_actions if key in capability_catalog}, ensure_ascii=False)}\n"
                    f"当前状态：{json.dumps(compact_state, ensure_ascii=False, default=str)[:18000]}"
                ),
            },
        ],
    )
    payload = _json_object(response.choices[0].message.content or "{}")
    return KnowledgeDecision.from_payload(
        payload, allowed_actions=allowed_actions, allowed_routes=available_routes,
    )


def grade_rag_evidence(question: str, evidence: list[dict], can_replan: bool) -> KnowledgeDecision:
    """判断文档证据能否支撑回答；模型只做决定，服务端控制允许的动作。"""
    compact_evidence = [
        {
            "source": item.get("source_ref"),
            "score": item.get("rerank_score", item.get("score")),
            "content": str(item.get("content") or "")[:900],
        }
        for item in evidence[:8]
    ]
    allowed: set[AgentAction] = {"answer", "refuse"}
    if can_replan:
        allowed.add("replan")
    response = _client(settings.llm_api_base, settings.llm_api_key, timeout=45.0).chat.completions.create(
        model=settings.chat_model,
        temperature=0.0,
        extra_body={"enable_thinking": False},
        messages=[
            {
                "role": "system",
                "content": (
                    "你是证据充分性评估器。只输出 JSON 对象，不回答问题。"
                    "判断现有证据是否直接覆盖问题所需事实，而不是仅判断是否存在检索结果。"
                    "next_action 只能从服务端给出的允许动作中选择。证据充分选 answer；"
                    "信息可通过换一种文档查询补齐时选 replan 并给最多 3 个 queries；"
                    "现有知识源无法支持时选 refuse。"
                    "JSON 字段：next_action,missing_information,queries,reason。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"允许动作：{sorted(allowed)}\n问题：{question}\n"
                    f"证据：{json.dumps(compact_evidence, ensure_ascii=False)}"
                ),
            },
        ],
    )
    payload = _json_object(response.choices[0].message.content or "{}")
    return KnowledgeDecision.from_payload(payload, allowed_actions=allowed)


def embed(text: str, return_usage: bool = False):
    """将一个文本分块转换成固定维度的向量；可选返回服务端 Token 用量。"""
    response = _client(settings.embedding_api_base, settings.embedding_api_key).embeddings.create(
        model=settings.embedding_model,
        input=[text],
        dimensions=settings.embedding_dimensions,
        encoding_format="float",
    )
    embedding = response.data[0].embedding
    if not return_usage:
        return embedding
    return embedding, {"prompt_tokens": _usage_value(getattr(response, "usage", None), "prompt_tokens")}


def answer(question: str, context: str, return_usage: bool = False):
    """只把检索证据交给模型；可选返回输入和输出 Token 用量。"""
    response = _client(settings.llm_api_base, settings.llm_api_key).chat.completions.create(
        model=settings.chat_model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是企业知识库助手。只能依据提供的证据回答，不要编造。"
                    "如果证据不足，明确说明无法确认。回答要简洁，并保留来源编号。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n证据：\n{context}",
            },
        ],
    )
    answer_text = response.choices[0].message.content or "无法生成回答。"
    if not return_usage:
        return answer_text
    usage = getattr(response, "usage", None)
    return answer_text, {
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
    }


def stream_answer(question: str, context: str, approved_claims: list[dict] | None = None):
    """流式生成最终回答；Agent 模式可传入已通过 Answer Gate 的 Claim 契约。"""
    claim_contract = json.dumps(approved_claims or [], ensure_ascii=False)
    contract_instruction = (
        "回答只能表述 Claim 契约中的结论，不得增加、扩展或猜测契约之外的事实。"
        "每条结论必须保留该 Claim 对应的证据编号；契约为空时按普通证据回答。"
        if approved_claims else ""
    )
    stream = _client(settings.llm_api_base, settings.llm_api_key).chat.completions.create(
        model=settings.chat_model,
        temperature=0.1,
        stream=True,
        stream_options={"include_usage": True},
        messages=[
            {
                "role": "system",
                "content": (
                    "你是企业知识库助手。只能依据提供的证据回答，不要编造。"
                    "如果证据不足，明确说明无法确认。回答要简洁，并保留来源编号。"
                    + contract_instruction
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\nClaim 契约：{claim_contract}\n\n证据：\n{context}",
            },
        ],
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield {"type": "delta", "content": delta}
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            yield {
                "type": "usage",
                "prompt_tokens": _usage_value(usage, "prompt_tokens"),
                "completion_tokens": _usage_value(usage, "completion_tokens"),
                "total_tokens": _usage_value(usage, "total_tokens"),
            }


def code_agent_completion(messages: list[dict], tools: list[dict] | None = None):
    """执行一次代码 Agent 决策；工具循环和权限边界由服务端控制。"""
    options = {
        "model": settings.chat_model,
        "temperature": 0.0,
        "messages": messages,
    }
    if tools:
        # 调查阶段必须选择一个业务工具或 finish_investigation，最终回答阶段不传 tools。
        options.update({
            "tools": tools,
            "tool_choice": "required",
            "extra_body": {"enable_thinking": False},
        })
    return _client(settings.llm_api_base, settings.llm_api_key, timeout=90.0).chat.completions.create(
        **options,
    )


def stream_code_agent_completion(messages: list[dict], tools: list[dict] | None = None):
    """流式执行代码 Agent，并在结束时重建完整文本和工具调用消息。"""
    options = {
        "model": settings.chat_model,
        "temperature": 0.0,
        "messages": messages,
        "stream": True,
    }
    if tools:
        # 调查阶段必须返回业务工具或显式结束工具，避免把无工具文本误判为最终回答。
        options.update({
            "tools": tools,
            "tool_choice": "required",
            "extra_body": {"enable_thinking": False},
        })

    stream = _client(settings.llm_api_base, settings.llm_api_key, timeout=90.0).chat.completions.create(
        **options,
    )
    content_parts: list[str] = []
    tool_parts: dict[int, dict[str, str]] = {}
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content_parts.append(delta.content)
            yield {"type": "delta", "content": delta.content}
        # 工具调用的名称和 JSON 参数也会被拆成多个增量，需要按 index 合并。
        for tool_delta in delta.tool_calls or []:
            item = tool_parts.setdefault(tool_delta.index, {"id": "", "name": "", "arguments": ""})
            if tool_delta.id:
                item["id"] += tool_delta.id
            function = tool_delta.function
            if function:
                if function.name:
                    item["name"] += function.name
                if function.arguments:
                    item["arguments"] += function.arguments

    tool_calls = [
        SimpleNamespace(
            id=item["id"] or f"tool_call_{index}",
            function=SimpleNamespace(name=item["name"], arguments=item["arguments"] or "{}"),
        )
        for index, item in sorted(tool_parts.items())
    ]
    yield {
        "type": "done",
        "message": SimpleNamespace(content="".join(content_parts), tool_calls=tool_calls),
    }


def stream_code_investigation_note(
    question: str,
    project_name: str,
    commit_hash: str,
    tool_calls: list,
    conversation: list[dict],
):
    """根据已有工具证据和本轮工具选择，流式生成面向用户的调查说明。"""
    selected_tools = [
        {"name": call.function.name, "arguments": call.function.arguments}
        for call in tool_calls
    ]
    recent_observations = [
        str(item.get("content") or "")[:2400]
        for item in conversation
        if item.get("role") == "tool"
    ][-3:]
    stream = _client(settings.llm_api_base, settings.llm_api_key, timeout=60.0).chat.completions.create(
        model=settings.chat_model,
        temperature=0.0,
        stream=True,
        extra_body={"enable_thinking": False},
        messages=[
            {
                "role": "system",
                "content": (
                    "你是代码调查过程说明器。只能根据给定的最近证据和本轮已选择工具，生成面向用户的公开调查摘要。"
                    "固定输出三行：当前已知、信息缺口、下一步。下一步要写明工具名称和选择原因。"
                    "不要回答用户最终问题，不要虚构尚未返回的工具结果，不要输出冗长推演。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"项目：{project_name}\nCommit：{commit_hash}\n问题：{question}\n"
                    f"最近工具证据：{json.dumps(recent_observations, ensure_ascii=False)}\n"
                    f"本轮已选择工具：{json.dumps(selected_tools, ensure_ascii=False)}"
                ),
            },
        ],
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content or ""
        if content:
            yield content


def rewrite_rag_queries(question: str, failed_query: str, reason: str = "no_candidates") -> list[str]:
    """让模型生成有限的检索改写；只返回查询字符串，不让模型直接修改检索结果。"""
    response = _client(settings.llm_api_base, settings.llm_api_key, timeout=45.0).chat.completions.create(
        model=settings.chat_model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是企业知识库的检索规划器。根据用户问题和上一轮检索失败原因，生成最多两个互不重复的检索查询。"
                    "保留用户问题中的关键实体，不要编造项目名称、事实或答案。只输出 JSON 数组，例如 [\"查询1\", \"查询2\"]。"
                ),
            },
            {
                "role": "user",
                "content": f"用户问题：{question}\n上一轮查询：{failed_query}\n失败原因：{reason}",
            },
        ],
    )
    text = response.choices[0].message.content or "[]"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item).strip()[:300] for item in payload if str(item).strip()][:2]


def rerank(query: str, results: list[dict], top_n: int) -> list[dict]:
    """使用阿里云原生 Text Rerank API 对向量初召回结果进行二次排序。"""
    if not results:
        return []
    if not settings.reranker_api_url or not settings.reranker_api_key:
        raise RuntimeError("Reranker API configuration is incomplete")

    response = httpx.post(
        settings.reranker_api_url,
        headers={"Authorization": f"Bearer {settings.reranker_api_key}"},
        json={
            "model": settings.reranker_model,
            "input": {"query": query, "documents": [item["content"] for item in results]},
            "parameters": {"return_documents": True, "top_n": top_n},
        },
        timeout=60.0,
    )
    response.raise_for_status()
    payload = response.json()
    ranked = payload.get("output", {}).get("results") or payload.get("results") or []

    reordered = []
    for item in ranked:
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < len(results):
            result = dict(results[index])
            result["rerank_score"] = float(item.get("relevance_score", 0.0))
            reordered.append(result)
    return reordered
