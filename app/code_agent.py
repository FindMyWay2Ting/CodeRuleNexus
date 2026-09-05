"""单项目 Code Wiki Agent 的受限工具层。

模型只能选择这里声明的工具。每个工具都绑定 project_id、当前 Commit 和结果预算，
不会接收任意磁盘路径，也不会把模型文本直接执行为代码或 SQL。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .code_wiki import (
    get_code_overview,
    get_code_symbol,
    list_code_architecture,
    list_code_architecture_links,
    list_code_config_facts,
    list_code_file_inventory,
    read_code_inventory_source,
    read_code_source,
    search_code_symbols,
    trace_code_call_chain,
)


CODE_AGENT_TOOLS = [
    {"type": "function", "function": {"name": "list_project_files", "description": "查看当前项目文件资产。支持按中文文件角色、扩展名或路径片段过滤；首次调查项目时调用，发现配置文件后再读取原文。", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_symbols", "description": "按名称搜索当前项目的函数、类、方法或文件。先搜索再读取详情。", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "symbol_kind": {"type": "string"}, "file_path": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_symbol", "description": "读取一个符号的定义位置、签名、调用、引用和实现关系。", "parameters": {"type": "object", "properties": {"symbol_id": {"type": "string"}}, "required": ["symbol_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "read_source", "description": "读取当前项目已索引文件的一段源码，单次最多 120 行。", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path", "start_line", "end_line"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "trace_call_path", "description": "从符号沿已解析 calls 边追踪有界调用链。", "parameters": {"type": "object", "properties": {"symbol_id": {"type": "string"}, "max_depth": {"type": "integer", "minimum": 1, "maximum": 4}}, "required": ["symbol_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "find_architecture", "description": "查询组件、模块、API、配置、客户端、下游调用以及配置到调用的关联。", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "find_config_facts", "description": "查询扫描时保存的 YAML、Docker Compose 和 Helm 配置事实。不要把多个组件拼成一个 query；组件调查应分别查询 kafka、mongo、redis 等关键词。count 是总匹配数，items 是当前页；当 count 大于 items 数量时必须使用 offset 分页。", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "path_prefix": {"type": "string"}, "fact_types": {"type": "array", "items": {"type": "string"}}, "formats": {"type": "array", "items": {"type": "string"}}, "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "finish_investigation", "description": "当现有证据已经足够回答，或继续调查不会产生必要新证据时结束调查。该控制工具必须单独调用；调用后服务端会启动独立的最终回答模型流。", "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "结束调查的简短原因"}, "evidence_sufficient": {"type": "boolean", "description": "现有证据是否足以回答"}}, "required": ["reason", "evidence_sufficient"], "additionalProperties": False}}},
]


def analysis_message_for_tool(name: str, arguments: dict) -> str:
    """生成可以展示给用户的公开分析摘要，不暴露模型的隐式思维链。"""
    path = str(arguments.get("path") or "").strip()
    query = str(arguments.get("query") or "").strip()
    messages = {
        "list_project_files": "我先查看项目文件资产，确认有哪些文件类型、配置格式以及当前解析覆盖范围",
        "search_symbols": f"我已经知道需要定位一个代码符号，接下来搜索 {query or '相关函数、类或方法'} 的定义位置",
        "get_symbol": "我已经找到候选符号，接下来读取它的定义、引用和实现关系，确认它在项目中的真实职责",
        "read_source": f"当前索引证据还不足以确认细节，接下来读取 {path or '对应源码或配置文件'} 的原文进行核对",
        "trace_call_path": "我已经确认了起点符号，接下来沿已解析的调用关系追踪调用方和被调用方",
        "find_architecture": f"我需要补充架构层事实，接下来检查 {query or '组件、配置、服务边界和调用关系'}",
        "find_config_facts": f"我已经知道需要查看配置层，接下来查询 {query or 'YAML、Compose 和 Helm 的通用配置事实'}",
    }
    return messages.get(name, f"我需要补充证据，接下来调用 {name} 进行验证")


def analysis_result_message(name: str, result: dict) -> str:
    """说明拿到结果后形成的下一步判断，内容只基于工具返回结果。"""
    if result.get("error"):
        return "这次查询没有得到可用证据，我会把它记录为证据缺口，避免把猜测当成结论"
    if name == "list_project_files":
        coverage = result.get("coverage", {})
        if coverage.get("partial", 0) or coverage.get("unclassified", 0):
            return "我知道项目里存在尚未完整解析的文件，下一步需要读取相关文件原文，补齐静态索引没有覆盖的事实"
        return "我已经知道项目的文件资产和解析范围，下一步可以根据问题选择配置、符号或架构关系进行验证"
    if name == "search_symbols":
        return "我已经拿到候选符号，下一步需要读取具体定义和关系，确认搜索命中是否就是问题中的目标"
    if name == "get_symbol":
        return "我已经知道这个符号的定义和关系数量，下一步根据问题决定是否需要继续读取源码或追踪调用链"
    if name == "read_source":
        return "我已经拿到源码或配置原文，下一步把它与索引事实对照，确认结论和文件行号"
    if name == "trace_call_path":
        return "我已经拿到调用链，下一步检查链路中是否存在未解析边或需要补充的架构事实"
    if name == "find_architecture":
        return "我已经拿到架构事实和关联，下一步判断证据是否足够回答问题；不足时继续补证"
    if name == "find_config_facts":
        if result.get("count", 0):
            return "我已经拿到配置层事实，下一步把声明的服务或依赖与源码中的实际初始化和调用进行交叉核对"
        return "配置解析器没有返回匹配事实，下一步需要检查文件清单和原始配置，区分项目确实没有配置还是解析覆盖不足"
    return "我已经拿到这次工具结果，下一步根据证据完整性决定是否继续查询"


@dataclass
class CitationRegistry:
    """给模型和页面复用同一组稳定的本次回答证据编号。"""

    items: list[dict] = field(default_factory=list)
    keys: dict[tuple, str] = field(default_factory=dict)

    def add(self, path: str, line: int, label: str, symbol_id: str | None = None) -> str:
        key = (path, int(line or 1), symbol_id or "")
        if key in self.keys:
            return self.keys[key]
        citation = f"C{len(self.items) + 1}"
        self.keys[key] = citation
        self.items.append({"id": citation, "path": path, "line": int(line or 1), "label": label, "symbol_id": symbol_id})
        return citation


def build_code_agent_messages(project_id: str, question: str, workspace_id: str | None = None) -> tuple[list[dict], dict]:
    """构建固定在指定工作空间、项目和 Commit 内的 Agent 初始消息。"""
    overview = get_code_overview(project_id, workspace_id)
    if not overview:
        raise ValueError("code project not found")
    project = {
        "project_id": project_id,
        "project_name": overview["project_name"],
        "commit": overview["current_commit"],
        "counts": overview["counts"],
    }
    system = (
        "你是单项目 Code Wiki 调查 Agent。只能依据工具返回的当前 Commit 证据回答，不能凭常识补全代码事实。"
        "每个问题都按调查协议执行：先判断问题类型，再拆成可验证的子问题，列出需要的证据，选择最小工具；每轮拿到结果后更新假设和未解决问题，最后检查证据覆盖后再回答。"
        "首次面对项目结构问题，先调用 list_project_files；配置问题先找配置文件和配置事实；符号问题先 search_symbols 再 get_symbol/read_source；调用链问题从符号详情开始，再 trace_call_path。"
        "识别基础设施组件时必须分别核验：依赖声明、配置声明、客户端初始化、实际调用。不能把业务服务目录当成 Kafka/Mongo 等基础设施，也不能只凭项目名称推断组件存在。"
        "配置查询必须使用单一关键词或结构化过滤。若 count 大于当前 items 数量，使用 offset 分页；不要重复请求同一页。若配置事实为空，依次检查文件资产、读取候选配置原文、检查依赖文件和源码初始化，最后才能判断未发现。"
        "如果发现 parser_status 为 partial、unsupported 或 unclassified，必须说明覆盖缺口，并优先调用 read_source 核验原文。工具返回为空只代表当前查询没有命中，不等于项目不存在该事实。"
        "结论按证据强度区分 configured（配置声明）、dependency_only（仅依赖声明）、initialized（客户端初始化）、used（实际调用）、suspected（弱证据）和 not_found（已检查范围内未发现）。"
        "工具结果不支持原假设时必须修正或否定假设；多个证据矛盾时列出矛盾。不要重复调用已成功完成且范围不变的工具，不要在停止前跳过未解决的必需证据。"
        "每次准备调用工具时，必须先在 assistant content 中输出面向用户的简短调查记录，固定为三行：当前已知、信息缺口、下一步。下一步必须写明准备调用的工具及原因，然后在同一响应中发起工具调用。"
        "调查记录必须基于当前对话和工具证据，不得虚构结果；只给结论式说明，不展开隐藏思维链或冗长逐步推演。"
        "当证据足够回答，或剩余缺口已经无法通过现有工具补齐时，单独调用 finish_investigation，不要直接输出最终回答。最终回答将由服务端在独立阶段生成。"
        "每个最终结论使用工具提供的 [C1] 格式引用；配置事实也必须有引用。普通 references 不等于运行时调用。"
        "若静态分析存在断链、动态依赖或证据不足，必须明确说明。最多进行有限次工具调用，回答使用中文，先给结论，再给证据、证据等级和边界。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"项目：{project['project_name']}\nCommit：{project['commit']}\n问题：{question}"},
    ], project


def assistant_message_dict(message) -> dict:
    """把 SDK 消息转成可继续提交的普通字典。"""
    payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        payload["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}
            for call in message.tool_calls
        ]
    return payload


def _compact_symbol(item: dict, citations: CitationRegistry) -> dict:
    citation = citations.add(item["path"], item["start_line"], item["qualified_name"], str(item["symbol_id"]))
    return {
        "citation": f"[{citation}]", "symbol_id": str(item["symbol_id"]), "name": item["name"],
        "qualified_name": item["qualified_name"], "kind": item["symbol_kind"],
        "path": item["path"], "start_line": item["start_line"], "end_line": item["end_line"],
    }


def execute_code_agent_tool(
    project_id: str,
    name: str,
    arguments: dict,
    citations: CitationRegistry,
    *,
    expected_commit: str,
) -> dict:
    """执行白名单工具；Agent 请求必须始终读取启动时锁定的 Commit。"""
    expected_commit = expected_commit.strip()
    if not expected_commit:
        raise ValueError("expected_commit is required")
    if name == "list_project_files":
        return list_code_file_inventory(
            project_id,
            str(arguments.get("query", "")),
            int(arguments.get("limit", 200)),
            int(arguments.get("offset", 0)),
            expected_commit,
        )

    if name == "search_symbols":
        query = str(arguments.get("query", "")).strip()[:120]
        items = search_code_symbols(
            project_id, query, 20,
            str(arguments.get("file_path") or "").strip() or None,
            str(arguments.get("symbol_kind") or "").strip() or None,
            expected_commit,
        )
        return {"items": [_compact_symbol(item, citations) for item in items]}

    if name == "get_symbol":
        item = get_code_symbol(str(arguments.get("symbol_id", "")))
        if (
            not item
            or str(item.get("project_id")) != project_id
            or item.get("commit_hash") != expected_commit
        ):
            return {"error": "symbol not found in selected project"}
        result = _compact_symbol(item, citations)
        result["signature"] = item.get("signature")
        result["docstring"] = item.get("docstring")
        result["outgoing"] = item.get("relations", [])[:20]
        result["incoming"] = item.get("incoming_relations", [])[:20]
        return result

    if name == "read_source":
        path = str(arguments.get("path", "")).strip()
        start = max(1, int(arguments.get("start_line", 1)))
        end = min(start + 119, max(start, int(arguments.get("end_line", start + 40))))
        source = read_code_source(project_id, path, start, end, expected_commit) or read_code_inventory_source(
            project_id, path, start, end, expected_commit,
        )
        if not source:
            return {"error": "source file not found in selected project"}
        citation = citations.add(source["path"], source["start_line"], f"{source['path']}:{source['start_line']}")
        return {"citation": f"[{citation}]", "path": source["path"], "start_line": source["start_line"], "end_line": source["end_line"], "stale": source.get("stale", False), "source": source["numbered_content"]}

    if name == "trace_call_path":
        depth = min(4, max(1, int(arguments.get("max_depth", 3))))
        chain = trace_code_call_chain(
            str(arguments.get("symbol_id", "")), depth, 40, expected_commit,
        )
        if not chain or chain.get("project_id") != project_id:
            return {"error": "symbol not found in selected project"}
        nodes = []
        for node in chain["nodes"]:
            compact = _compact_symbol(node, citations)
            compact["depth"] = node["depth"]
            compact["architecture_facts"] = node.get("architecture_facts", [])[:8]
            nodes.append(compact)
        return {"nodes": nodes, "edges": chain["edges"][:80], "unresolved_edges": chain["unresolved_edges"][:40], "cycles": chain["cycles"], "truncated": chain["truncated"]}

    if name == "find_architecture":
        query = str(arguments.get("query", "")).strip().casefold()[:120]
        facts = list_code_architecture(project_id, expected_commit)
        links = list_code_architecture_links(project_id, expected_commit)
        def matches(item: dict) -> bool:
            return not query or query in " ".join(str(value or "") for value in item.values()).casefold()
        fact_results = []
        for fact in (item for item in facts if matches(item)):
            citation = citations.add(fact["source_path"], fact["source_line"], fact["name"], str(fact["source_symbol_id"]) if fact.get("source_symbol_id") else None)
            fact_results.append({**fact, "citation": f"[{citation}]"})
            if len(fact_results) >= 30:
                break
        link_results = []
        for link in (item for item in links if matches(item)):
            source_citation = citations.add(link["source_path"], link["source_line"], link["source_name"], str(link["source_symbol_id"]) if link.get("source_symbol_id") else None)
            target_citation = citations.add(link["target_path"], link["target_line"], link["target_name"], str(link["target_symbol_id"]) if link.get("target_symbol_id") else None)
            link_results.append({**link, "source_citation": f"[{source_citation}]", "target_citation": f"[{target_citation}]"})
            if len(link_results) >= 30:
                break
        return {"facts": fact_results, "links": link_results}

    if name == "find_config_facts":
        return list_code_config_facts(
            project_id,
            str(arguments.get("query", "")),
            int(arguments.get("limit", 100)),
            int(arguments.get("offset", 0)),
            str(arguments.get("path_prefix", "")),
            arguments.get("fact_types", []),
            arguments.get("formats", []),
            citations,
            expected_commit,
        )

    return {"error": f"tool {name} is not allowed"}


def tool_result_message(tool_call_id: str, name: str, result: dict) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": json.dumps(result, ensure_ascii=False, default=str)[:30000]}
