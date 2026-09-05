"""顶层知识 Agent 的状态契约。

这里定义跨知识源编排需要的状态，但暂不绑定具体模型或工作流框架。文档 RAG、Code
Wiki 和 Hybrid 都应该通过这个状态记录任务、证据和下一步动作，避免把 Agentic RAG
误认为某个检索器内部的查询重试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4


Route = Literal["rag", "wiki", "hybrid"]
ScopeMode = Literal["strict_project", "soft_project", "candidate_projects"]
ExpansionRequired = Literal["none", "rag_allowed", "cross_project_required", "user_confirmation_required"]
CrossProjectPolicy = Literal["deny", "allow_authorized", "candidate_set"]
AgentAction = Literal[
    "retrieve_rag", "query_code_wiki", "traverse_code_graph", "fetch_document",
    "fetch_source", "assess_evidence", "answer", "refuse", "replan",
]
KnowledgePhase = Literal["plan", "execute", "observe", "replan", "finish", "refuse"]
ToolStatus = Literal["success", "empty", "partial", "failed", "timeout", "cancelled"]


@dataclass(frozen=True)
class ClaimSupport:
    """模型准备写入答案的一条 Claim 及其显式证据映射。"""

    text: str
    evidence_ids: tuple[str, ...]


def validate_answer_contract(
    answer_text: str,
    claims: tuple[ClaimSupport, ...],
    allowed_evidence_ids: set[str],
) -> tuple[bool, str]:
    """机械校验最终答案的引用边界；语义蕴含仍由评测集单独验证。"""
    if not answer_text.strip() or not claims or not allowed_evidence_ids:
        return False, "empty_answer_contract"
    cited_ids = set(re.findall(r"\[([A-Za-z0-9][A-Za-z0-9_-]{0,80})\]", answer_text))
    required_ids = {evidence_id for claim in claims for evidence_id in claim.evidence_ids}
    if cited_ids - allowed_evidence_ids:
        return False, "unapproved_citation"
    if not required_ids.issubset(cited_ids):
        return False, "missing_required_citation"
    remaining_claims = {(claim.text.strip(), frozenset(claim.evidence_ids)) for claim in claims}
    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line_ids = frozenset(re.findall(r"\[([A-Za-z0-9][A-Za-z0-9_-]{0,80})\]", line))
        text = re.sub(r"\[([A-Za-z0-9][A-Za-z0-9_-]{0,80})\]", "", line)
        text = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", text).strip()
        text = text.rstrip("。；; ")
        matched = next(
            (
                item for item in remaining_claims
                if text == item[0].rstrip("。；; ") and line_ids == item[1]
            ),
            None,
        )
        if matched is None:
            return False, "claim_text_or_mapping_changed"
        remaining_claims.remove(matched)
    if remaining_claims:
        return False, "missing_approved_claim"
    return True, "validated"


def render_claim_contract_answer(claims: tuple[ClaimSupport, ...]) -> str:
    """模型输出不符合契约时，仅用已批准 Claim 和引用生成保守答案。"""
    lines = []
    for claim in claims:
        citations = " ".join(f"[{evidence_id}]" for evidence_id in claim.evidence_ids)
        lines.append(f"- {claim.text} {citations}".rstrip())
    return "\n".join(lines) or "当前证据不足以形成经过验证的回答。"


@dataclass(frozen=True)
class KnowledgeEvidence:
    """跨来源统一证据记录；自然语言总结不能替代 locator 和版本信息。"""

    evidence_id: str
    source_type: str
    content_or_fact: str
    locator: str | None = None
    project_id: str | None = None
    repository_id: str | None = None
    commit_id: str | None = None
    document_id: str | None = None
    evidence_kind: str = "retrieval"
    confidence: float | None = None
    derivation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """将证据转成 SSE、日志和最终回答可复用的普通字典。"""
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "content_or_fact": self.content_or_fact,
            "locator": self.locator,
            "project_id": self.project_id,
            "repository_id": self.repository_id,
            "commit_id": self.commit_id,
            "document_id": self.document_id,
            "evidence_kind": self.evidence_kind,
            "confidence": self.confidence,
            "derivation": self.derivation,
        }


@dataclass(frozen=True)
class KnowledgeToolResult:
    """RAG、Code Wiki 等子执行器必须遵守的统一结果协议。"""

    tool_call_id: str
    capability: str
    status: ToolStatus
    result: Any = None
    evidence: tuple[KnowledgeEvidence, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """保留 empty/partial/failed 等状态，避免把失败伪装成空结果。"""
        return {
            "tool_call_id": self.tool_call_id,
            "capability": self.capability,
            "status": self.status,
            "result": self.result,
            "evidence": [item.to_dict() for item in self.evidence],
            "warnings": list(self.warnings),
            "error": self.error,
            "usage": dict(self.usage),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class KnowledgeDecision:
    """Planner 或 Evidence Grader 返回的结构化决定。"""

    next_action: AgentAction
    route: Route | None = None
    intent: str | None = None
    sub_questions: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    queries: tuple[str, ...] = ()
    reason: str = ""
    scope_mode: ScopeMode = "soft_project"
    expansion_required: ExpansionRequired = "none"
    claims: tuple[ClaimSupport, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    public_update: str = ""
    target_project_id: str | None = None

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        allowed_actions: set[AgentAction],
        allowed_routes: set[Route] | None = None,
    ) -> "KnowledgeDecision":
        """校验模型 JSON；模型不能通过自由文本创造未知动作或知识源。"""
        action = str(payload.get("next_action") or "").strip()
        if action not in allowed_actions:
            raise ValueError(f"unsupported next_action: {action}")
        route_value = str(payload.get("route") or "").strip() or None
        if route_value is not None and (allowed_routes is None or route_value not in allowed_routes):
            raise ValueError(f"unsupported route: {route_value}")
        # 单独合法的 route/action 也可能互相矛盾，必须在进入状态机前拦截。
        if action == "retrieve_rag" and route_value not in {"rag", "hybrid"}:
            raise ValueError(f"route/action mismatch: {route_value}/{action}")
        if action == "query_code_wiki" and route_value not in {"wiki", "hybrid"}:
            raise ValueError(f"route/action mismatch: {route_value}/{action}")
        scope_mode = str(payload.get("scope_mode") or "soft_project")
        expansion_required = str(payload.get("expansion_required") or "none")
        if scope_mode not in {"strict_project", "soft_project", "candidate_projects"}:
            raise ValueError(f"unsupported scope_mode: {scope_mode}")
        if expansion_required not in {"none", "rag_allowed", "cross_project_required", "user_confirmation_required"}:
            raise ValueError(f"unsupported expansion_required: {expansion_required}")

        def normalized_list(name: str, limit: int) -> tuple[str, ...]:
            value = payload.get(name) or []
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
            return tuple(str(item).strip()[:300] for item in value if str(item).strip())[:limit]

        claim_items = payload.get("claims") or []
        if not isinstance(claim_items, list):
            raise ValueError("claims must be a list")
        claims = []
        for item in claim_items[:8]:
            if not isinstance(item, dict):
                raise ValueError("each claim must contain text and evidence_ids")
            text = str(item.get("text") or "").strip()[:300]
            evidence_ids = item.get("evidence_ids") or []
            if not text or not isinstance(evidence_ids, list):
                raise ValueError("each claim must contain text and evidence_ids")
            claims.append(ClaimSupport(text, tuple(str(value).strip() for value in evidence_ids if str(value).strip())[:20]))

        return cls(
            next_action=action,  # type: ignore[arg-type]
            route=route_value,  # type: ignore[arg-type]
            intent=str(payload.get("intent") or "").strip()[:200] or None,
            sub_questions=normalized_list("sub_questions", 5),
            missing_information=normalized_list("missing_information", 5),
            queries=normalized_list("queries", 3),
            reason=str(payload.get("reason") or "").strip()[:500],
            scope_mode=scope_mode,  # type: ignore[arg-type]
            expansion_required=expansion_required,  # type: ignore[arg-type]
            claims=tuple(claims),
            supporting_evidence_ids=normalized_list("supporting_evidence_ids", 20),
            public_update=str(payload.get("public_update") or "").strip()[:500],
            target_project_id=str(payload.get("target_project_id") or "").strip() or None,
        )


@dataclass
class KnowledgeAgentState:
    """统一入口的一次性编排状态，不保存跨请求记忆。

    这是控制器状态，不是模型私有思维链。模型只能提出动作；项目范围、预算、
    重复调用和最终停止状态都由代码维护，RAG/Code Wiki 则作为子执行器接入。
    """

    question: str
    request_id: str = field(default_factory=lambda: str(uuid4()))
    phase: KnowledgePhase = "plan"
    project_id: str | None = None
    commit_id: str | None = None
    scope_mode: ScopeMode = "soft_project"
    expansion_required: ExpansionRequired = "none"
    cross_project_policy: CrossProjectPolicy = "deny"
    allowed_project_ids: set[str] = field(default_factory=set)
    authorized_projects: list[dict[str, Any]] = field(default_factory=list)
    max_rounds: int = 10
    max_tool_calls: int = 20
    used_rounds: int = 0
    used_tool_calls: int = 0
    visited_actions: set[str] = field(default_factory=set)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    route: Route | None = None
    intent: str | None = None
    sub_questions: list[str] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
    executed_actions: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    next_action: AgentAction | None = None
    answer_ready: bool = False
    refused: bool = False
    stop_reason: str | None = None
    decision_history: list[KnowledgeDecision] = field(default_factory=list)

    def begin_round(self) -> int:
        """开始一轮顶层决策；轮数耗尽时立即进入拒答，防止循环无限运行。"""
        if self.answer_ready or self.refused:
            return self.used_rounds
        if self.used_rounds >= self.max_rounds:
            self.mark_refused("max_rounds_exceeded")
            self.phase = "refuse"
            return self.used_rounds
        self.used_rounds += 1
        self.phase = "execute"
        return self.used_rounds

    def admit_action(self, action: str, signature: str, *, requires_project: bool = False) -> None:
        """执行工具前做统一的范围、预算和重复调用校验。"""
        if self.answer_ready or self.refused:
            raise RuntimeError("knowledge loop is already closed")
        if self.used_tool_calls >= self.max_tool_calls:
            self.mark_refused("max_tool_calls_exceeded")
            self.phase = "refuse"
            raise RuntimeError("knowledge tool-call budget exceeded")
        if requires_project and not self.project_id:
            self.mark_refused("project_scope_required")
            self.phase = "refuse"
            raise ValueError("project scope is required for this action")
        if signature in self.visited_actions:
            raise ValueError(f"duplicate knowledge action: {signature}")
        self.visited_actions.add(signature)
        self.used_tool_calls += 1
        self.add_action(action, signature=signature, tool_call_no=self.used_tool_calls)

    def record_tool_result(self, result: KnowledgeToolResult) -> None:
        """记录统一工具结果，并把有效证据纳入顶层状态。"""
        # 子执行器返回的证据必须继承顶层范围，防止跨项目或跨 Commit 串证据。
        for item in result.evidence:
            if item.source_type == "code_wiki" and (not item.project_id or not item.commit_id):
                raise ValueError("code evidence project and commit scope are required")
            if item.project_id and item.project_id not in self.allowed_project_ids and item.project_id != self.project_id:
                raise ValueError("evidence project scope mismatch")
            # 锚点项目是默认项目，不是硬编码的唯一项目；其他项目必须来自授权候选集合。
            if self.commit_id and item.project_id == self.project_id and item.commit_id and item.commit_id != self.commit_id:
                raise ValueError("evidence commit scope mismatch")
        self.tool_results.append(result.to_dict())
        self.add_evidence([item.to_dict() for item in result.evidence])
        self.phase = "observe"

    def record_child_tool_calls(self, count: int) -> None:
        """把子 Agent 的真实工具调用并入顶层预算，避免聚合动作掩盖实际消耗。"""
        additional = max(0, int(count))
        self.used_tool_calls += additional
        if self.used_tool_calls > self.max_tool_calls:
            self.mark_refused("max_tool_calls_exceeded")
            raise RuntimeError("knowledge tool-call budget exceeded")

    def record_child_rounds(self, count: int) -> None:
        """把子 Agent 的完成轮次并入顶层轮次，统一安全熔断和可观测计数。"""
        self.used_rounds += max(0, int(count))
        if self.used_rounds > self.max_rounds:
            self.mark_refused("max_rounds_exceeded")
            raise RuntimeError("knowledge round budget exceeded")

    def replan(self, reason: str, missing_information: list[str] | None = None) -> None:
        """进入下一轮规划；原因和缺口会进入审计快照而不是隐藏在日志里。"""
        self.phase = "replan"
        self.stop_reason = None
        self.missing_information = list(missing_information or [])
        self.add_action("replan", reason=reason, missing_information=self.missing_information)

    def set_route(self, route: Route, intent: str | None = None) -> None:
        """记录 Router 的结果；路由是编排状态，不是某个检索器的内部状态。"""
        self.route = route
        self.intent = intent or route

    def set_scope(
        self,
        scope_mode: ScopeMode,
        expansion_required: ExpansionRequired,
        *,
        allowed_project_ids: set[str] | None = None,
    ) -> None:
        """记录问题范围分类；扩大代码范围必须经过显式策略校验。"""
        self.scope_mode = scope_mode
        self.expansion_required = expansion_required
        if allowed_project_ids is not None:
            self.allowed_project_ids = set(allowed_project_ids)

    def apply_scope_suggestion(self, decision: KnowledgeDecision) -> None:
        """把模型的范围建议收敛到服务端策略，模型不能直接扩大授权范围。"""
        if decision.scope_mode == "strict_project" and self.project_id:
            self.set_scope("strict_project", "none")
            return
        if decision.scope_mode == "candidate_projects":
            # 候选集合由服务端预先计算；模型只能选择集合中的项目，不能自行添加 ID。
            self.set_scope(
                "candidate_projects",
                "cross_project_required" if self.cross_project_policy in {"allow_authorized", "candidate_set"}
                else "user_confirmation_required",
            )
            return
        self.set_scope("soft_project", "rag_allowed" if self.project_id else "none")

    def can_expand_to_project(self, project_id: str) -> bool:
        """判断模型建议的跨项目查询是否得到代码侧授权。"""
        return self.cross_project_policy in {"allow_authorized", "candidate_set"} and (
            project_id in self.allowed_project_ids
        )

    def can_answer(self, decision: KnowledgeDecision) -> bool:
        """Answer Gate：模型引用的证据必须真实存在，且至少覆盖一条有效证据。"""
        successful_ids = {
            str(evidence.get("evidence_id"))
            for result in self.tool_results if result.get("status") == "success"
            for evidence in result.get("evidence", []) if evidence.get("evidence_id")
        }
        available = {
            str(item.get("evidence_id"))
            for item in self.evidence
            if item.get("evidence_id") and item.get("locator") and str(item.get("content_or_fact") or "").strip()
            and str(item.get("evidence_id")) in successful_ids
            and (item.get("source_type") != "code_wiki" or (
                item.get("project_id") in (self.allowed_project_ids or {self.project_id}) and item.get("commit_id")
            ))
        }
        requested = set(decision.supporting_evidence_ids)
        mapped = set()
        for claim in decision.claims:
            claim_ids = set(claim.evidence_ids)
            if not claim.text or not claim_ids or not claim_ids.issubset(available):
                return False
            mapped.update(claim_ids)
        return bool(decision.claims and available and requested and requested.issubset(available) and mapped == requested)

    def approved_claims(self) -> tuple[ClaimSupport, ...]:
        """返回最近一次通过状态机并进入 answer 的 Claim 契约。"""
        for decision in reversed(self.decision_history):
            if decision.next_action == "answer":
                return decision.claims
        return ()

    def bounded_answer_material(self, max_chars: int) -> tuple[str, tuple[ClaimSupport, ...], set[str]]:
        """只保留完整证据能装入预算的 Claim，避免 Gate 契约与最终上下文脱节。"""
        evidence_by_id = {
            str(item.get("evidence_id")): item
            for item in self.evidence if item.get("evidence_id")
        }
        selected_claims: list[ClaimSupport] = []
        selected_ids: set[str] = set()
        parts: list[str] = []
        used_chars = 0
        for claim in self.approved_claims():
            new_ids = [item_id for item_id in claim.evidence_ids if item_id not in selected_ids]
            new_parts = []
            complete = True
            for evidence_id in new_ids:
                item = evidence_by_id.get(evidence_id)
                if not item:
                    complete = False
                    break
                new_parts.append(
                    f"[{evidence_id}] 类型={item['source_type']} 定位={item.get('locator')}\n"
                    f"{item.get('content_or_fact', '')}"
                )
            added_chars = sum(len(part) for part in new_parts) + (2 * len(new_parts))
            if complete and used_chars + added_chars <= max_chars:
                selected_claims.append(claim)
                selected_ids.update(new_ids)
                parts.extend(new_parts)
                used_chars += added_chars
        return "\n\n".join(parts), tuple(selected_claims), selected_ids

    def action_attempt_count(self, action: str) -> int:
        """按完整动作签名前缀计数，允许同一能力使用不同查询补证。"""
        return sum(1 for signature in self.visited_actions if signature.startswith(f"{action}:"))

    def add_action(self, action: str, **details: Any) -> None:
        """记录一次真实执行动作，供 SSE、审计和后续 Replan 使用。"""
        self.executed_actions.append({"action": action, **details})

    def add_evidence(self, items: list[dict[str, Any]]) -> None:
        """追加证据并保留原始结构，避免用候选数量冒充证据质量。"""
        self.evidence.extend(items)

    def apply_decision(self, decision: KnowledgeDecision) -> None:
        """应用经过校验的 Agent 决定，并维护互斥的完成状态。"""
        self.decision_history.append(decision)
        if decision.route is not None:
            self.set_route(decision.route, decision.intent)
        elif decision.intent:
            self.intent = decision.intent
        if decision.sub_questions:
            self.sub_questions = list(decision.sub_questions)
        self.missing_information = list(decision.missing_information)
        self.next_action = decision.next_action
        if decision.next_action == "answer":
            self.answer_ready = True
            self.refused = False
            self.phase = "finish"
            self.stop_reason = "evidence_sufficient"
        elif decision.next_action == "refuse":
            self.answer_ready = False
            self.refused = True
            self.phase = "refuse"
            self.stop_reason = decision.reason or "evidence_insufficient"
        else:
            self.answer_ready = False
            self.refused = False
            self.phase = "replan" if decision.next_action == "replan" else "execute"
            self.stop_reason = None

    def mark_answer_ready(self) -> None:
        self.answer_ready = True
        self.refused = False
        self.phase = "finish"
        self.next_action = "answer"
        self.stop_reason = "evidence_sufficient"

    def mark_refused(self, reason: str) -> None:
        self.refused = True
        self.answer_ready = False
        self.phase = "refuse"
        self.next_action = "refuse"
        self.stop_reason = reason

    def snapshot(self) -> dict[str, Any]:
        """返回可观测状态，不暴露模型私有思维链。"""
        return {
            "request_id": self.request_id,
            "question": self.question,
            "phase": self.phase,
            "scope": {"project_id": self.project_id, "commit_id": self.commit_id},
            "query_scope": {
                "mode": self.scope_mode,
                "expansion_required": self.expansion_required,
                "cross_project_policy": self.cross_project_policy,
                "allowed_project_ids": sorted(self.allowed_project_ids),
                "authorized_projects": list(self.authorized_projects),
            },
            "budget": {
                "max_rounds": self.max_rounds,
                "max_tool_calls": self.max_tool_calls,
                "used_rounds": self.used_rounds,
                "used_tool_calls": self.used_tool_calls,
            },
            "route": self.route,
            "intent": self.intent,
            "sub_questions": list(self.sub_questions),
            "plan": list(self.plan),
            "executed_actions": list(self.executed_actions),
            "evidence_count": len(self.evidence),
            "evidence": [
                {**item, "content_or_fact": str(item.get("content_or_fact") or "")[:600]}
                for item in self.evidence[-20:]
            ],
            "tool_result_count": len(self.tool_results),
            "tool_results": list(self.tool_results),
            "missing_information": list(self.missing_information),
            "conflicts": list(self.conflicts),
            "next_action": self.next_action,
            "answer_ready": self.answer_ready,
            "refused": self.refused,
            "stop_reason": self.stop_reason,
            "decision_count": len(self.decision_history),
        }

    def public_snapshot(self) -> dict[str, Any]:
        """返回浏览器可见的运行摘要，不携带证据正文或原始工具载荷。"""
        return {
            "request_id": self.request_id,
            "phase": self.phase,
            "scope": {"project_id": self.project_id, "commit_id": self.commit_id},
            "budget": {
                "max_rounds": self.max_rounds,
                "max_tool_calls": self.max_tool_calls,
                "used_rounds": self.used_rounds,
                "used_tool_calls": self.used_tool_calls,
            },
            "route": self.route,
            "intent": self.intent,
            "evidence_count": len(self.evidence),
            "tool_result_count": len(self.tool_results),
            "missing_information": list(self.missing_information),
            "next_action": self.next_action,
            "answer_ready": self.answer_ready,
            "refused": self.refused,
            "stop_reason": self.stop_reason,
            "decision_count": len(self.decision_history),
        }
