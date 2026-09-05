"""文档 RAG 的自适应检索状态。

这个模块只负责文档检索内部的查询改写和重试，不负责选择知识源，也不代表顶层
Knowledge Agent。跨文档 RAG、Code Wiki 和 Hybrid 的任务编排由 knowledge_agent 管理。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RetrievalPlanStep:
    """一个可观察、可重试的文档检索步骤。"""

    step_id: str
    objective: str
    query: str
    status: str = "pending"
    result_count: int = 0
    note: str = ""


@dataclass
class AdaptiveRetrievalState:
    """单次文档 RAG 请求的有限查询改写状态。"""

    question: str
    max_replans: int = 2
    steps: list[RetrievalPlanStep] = field(default_factory=list)
    replan_count: int = 0
    evidence_count: int = 0

    def create(self) -> RetrievalPlanStep:
        """创建首个精确检索计划，保留用户原问题作为最高优先级查询。"""
        step = RetrievalPlanStep("retrieve_1", "先用原问题检索最相关证据", self.question.strip())
        self.steps.append(step)
        return step

    def complete(self, step: RetrievalPlanStep, result_count: int, note: str = "") -> None:
        step.status = "completed"
        step.result_count = result_count
        step.note = note
        self.evidence_count += max(0, result_count)

    def replan(self) -> RetrievalPlanStep | None:
        """证据不足时生成通用查询变体；去重后最多重规划两次。"""
        if self.replan_count >= self.max_replans:
            return None
        existing = {step.query.casefold() for step in self.steps}
        text = re.sub(r"[^\w\u4e00-\u9fff\s./:-]", " ", self.question)
        tokens = [token for token in text.split() if len(token) > 1]
        candidates = []
        if tokens:
            candidates.append(" ".join(tokens[:12]))
            if len(tokens) > 2:
                candidates.append(" ".join(tokens[:-1]))
                candidates.append(" ".join(tokens[1:]))
        if len(tokens) >= 2:
            candidates.append(" ".join(tokens[-8:]))
        query = next((item for item in candidates if item.casefold() not in existing), None)
        if not query:
            return None
        self.replan_count += 1
        step = RetrievalPlanStep(
            f"retrieve_{len(self.steps) + 1}",
            "上一轮证据不足，改变查询表达后再次检索",
            query,
        )
        self.steps.append(step)
        return step

    def replan_with_queries(self, queries: list[str]) -> RetrievalPlanStep | None:
        """使用模型给出的候选查询重规划；候选仍受本地去重和次数预算约束。"""
        if self.replan_count >= self.max_replans:
            return None
        existing = {step.query.casefold() for step in self.steps}
        query = next((item.strip() for item in queries if item.strip().casefold() not in existing), None)
        if not query:
            return None
        self.replan_count += 1
        step = RetrievalPlanStep(
            f"retrieve_{len(self.steps) + 1}",
            "上一轮证据不足，依据失败原因重写查询后再次检索",
            query[:300],
        )
        self.steps.append(step)
        return step

    def snapshot(self) -> list[dict]:
        """返回前端可展示的检索计划状态。"""
        return [
            {"step_id": step.step_id, "objective": step.objective, "query": step.query, "status": step.status, "result_count": step.result_count, "note": step.note}
            for step in self.steps
        ]
