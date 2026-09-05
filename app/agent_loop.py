"""受限 Agent Loop 的状态和停止规则。

模型负责提出工具请求，本模块负责控制请求是否值得执行以及循环是否继续。
这样“需要重复调用”和“陷入重复循环”可以被明确区分，而不是只依赖最大轮数。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LoopTransition:
    """Loop Engine 对主执行器返回的唯一状态转移结果。"""

    action: str
    reason: str | None
    completed_rounds: int
    tool_calls: int
    unchanged_rounds: int

    @property
    def should_continue(self) -> bool:
        return self.action == "continue"


@dataclass
class AgentLoopState:
    """一次 Agent 运行的短期状态，不跨请求持久化。"""

    max_rounds: int = 10
    max_tool_calls: int = 12
    signatures: set[tuple[str, str]] = field(default_factory=set)
    result_fingerprints: set[str] = field(default_factory=set)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    unchanged_rounds: int = 0
    last_progress: bool = True
    round_progress: bool = False
    stop_reason: str | None = None
    completed_rounds: int = 0

    def call_signature(self, name: str, arguments: dict) -> tuple[str, str]:
        """规范化工具参数，保证字典顺序不会制造假差异。"""
        return name, json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)

    def admit(self, name: str, arguments: dict) -> tuple[bool, str | None]:
        """判断工具请求是否可以执行。"""
        signature = self.call_signature(name, arguments)
        if self.tool_calls >= self.max_tool_calls:
            return False, "tool_budget_exhausted"
        # 全量文件资产是本轮固定快照；重复查看它不会产生新证据。
        # 空查询的 offset=0 是固定的全量首页；后续 offset 属于新的分页证据，必须允许执行。
        if name == "list_project_files" and any(item["name"] == name for item in self.tool_history) and not str(arguments.get("query", "")).strip() and int(arguments.get("offset", 0) or 0) == 0:
            return False, "project_files_already_inspected"
        if signature in self.signatures:
            return False, "same_query_repeated"
        # 连续无进展后，要求模型先结束或切换到未使用的证据路径。
        if self.unchanged_rounds >= 2:
            return False, "no_new_evidence"
        self.signatures.add(signature)
        self.tool_calls += 1
        return True, None

    def _fingerprint(self, name: str, result: dict) -> str:
        """只对工具结果做稳定哈希，避免把原始大文本重复塞进状态。"""
        payload = {key: value for key, value in result.items() if key not in {"source", "numbered_content"}}
        if "source" in result:
            payload["source_length"] = len(str(result["source"]))
        if "numbered_content" in result:
            payload["source_length"] = len(str(result["numbered_content"]))
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(f"{name}:{encoded}".encode("utf-8")).hexdigest()

    @staticmethod
    def _evidence_count(result: dict) -> int:
        """估算本次结果能贡献的事实数量，不把工具错误算作进展。"""
        if result.get("error"):
            return 0
        total = 0
        for key in ("items", "nodes", "facts", "links", "edges", "outgoing", "incoming"):
            value = result.get(key)
            if isinstance(value, list):
                total += len(value)
        if result.get("source") or result.get("numbered_content"):
            total += 1
        return total or (1 if result.get("count") else 0)

    def record(self, name: str, arguments: dict, result: dict) -> dict:
        """记录工具结果并返回进展信息。"""
        fingerprint = self._fingerprint(name, result)
        is_new_result = fingerprint not in self.result_fingerprints
        evidence_count = self._evidence_count(result)
        progress = is_new_result and evidence_count > 0
        self.result_fingerprints.add(fingerprint)
        self.last_progress = progress
        # 一轮可以执行多个工具；这里只聚合本轮进展，不能把空工具数当成空轮数。
        self.round_progress = self.round_progress or progress
        event = {
            "name": name, "arguments": arguments, "evidence_count": evidence_count,
            "new_evidence": progress, "result_fingerprint": fingerprint,
        }
        self.tool_history.append(event)
        return event

    def finish(self, reason: str, round_index: int | None = None) -> LoopTransition:
        """由 Agent 或执行器显式结束调查，并固化唯一停止原因。"""
        if round_index is not None:
            self.completed_rounds = max(self.completed_rounds, round_index + 1)
        if not self.stop_reason:
            self.stop_reason = reason
        return self._transition("stop", self.stop_reason)

    def evaluate_round(
        self,
        round_index: int,
        executed_calls: int,
        blocked_reasons: list[str] | None = None,
    ) -> LoopTransition:
        """根据本轮结果统一决定继续或停止，主循环不得再复制这些规则。"""
        self.completed_rounds = max(self.completed_rounds, round_index + 1)
        blocked_reasons = blocked_reasons or []

        if self.stop_reason:
            return self._transition("stop", self.stop_reason)
        if blocked_reasons and executed_calls == 0:
            return self.finish(blocked_reasons[0])
        if executed_calls > 0:
            self.unchanged_rounds = 0 if self.round_progress else self.unchanged_rounds + 1
        self.round_progress = False
        if self.unchanged_rounds >= 2:
            return self.finish("no_new_evidence")
        if self.tool_calls >= self.max_tool_calls:
            return self.finish("tool_budget_exhausted")
        if self.completed_rounds >= self.max_rounds:
            return self.finish("round_budget_exhausted")
        return self._transition("continue", None)

    def _transition(self, action: str, reason: str | None) -> LoopTransition:
        return LoopTransition(
            action=action,
            reason=reason,
            completed_rounds=self.completed_rounds,
            tool_calls=self.tool_calls,
            unchanged_rounds=self.unchanged_rounds,
        )
