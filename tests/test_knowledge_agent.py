"""顶层 Knowledge Agent 状态契约测试。"""

import pytest
import asyncio
from types import SimpleNamespace

from app import llm
from app import main as main_module
from app.knowledge_agent import ClaimSupport, KnowledgeAgentState, KnowledgeDecision, KnowledgeEvidence, KnowledgeToolResult
from app.main import KnowledgeStreamRequest, _split_code_payloads_by_citation, _with_agentic_dispatch_trace, agentic_knowledge_stream
from fastapi.responses import StreamingResponse


def test_knowledge_agent_keeps_route_and_evidence_separate_from_retrieval_state():
    state = KnowledgeAgentState("服务是否符合规范？")

    state.set_route("hybrid", "code_and_policy_comparison")
    state.add_action("retrieve_rag", query="服务规范")
    state.add_evidence([{"source": "policy.md", "content": "..."}])

    snapshot = state.snapshot()
    assert snapshot["route"] == "hybrid"
    assert snapshot["intent"] == "code_and_policy_comparison"
    assert snapshot["evidence_count"] == 1
    assert snapshot["executed_actions"][0]["action"] == "retrieve_rag"


def test_knowledge_agent_can_finish_or_refuse_explicitly():
    answer_state = KnowledgeAgentState("问题")
    answer_state.mark_answer_ready()
    assert answer_state.answer_ready is True
    assert answer_state.next_action == "answer"

    refuse_state = KnowledgeAgentState("问题")
    refuse_state.mark_refused("evidence_insufficient")
    assert refuse_state.refused is True
    assert refuse_state.next_action == "refuse"


def test_structured_decision_rejects_unknown_actions_and_routes():
    with pytest.raises(ValueError):
        KnowledgeDecision.from_payload(
            {"route": "internet", "next_action": "browse_everything"},
            allowed_actions={"retrieve_rag"},
            allowed_routes={"rag"},
        )


def test_structured_decision_rejects_route_action_mismatch():
    with pytest.raises(ValueError, match="route/action mismatch"):
        KnowledgeDecision.from_payload(
            {"route": "rag", "next_action": "query_code_wiki"},
            allowed_actions={"retrieve_rag", "query_code_wiki"},
            allowed_routes={"rag", "wiki", "hybrid"},
        )


def test_query_scope_requires_explicit_cross_project_policy():
    state = KnowledgeAgentState("比较两个项目", project_id="repo-1")
    state.set_scope("candidate_projects", "cross_project_required", allowed_project_ids={"repo-2"})
    assert state.can_expand_to_project("repo-2") is False
    state.cross_project_policy = "candidate_set"
    assert state.can_expand_to_project("repo-2") is True
    assert state.can_expand_to_project("repo-3") is False


def test_model_scope_suggestion_cannot_open_cross_project_access():
    state = KnowledgeAgentState("比较项目", project_id="repo-1", cross_project_policy="deny")
    decision = KnowledgeDecision(
        next_action="replan", scope_mode="candidate_projects",
        expansion_required="cross_project_required",
    )
    state.apply_scope_suggestion(decision)
    assert state.scope_mode == "candidate_projects"
    assert state.expansion_required == "user_confirmation_required"
    assert state.can_expand_to_project("repo-2") is False


def test_answer_gate_requires_existing_evidence_ids():
    state = KnowledgeAgentState("问题")
    state.record_tool_result(KnowledgeToolResult(
        tool_call_id="rag-1", capability="retrieve_rag", status="success",
        evidence=(KnowledgeEvidence(
            evidence_id="E1", source_type="rag", content_or_fact="事实", locator="文档.md#章节",
        ),),
    ))
    assert state.can_answer(KnowledgeDecision(next_action="answer")) is False
    assert state.can_answer(KnowledgeDecision(next_action="answer", supporting_evidence_ids=("E2",))) is False
    assert state.can_answer(KnowledgeDecision(
        next_action="answer", claims=(ClaimSupport("事实成立", ("E1",)),), supporting_evidence_ids=("E1",),
    )) is True

    missing_locator = KnowledgeAgentState("问题")
    missing_locator.add_evidence([{"evidence_id": "E1", "source_type": "rag", "content_or_fact": "事实"}])
    assert missing_locator.can_answer(KnowledgeDecision(
        next_action="answer", claims=(ClaimSupport("事实成立", ("E1",)),), supporting_evidence_ids=("E1",),
    )) is False

    partial = KnowledgeAgentState("问题")
    partial.record_tool_result(KnowledgeToolResult(
        tool_call_id="rag-2", capability="retrieve_rag", status="partial",
        evidence=(KnowledgeEvidence(
            evidence_id="E2", source_type="rag", content_or_fact="可能相关", locator="文档.md#其他",
        ),),
    ))
    assert partial.can_answer(KnowledgeDecision(
        next_action="answer", claims=(ClaimSupport("事实成立", ("E2",)),), supporting_evidence_ids=("E2",),
    )) is False


def test_apply_decision_keeps_answer_and_refuse_mutually_exclusive():
    state = KnowledgeAgentState("问题")
    state.apply_decision(KnowledgeDecision(next_action="refuse", reason="没有证据"))
    assert state.refused is True
    assert state.answer_ready is False

    state.apply_decision(KnowledgeDecision(next_action="answer"))
    assert state.answer_ready is True
    assert state.refused is False
    assert state.snapshot()["decision_count"] == 2


def test_answer_state_keeps_the_gate_approved_claim_contract():
    state = KnowledgeAgentState("问题")
    approved = ClaimSupport("配置位于 compose.yaml", ("E1",))

    state.apply_decision(KnowledgeDecision(
        next_action="answer", claims=(approved,), supporting_evidence_ids=("E1",),
    ))

    assert state.approved_claims() == (approved,)


def test_answer_budget_keeps_only_claims_with_complete_evidence():
    state = KnowledgeAgentState("问题")
    first = ClaimSupport("第一条", ("E1",))
    second = ClaimSupport("第二条", ("E2",))
    state.add_evidence([
        {"evidence_id": "E1", "source_type": "rag", "locator": "a.md", "content_or_fact": "A" * 20},
        {"evidence_id": "E2", "source_type": "rag", "locator": "b.md", "content_or_fact": "B" * 200},
    ])
    state.apply_decision(KnowledgeDecision(
        next_action="answer", claims=(first, second,), supporting_evidence_ids=("E1", "E2"),
    ))

    context, claims, evidence_ids = state.bounded_answer_material(100)

    assert claims == (first,)
    assert evidence_ids == {"E1"}
    assert "[E1]" in context
    assert "[E2]" not in context


def test_snapshot_exposes_only_server_supplied_project_catalog():
    state = KnowledgeAgentState(
        "比较两个项目", allowed_project_ids={"repo-1"},
        authorized_projects=[{"project_id": "repo-1", "project_name": "orders", "languages": ["go"]}],
    )

    assert state.snapshot()["query_scope"]["authorized_projects"] == [
        {"project_id": "repo-1", "project_name": "orders", "languages": ["go"]}
    ]


def _model_client(content: str):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    completions = SimpleNamespace(create=lambda **_kwargs: response)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_planner_parses_fenced_json_within_server_route_allowlist(monkeypatch):
    monkeypatch.setattr(llm, "_client", lambda *_args, **_kwargs: _model_client(
        '```json\n{"route":"rag","intent":"policy","sub_questions":["规则是什么"],"next_action":"retrieve_rag","reason":"需要文档证据"}\n```'
    ))

    decision = llm.plan_knowledge_query("规则是什么？", {"rag"})
    assert decision.route == "rag"
    assert decision.next_action == "retrieve_rag"
    assert decision.sub_questions == ("规则是什么",)


def test_evidence_grader_cannot_replan_after_budget_is_closed(monkeypatch):
    monkeypatch.setattr(llm, "_client", lambda *_args, **_kwargs: _model_client(
        '{"next_action":"replan","missing_information":["配置"],"queries":["配置文件"],"reason":"缺少证据"}'
    ))

    with pytest.raises(ValueError):
        llm.grade_rag_evidence("如何配置？", [{"content": "片段"}], can_replan=False)


def test_knowledge_stream_request_supports_agentic_modes():
    assert KnowledgeStreamRequest(message="问题").mode == "auto"
    assert KnowledgeStreamRequest(message="问题", mode="hybrid").mode == "hybrid"


def test_agentic_dispatch_trace_precedes_executor_events():
    async def downstream():
        yield 'event: done\ndata: {"answer":"ok"}\n\n'

    response = _with_agentic_dispatch_trace(
        StreamingResponse(downstream(), media_type="text/event-stream"),
        KnowledgeDecision(
            next_action="query_code_wiki", route="wiki", intent="code_question",
            sub_questions=("定位实现",), reason="需要源码证据",
        ),
        planner_fallback=False,
    )
    async def collect():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return chunks

    chunks = asyncio.run(collect())
    content = "".join(chunks)
    assert content.index("knowledge_planner") < content.index('event: done')
    assert '"route": "wiki"' in content
    assert '"phase": "investigation"' in content


def test_shared_loop_enforces_round_and_duplicate_action_budget():
    state = KnowledgeAgentState("问题", max_rounds=1, max_tool_calls=2)
    assert state.begin_round() == 1
    state.admit_action("retrieve_rag", "rag:问题")

    with pytest.raises(ValueError, match="duplicate knowledge action"):
        state.admit_action("retrieve_rag", "rag:问题")

    budget_state = KnowledgeAgentState("问题", max_tool_calls=1)
    budget_state.begin_round()
    budget_state.admit_action("retrieve_rag", "rag:问题")
    with pytest.raises(RuntimeError, match="budget exceeded"):
        budget_state.admit_action("retrieve_rag", "rag:另一个问题")

    finished = KnowledgeAgentState("问题", max_rounds=1)
    assert finished.begin_round() == 1
    assert finished.begin_round() == 1
    assert finished.refused is True
    assert finished.stop_reason == "max_rounds_exceeded"


def test_shared_loop_records_typed_tool_result_and_scope():
    state = KnowledgeAgentState("代码问题", project_id="repo-1")
    state.begin_round()
    state.admit_action("query_code_wiki", "codewiki:repo-1:代码问题", requires_project=True)
    state.record_tool_result(KnowledgeToolResult(
        tool_call_id="call-1", capability="query_codewiki", status="partial",
        evidence=(KnowledgeEvidence(
            evidence_id="code-1", source_type="code_wiki", content_or_fact="函数定义",
            locator="app/main.py:10", project_id="repo-1", commit_id="abc123",
        ),),
    ))

    snapshot = state.snapshot()
    assert snapshot["scope"]["project_id"] == "repo-1"
    assert snapshot["budget"]["used_tool_calls"] == 1
    assert snapshot["tool_result_count"] == 1
    assert snapshot["evidence_count"] == 1
    assert snapshot["tool_results"][0]["status"] == "partial"


def test_shared_loop_rejects_evidence_from_another_project_or_commit():
    state = KnowledgeAgentState("代码问题", project_id="repo-1", commit_id="abc123")
    state.begin_round()
    state.admit_action("query_code_wiki", "codewiki:repo-1:代码问题", requires_project=True)

    with pytest.raises(ValueError, match="project scope mismatch"):
        state.record_tool_result(KnowledgeToolResult(
            tool_call_id="call-1", capability="query_codewiki", status="success",
            evidence=(KnowledgeEvidence(
                evidence_id="bad-project", source_type="code_wiki", content_or_fact="事实",
                locator="x.py:1", project_id="repo-2", commit_id="abc123",
            ),),
        ))

    with pytest.raises(ValueError, match="commit scope mismatch"):
        state.record_tool_result(KnowledgeToolResult(
            tool_call_id="call-2", capability="query_codewiki", status="success",
            evidence=(KnowledgeEvidence(
                evidence_id="bad-commit", source_type="code_wiki", content_or_fact="事实",
                locator="x.py:1", project_id="repo-1", commit_id="def456",
            ),),
        ))

    with pytest.raises(ValueError, match="project and commit scope are required"):
        state.record_tool_result(KnowledgeToolResult(
            tool_call_id="call-3", capability="query_codewiki", status="success",
            evidence=(KnowledgeEvidence(
                evidence_id="missing-scope", source_type="code_wiki", content_or_fact="事实",
                locator="x.py:1",
            ),),
        ))


def test_agentic_stream_replans_after_empty_result_and_emits_one_done(monkeypatch):
    decisions = iter([
        KnowledgeDecision(next_action="retrieve_rag", route="rag", queries=("第一次查询",), public_update="先查文档"),
        KnowledgeDecision(next_action="retrieve_rag", route="rag", queries=("改写查询",), public_update="换一种查询补证"),
        KnowledgeDecision(
            next_action="answer", route="rag", claims=(ClaimSupport("事实成立", ("rag-2-E1",)),),
            supporting_evidence_ids=("rag-2-E1",), public_update="证据已覆盖问题",
        ),
    ])
    monkeypatch.setattr(main_module, "decide_next_knowledge_action", lambda *_args, **_kwargs: next(decisions))

    async def fake_rag(query, tool_call_id, _scope=None):
        if query == "第一次查询":
            return KnowledgeToolResult(tool_call_id=tool_call_id, capability="retrieve_rag", status="empty"), []
        evidence = KnowledgeEvidence(
            evidence_id=f"{tool_call_id}-E1", source_type="rag", content_or_fact="可验证事实",
            locator="规范.md#事实",
        )
        return KnowledgeToolResult(
            tool_call_id=tool_call_id, capability="retrieve_rag", status="success", evidence=(evidence,),
        ), [{"id": evidence.evidence_id, "type": "rag", "ref": evidence.locator}]

    monkeypatch.setattr(main_module, "_retrieve_rag_capability", fake_rag)
    monkeypatch.setattr(main_module, "stream_answer", lambda *_args: iter([
        {"type": "delta", "content": "最终回答"}, {"type": "usage", "completion_tokens": 4},
    ]))

    async def collect():
        scope = main_module.AccessScope("user-1", "workspace-1", "测试空间", None, frozenset(), "owner")
        response = await agentic_knowledge_stream("问题", None, scope)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    content = asyncio.run(collect())
    assert "retrieve_rag:-:改写查询" in content
    assert content.count("event: done") == 1
    assert "最终回答" in content


def test_agentic_stream_reports_authorization_revocation_without_retry(monkeypatch):
    """长请求中途撤权不能伪装成可重试的模型故障。"""
    monkeypatch.setattr(
        main_module,
        "refresh_request_scope",
        lambda _scope: (_ for _ in ()).throw(PermissionError("membership revoked")),
    )

    async def collect():
        scope = main_module.AccessScope("user-1", "workspace-1", "测试空间", None, frozenset(), "owner")
        response = await agentic_knowledge_stream("问题", None, scope)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    content = asyncio.run(collect())
    assert '"error_code": "AUTHORIZATION_REVOKED"' in content
    assert '"retryable": false' in content
    assert "KNOWLEDGE_AGENT_FAILED" not in content


def test_nested_agent_preserves_authorization_revocation_semantics():
    """父 Agent 解析子流时必须继续走撤权分支，而不是改成可重试故障。"""
    with pytest.raises(PermissionError, match="权限已撤销"):
        main_module._raise_child_agent_error(
            {"message": "权限已撤销", "error_code": "AUTHORIZATION_REVOKED"},
            "子 Agent 失败",
        )
    with pytest.raises(RuntimeError, match="模型暂时不可用"):
        main_module._raise_child_agent_error(
            {"message": "模型暂时不可用", "error_code": "CODE_AGENT_FAILED"},
            "子 Agent 失败",
        )


def test_code_payloads_are_split_by_fact_citation():
    mapped = _split_code_payloads_by_citation([{
        "capability": "find_architecture",
        "result": {"facts": [
            {"name": "Kafka", "citation": "[C1]", "path": "a.yml"},
            {"name": "Redis", "citation": "[C2]", "path": "b.yml"},
        ]},
    }])
    assert "Kafka" in "".join(mapped["C1"])
    assert "Redis" not in "".join(mapped["C1"])
    assert "Redis" in "".join(mapped["C2"])
