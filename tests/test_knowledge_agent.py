"""顶层 Knowledge Agent 状态契约测试。"""

import pytest
import asyncio
import json
from types import SimpleNamespace

from app import llm
from app import main as main_module
from app.knowledge_agent import (
    ClaimSupport,
    KnowledgeAgentState,
    KnowledgeDecision,
    KnowledgeEvidence,
    KnowledgeToolResult,
    render_claim_contract_answer,
    validate_answer_contract,
)
from app.main import (
    KnowledgeStreamRequest,
    _split_code_payloads_by_citation,
    _with_agentic_dispatch_trace,
    agentic_knowledge_stream,
    code_wiki_agent_stream,
    knowledge_stream,
    refresh_answer_authorization,
)
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


def test_final_answer_contract_rejects_missing_and_unknown_citations():
    claims = (ClaimSupport("配置位于 compose.yaml", ("E1", "E2")),)
    assert validate_answer_contract("配置位于 compose.yaml [E1] [E2]", claims, {"E1", "E2"}) == (
        True, "validated",
    )
    assert validate_answer_contract("配置位于 compose.yaml [E1]", claims, {"E1", "E2"}) == (
        False, "missing_required_citation",
    )
    assert validate_answer_contract("配置位于 compose.yaml [E1] [E2] [E3]", claims, {"E1", "E2"}) == (
        False, "unapproved_citation",
    )
    assert validate_answer_contract(
        "配置位于 compose.yaml，而且一定会自动加载 [E1] [E2]", claims, {"E1", "E2"},
    ) == (False, "claim_text_or_mapping_changed")


def test_claim_contract_fallback_contains_only_approved_claims_and_citations():
    claims = (
        ClaimSupport("第一条事实", ("E1",)),
        ClaimSupport("第二条事实", ("E2", "E3")),
    )
    assert render_claim_contract_answer(claims) == "- 第一条事实 [E1]\n- 第二条事实 [E2] [E3]"


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


def test_fixed_modes_use_top_level_agent_with_capability_constraints(monkeypatch):
    project_id = "11111111-1111-4111-8111-111111111111"
    scope = main_module.AccessScope(
        "user-1", "workspace-1", "测试空间", None, frozenset({project_id}), "owner",
    )
    monkeypatch.setattr(main_module, "request_scope", lambda _workspace_id=None: scope)
    monkeypatch.setattr(main_module, "get_code_overview", lambda *_args: {"project_id": project_id})

    async def fake_agentic(message, project_id, scope=None, capability_mode="auto"):
        return {"message": message, "project_id": project_id, "mode": capability_mode, "scope": scope}

    monkeypatch.setattr(main_module, "agentic_knowledge_stream", fake_agentic)
    rag = asyncio.run(knowledge_stream(KnowledgeStreamRequest(message="文档问题", mode="rag")))
    code = asyncio.run(knowledge_stream(KnowledgeStreamRequest(
        message="代码问题", mode="codewiki", project_id=project_id,
    )))
    hybrid = asyncio.run(knowledge_stream(KnowledgeStreamRequest(
        message="联合问题", mode="hybrid", project_id=project_id,
    )))

    assert rag["mode"] == "rag" and rag["project_id"] is None
    assert code["mode"] == "codewiki" and code["project_id"] == project_id
    assert hybrid["mode"] == "hybrid" and hybrid["project_id"] == project_id


def test_legacy_code_wiki_endpoint_delegates_to_unified_answer_gate(monkeypatch):
    project_id = "11111111-1111-4111-8111-111111111111"
    scope = main_module.AccessScope(
        "user-1", "workspace-1", "测试空间", None, frozenset({project_id}), "owner",
    )
    monkeypatch.setattr(main_module, "request_scope", lambda _workspace_id=None: scope)
    monkeypatch.setattr(main_module, "get_code_overview", lambda *_args: {"project_id": project_id})

    async def fake_agentic(message, selected_project_id, scope=None, capability_mode="auto"):
        return {"message": message, "project_id": selected_project_id, "mode": capability_mode, "scope": scope}

    monkeypatch.setattr(main_module, "agentic_knowledge_stream", fake_agentic)
    result = asyncio.run(code_wiki_agent_stream(main_module.CodeAgentRequest(
        project_id=project_id, message="代码问题",
    )))
    assert result["mode"] == "codewiki"
    assert result["project_id"] == project_id


def test_legacy_rag_endpoints_delegate_to_unified_answer_gate(monkeypatch):
    scope = main_module.AccessScope(
        "user-1", "workspace-1", "测试空间", None, frozenset(), "owner",
    )
    calls = []
    monkeypatch.setattr(main_module, "request_scope", lambda _workspace_id=None: scope)

    async def fake_agentic(message, project_id, scope=None, capability_mode="auto"):
        calls.append((message, project_id, capability_mode, scope))

        async def events():
            yield 'event: done\ndata: {"answer":"已批准回答","sources":[]}\n\n'

        return StreamingResponse(events(), media_type="text/event-stream")

    monkeypatch.setattr(main_module, "agentic_knowledge_stream", fake_agentic)
    sync_result = asyncio.run(main_module.chat(main_module.ChatRequest(message="文档问题")))
    stream_result = asyncio.run(main_module.chat_stream(main_module.ChatRequest(message="文档问题")))

    assert sync_result["answer"] == "已批准回答"
    assert isinstance(stream_result, StreamingResponse)
    assert calls == [
        ("文档问题", None, "rag", scope),
        ("文档问题", None, "rag", scope),
    ]


def test_public_answer_routes_have_no_legacy_executor_endpoints():
    endpoints = {
        route.path: route.endpoint.__name__
        for route in main_module.app.routes
        if getattr(route, "path", None) in {
            "/api/chat", "/api/chat/stream", "/api/code-wiki/agent/stream", "/api/knowledge/stream",
        }
    }
    assert endpoints == {
        "/api/chat": "chat",
        "/api/chat/stream": "chat_stream",
        "/api/code-wiki/agent/stream": "code_wiki_agent_stream",
        "/api/knowledge/stream": "knowledge_stream",
    }


def test_answer_authorization_checkpoint_rejects_project_revocation(monkeypatch):
    original = main_module.AccessScope(
        "user-1", "workspace-1", "测试空间", None, frozenset({"repo-1"}), "owner",
    )
    revoked = main_module.AccessScope(
        "user-1", "workspace-1", "测试空间", None, frozenset(), "owner",
    )
    monkeypatch.setattr(main_module, "refresh_request_scope", lambda _scope: revoked)

    with pytest.raises(PermissionError, match="回答期间撤销"):
        asyncio.run(refresh_answer_authorization(
            original, 0.0, required_project_ids={"repo-1"}, force=True,
        ))


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
        {"type": "delta", "content": "事实成立 [rag-2-E1]"}, {"type": "usage", "completion_tokens": 4},
    ]))

    async def collect():
        scope = main_module.AccessScope("user-1", "workspace-1", "测试空间", None, frozenset(), "owner")
        response = await agentic_knowledge_stream("问题", None, scope)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    content = asyncio.run(collect())
    assert '"id": "knowledge_round_2"' in content
    assert content.count("event: done") == 1
    assert "事实成立 [rag-2-E1]" in content


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


def test_agentic_investigation_event_is_not_forwarded_after_revocation(monkeypatch):
    """长时间代码取证后必须先复核项目 ACL，再公开路径或符号摘要。"""
    project_id = "11111111-1111-4111-8111-111111111111"
    monkeypatch.setattr(
        main_module,
        "decide_next_knowledge_action",
        lambda *_args, **_kwargs: KnowledgeDecision(
            next_action="query_code_wiki", route="wiki", queries=("定位配置",),
            target_project_id=project_id, public_update="开始代码调查",
        ),
    )
    monkeypatch.setattr(main_module, "refresh_request_scope", lambda scope: scope)
    monkeypatch.setattr(
        main_module,
        "_public_agent_trace",
        lambda *_args, **_kwargs: iter(['event: model\ndata: {"delta":"规划"}\n\n']),
    )

    async def fake_code_evidence(_request):
        async def events():
            yield 'event: model\ndata: {"delta":"secret/path.py"}\n\n'
        return StreamingResponse(events(), media_type="text/event-stream")

    checkpoint_calls = 0

    async def revoke_after_planning(scope, _last_checked, **_kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if checkpoint_calls >= 3:
            raise PermissionError("project access revoked")
        return scope, 1.0

    monkeypatch.setattr(main_module, "_code_wiki_evidence_stream", fake_code_evidence)
    monkeypatch.setattr(main_module, "refresh_answer_authorization", revoke_after_planning)

    async def collect():
        scope = main_module.AccessScope(
            "user-1", "workspace-1", "测试空间", None, frozenset({project_id}), "owner",
        )
        # 不传锚点项目，覆盖 Planner 从授权候选集合动态选择项目的分支。
        response = await agentic_knowledge_stream("配置在哪里", None, scope)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    content = asyncio.run(collect())
    assert "规划" in content
    assert "secret/path.py" not in content
    assert '"error_code": "AUTHORIZATION_REVOKED"' in content
    assert "event: done" not in content


def test_revoked_dynamic_project_evidence_never_reenters_planner(monkeypatch):
    """跨轮撤权必须发生在旧代码证据再次发送给 Planner 模型之前。"""
    project_id = "11111111-1111-4111-8111-111111111111"
    planner_calls = 0

    def decide(*_args, **_kwargs):
        nonlocal planner_calls
        planner_calls += 1
        return KnowledgeDecision(
            next_action="query_code_wiki", route="wiki", queries=("定位配置",),
            target_project_id=project_id, public_update="开始代码调查",
        )

    monkeypatch.setattr(main_module, "decide_next_knowledge_action", decide)
    monkeypatch.setattr(main_module, "refresh_request_scope", lambda scope: scope)
    monkeypatch.setattr(main_module, "_public_agent_trace", lambda *_args, **_kwargs: iter(()))

    async def fake_code_evidence(_request):
        payload = {
            "project": {"commit": "commit-a"},
            "sources": [{"id": "C1", "path": "config.yaml", "line": 1}],
            "evidence_payloads": [{
                "result": {"name": "service", "citation": "[C1]"},
            }],
            "observability": {
                "tool_calls": 1, "completed_rounds": 1, "evidence_sufficient": True,
            },
        }

        async def events():
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    checkpoint_calls = 0

    async def revoke_before_second_planner(scope, _last_checked, **_kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        # 第一次在首轮 Planner 前，第二次在已有代码证据进入下一轮 Planner 前。
        if checkpoint_calls >= 2:
            raise PermissionError("project access revoked")
        return scope, 1.0

    monkeypatch.setattr(main_module, "_code_wiki_evidence_stream", fake_code_evidence)
    monkeypatch.setattr(main_module, "refresh_answer_authorization", revoke_before_second_planner)

    async def collect():
        scope = main_module.AccessScope(
            "user-1", "workspace-1", "测试空间", None, frozenset({project_id}), "owner",
        )
        response = await agentic_knowledge_stream("配置在哪里", None, scope)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    content = asyncio.run(collect())
    assert planner_calls == 1
    assert '"error_code": "AUTHORIZATION_REVOKED"' in content
    assert "event: done" not in content


def test_refusal_done_rechecks_authorization_and_omits_raw_evidence(monkeypatch):
    """拒答 done 也必须鉴权；公开状态永远不携带证据正文或工具载荷。"""
    decisions = iter([
        KnowledgeDecision(next_action="retrieve_rag", route="rag", queries=("查询一",)),
        KnowledgeDecision(next_action="retrieve_rag", route="rag", queries=("查询二",)),
        KnowledgeDecision(next_action="refuse", route="rag", reason="合法路径已耗尽"),
    ])
    monkeypatch.setattr(main_module, "decide_next_knowledge_action", lambda *_args, **_kwargs: next(decisions))
    monkeypatch.setattr(main_module, "refresh_request_scope", lambda scope: scope)
    monkeypatch.setattr(
        main_module,
        "_public_agent_trace",
        lambda *_args, **_kwargs: iter(['event: model\ndata: {"delta":"状态"}\n\n']),
    )

    async def empty_rag(_query, tool_call_id, _scope=None):
        return KnowledgeToolResult(
            tool_call_id=tool_call_id, capability="retrieve_rag", status="empty",
        ), []

    checkpoint_calls = 0

    async def revoke_before_done(scope, _last_checked, **_kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        # 两轮各有一次规划事件和一次 RAG 结果，第三轮有一次规划事件；
        # 第六次调用就是拒答 done 的最终出口检查。
        if checkpoint_calls >= 6:
            raise PermissionError("membership revoked")
        return scope, 1.0

    monkeypatch.setattr(main_module, "_retrieve_rag_capability", empty_rag)
    monkeypatch.setattr(main_module, "refresh_answer_authorization", revoke_before_done)

    async def collect():
        scope = main_module.AccessScope("user-1", "workspace-1", "测试空间", None, frozenset(), "owner")
        response = await agentic_knowledge_stream("没有答案的问题", None, scope)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    content = asyncio.run(collect())
    assert '"error_code": "AUTHORIZATION_REVOKED"' in content
    assert "event: done" not in content

    state = KnowledgeAgentState("问题")
    state.record_tool_result(KnowledgeToolResult(
        tool_call_id="rag-1", capability="retrieve_rag", status="success",
        result={"raw_secret": "不能公开"},
        evidence=(KnowledgeEvidence(
            evidence_id="E1", source_type="rag", content_or_fact="敏感证据正文",
        ),),
    ))
    public_state = state.public_snapshot()
    assert "evidence" not in public_state
    assert "tool_results" not in public_state
    assert "敏感证据正文" not in str(public_state)
    assert "不能公开" not in str(public_state)


def test_agentic_final_answer_does_not_emit_buffered_text_after_revocation(monkeypatch):
    decisions = iter([
        KnowledgeDecision(next_action="retrieve_rag", route="rag", queries=("查询",)),
        KnowledgeDecision(
            next_action="answer", route="rag",
            claims=(ClaimSupport("事实成立", ("rag-1-E1",)),),
            supporting_evidence_ids=("rag-1-E1",),
        ),
    ])
    monkeypatch.setattr(main_module, "decide_next_knowledge_action", lambda *_args, **_kwargs: next(decisions))

    async def fake_rag(_query, tool_call_id, _scope=None):
        evidence = KnowledgeEvidence(
            evidence_id=f"{tool_call_id}-E1", source_type="rag",
            content_or_fact="可验证事实", locator="规范.md#事实",
        )
        return KnowledgeToolResult(
            tool_call_id=tool_call_id, capability="retrieve_rag", status="success", evidence=(evidence,),
        ), [{"id": evidence.evidence_id, "type": "rag", "ref": evidence.locator}]

    checkpoint_calls = 0

    async def revoke_during_answer(scope, _last_checked, **_kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if checkpoint_calls >= 2:
            raise PermissionError("membership revoked")
        return scope, 1.0

    monkeypatch.setattr(main_module, "_retrieve_rag_capability", fake_rag)
    monkeypatch.setattr(main_module, "refresh_answer_authorization", revoke_during_answer)
    monkeypatch.setattr(main_module, "stream_answer", lambda *_args: iter([
        {"type": "delta", "content": "不应泄漏的"},
        {"type": "delta", "content": "回答 [rag-1-E1]"},
    ]))

    async def collect():
        scope = main_module.AccessScope("user-1", "workspace-1", "测试空间", None, frozenset(), "owner")
        response = await agentic_knowledge_stream("问题", None, scope)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    content = asyncio.run(collect())
    assert '"error_code": "AUTHORIZATION_REVOKED"' in content
    assert "不应泄漏的" not in content
    assert "event: done" not in content


def test_agentic_final_answer_rechecks_authorization_while_sending(monkeypatch):
    """模型缓存已通过校验后，撤权也必须阻止首个答案分片和最终 done。"""
    decisions = iter([
        KnowledgeDecision(next_action="retrieve_rag", route="rag", queries=("查询",)),
        KnowledgeDecision(
            next_action="answer", route="rag",
            claims=(ClaimSupport("事实成立", ("rag-1-E1",)),),
            supporting_evidence_ids=("rag-1-E1",),
        ),
    ])
    monkeypatch.setattr(main_module, "decide_next_knowledge_action", lambda *_args, **_kwargs: next(decisions))

    async def fake_rag(_query, tool_call_id, _scope=None):
        evidence = KnowledgeEvidence(
            evidence_id=f"{tool_call_id}-E1", source_type="rag",
            content_or_fact="可验证事实", locator="规范.md#事实",
        )
        return KnowledgeToolResult(
            tool_call_id=tool_call_id, capability="retrieve_rag", status="success", evidence=(evidence,),
        ), [{"id": evidence.evidence_id, "type": "rag", "ref": evidence.locator}]

    checkpoint_calls = 0

    async def revoke_before_sending(scope, _last_checked, **_kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        # 1=读取模型 delta，2=模型结束后的强制检查，3=首个缓存分片发送前。
        if checkpoint_calls >= 3:
            raise PermissionError("membership revoked")
        return scope, 1.0

    monkeypatch.setattr(main_module, "_retrieve_rag_capability", fake_rag)
    monkeypatch.setattr(main_module, "refresh_answer_authorization", revoke_before_sending)
    monkeypatch.setattr(main_module, "stream_answer", lambda *_args: iter([
        {"type": "delta", "content": "事实成立 [rag-1-E1]"},
    ]))

    async def collect():
        scope = main_module.AccessScope("user-1", "workspace-1", "测试空间", None, frozenset(), "owner")
        response = await agentic_knowledge_stream("问题", None, scope)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    content = asyncio.run(collect())
    assert '"error_code": "AUTHORIZATION_REVOKED"' in content
    assert "事实成立 [rag-1-E1]" not in content
    assert "event: done" not in content


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
