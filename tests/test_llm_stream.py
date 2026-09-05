from types import SimpleNamespace

from app import llm
from app.code_agent import CODE_AGENT_TOOLS
from app.main import _public_agent_trace


def _chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_stream_code_agent_completion_merges_text_and_tool_deltas(monkeypatch):
    """模型文字和拆分的工具参数都必须按原顺序合并。"""
    chunks = [
        _chunk(content="当前已知：已定位配置。\n"),
        _chunk(content="下一步：读取源码。"),
        _chunk(tool_calls=[SimpleNamespace(
            index=0,
            id="call_1",
            function=SimpleNamespace(name="read_", arguments='{"path":"a.py",'),
        )]),
        _chunk(tool_calls=[SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(name="source", arguments='"start_line":1,"end_line":20}'),
        )]),
    ]
    request_options = {}

    def create(**kwargs):
        request_options.update(kwargs)
        return iter(chunks)

    completions = SimpleNamespace(create=create)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm, "_client", lambda *_args, **_kwargs: client)

    events = list(llm.stream_code_agent_completion(
        [{"role": "user", "content": "test"}],
        [{"type": "function", "function": {"name": "read_source"}}],
    ))

    assert [item["content"] for item in events if item["type"] == "delta"] == [
        "当前已知：已定位配置。\n",
        "下一步：读取源码。",
    ]
    message = events[-1]["message"]
    assert message.content == "当前已知：已定位配置。\n下一步：读取源码。"
    assert message.tool_calls[0].id == "call_1"
    assert message.tool_calls[0].function.name == "read_source"
    assert message.tool_calls[0].function.arguments == '{"path":"a.py","start_line":1,"end_line":20}'
    assert request_options["tool_choice"] == "required"
    assert request_options["extra_body"] == {"enable_thinking": False}


def test_code_agent_has_explicit_finish_control_tool():
    """Agent 结束调查必须使用语义事件，不能由固定轮次推断。"""
    names = [item["function"]["name"] for item in CODE_AGENT_TOOLS]
    assert "finish_investigation" in names


def test_public_agent_trace_streams_verifiable_investigation_deltas():
    """RAG 调查说明必须使用 investigation 阶段，并按增量推送。"""
    events = list(_public_agent_trace("trace-1", "RAG Agent", "abcdef", chunk_size=2))

    assert '"phase": "investigation"' in events[0]
    assert '"status": "started"' in events[0]
    assert '"delta": "ab"' in events[1]
    assert '"delta": "ef"' in events[3]
    assert '"status": "completed"' in events[-1]


def test_final_answer_receives_gate_approved_claim_contract(monkeypatch):
    request_options = {}

    def create(**kwargs):
        request_options.update(kwargs)
        return iter([_chunk(content="配置位于 compose.yaml [E1]")])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_client", lambda *_args, **_kwargs: client)

    events = list(llm.stream_answer(
        "配置在哪里？", "[E1] compose.yaml:1",
        [{"text": "配置位于 compose.yaml", "evidence_ids": ["E1"]}],
    ))

    assert events[0]["content"].endswith("[E1]")
    assert "机器可校验白名单" in request_options["messages"][0]["content"]
    assert "Claim 文本必须原样保留" in request_options["messages"][0]["content"]
    assert '"evidence_ids": ["E1"]' in request_options["messages"][1]["content"]


def test_investigation_note_stream_uses_selected_tools_and_recent_evidence(monkeypatch):
    """可见调查说明必须由模型基于真实工具选择和最近证据生成。"""
    chunks = [_chunk(content="当前已知：已找到配置。\n"), _chunk(content="信息缺口：还未核对源码。\n下一步：调用 read_source。")]
    request_options = {}

    def create(**kwargs):
        request_options.update(kwargs)
        return iter(chunks)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_client", lambda *_args, **_kwargs: client)
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="read_source", arguments='{"path":"config.py"}'),
    )

    text = "".join(llm.stream_code_investigation_note(
        "配置在哪里？",
        "demo",
        "abc123",
        [tool_call],
        [{"role": "tool", "content": '{"path":"config.yml"}'}],
    ))

    assert text.startswith("当前已知：")
    prompt = request_options["messages"][1]["content"]
    assert "config.yml" in prompt
    assert "read_source" in prompt
    assert "tools" not in request_options
    assert request_options["extra_body"] == {"enable_thinking": False}
