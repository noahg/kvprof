from kvprof.request import Request, client_info, detect_api

ANTH = {"model": "m", "stream": True, "max_tokens": 100, "system": [{"type": "text", "text": "S1"}, {"type": "text", "text": "S2"}],
        "tools": [{"name": "Read", "description": "read", "input_schema": {"type": "object"}}],
        "metadata": {"user_id": '{"device_id":"d","session_id":"sess-1"}'},
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "t"}, {"type": "text", "text": "ok"},
                                              {"type": "tool_use", "id": "c1", "name": "Read", "input": {"p": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "is_error": True, "content": [{"type": "text", "text": "E"}]},
                                         {"type": "image", "source": {}}, {"type": "text", "text": "look"}]}]}
OAI = {"model": "m", "stream": False, "max_completion_tokens": 50, "user": "u-42",
       "tools": [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {"type": "object"}}}],
       "messages": [{"role": "system", "content": "SYS"},
                    {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}]},
                    {"role": "assistant", "content": None, "reasoning_content": "why",
                     "tool_calls": [{"id": "1", "type": "function", "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}}]},
                    {"role": "tool", "tool_call_id": "1", "content": "file.txt"}]}


def test_detect_api():
    assert detect_api("/v1/messages") == "anthropic" and detect_api("/v1/chat/completions") == "openai"
    assert detect_api("/v1/messages/count_tokens") is None and detect_api("/v1/messages?beta=true") == "anthropic"


def test_anthropic_units():
    r = Request(ANTH, "anthropic", {"is_error_prefix": "Error: "})
    assert (r.model, r.stream, r.max_tokens, r.n_messages, r.n_tools, r.has_tools) == ("m", True, 100, 3, 1, True)
    assert r.system_text == "S1 S2" and r.metadata_session == "sess-1"
    assert r.last_message_kinds == {"tool_result", "image", "text"}
    units = r.render_units()
    ids = [[u.id for u in mu] for mu in units]
    assert ids == [["system.text"], ["msg[0].text"], ["msg[1].block[0]", "msg[1].block[1]", "msg[1].block[2]"],
                   ["msg[2].block[0]"], ["msg[2].block[1]", "msg[2].block[2]"]]
    assert units[1][0].whole and units[2][0].kind == "reasoning" and units[2][2].kind == "tool_use"
    assert units[2][2].text == '{"p":1}' and units[2][2].label == "tool_use Read"
    tr = units[3][0]
    assert tr.kind == "tool_result" and tr.is_error and tr.text == "Error: E" and tr.raw_text == "E"
    assert units[4][0].text == "[image]" and r.unit_text("msg[2].block[1]") is None
    assert r.unit_text("system.text") == "S1\n\nS2"
    assert r.unit_text("tool:Read") == 'Read\nread\n{"type": "object"}'
    assert r.unit_text("tools.preamble") is None


def test_openai_units():
    r = Request(OAI, "openai")
    assert (r.max_tokens, r.n_tools, r.system_text, r.metadata_session) == (50, 1, "SYS", "u-42")
    assert r.last_message_kinds == {"tool_result"}
    msgs, tools, origins, _ = r.to_openai()
    assert msgs is not None and origins[0] == {"msg": None, "kind": "system"} and origins[1] == {"msg": 1, "blocks": None}
    ids = [[u.id for u in mu] for mu in r.render_units()]
    assert ids == [["system.text"], ["msg[1].part[0]", "msg[1].part[1]"], ["msg[2].reasoning", "msg[2].tool_call[0]"], ["msg[3].tool_result"]]
    assert r.unit_text("msg[2].tool_call[0]") == '{"cmd":"ls"}' and r.unit_text("tool:bash").startswith("bash\nrun\n")
    assert r.unit("msg[1].part[1]").text == "[image_url]"


def test_hint_text_without_system():
    r = Request({"messages": [{"role": "user", "content": "Generate a title"}]}, "openai")
    assert r.system_text == "" and r.hint_text() == "Generate a title"


def test_client_info():
    ci = client_info({"User-Agent": "claude-cli/2.1.283 (external, cli)", "X-Claude-Code-Session-Id": "abc"}, {})
    assert ci == {"client": "Claude Code", "version": "2.1.283", "user_agent": "claude-cli/2.1.283 (external, cli)", "session": "abc", "session_source": "header"}
    ci = client_info({"user-agent": "claude-cli/2.1.283"}, ANTH)
    assert ci["session"] == "sess-1" and ci["session_source"] == "metadata"
    ci = client_info({"user-agent": "opencode/0.15.3 ai-sdk/provider-utils/3.0.9 runtime/bun/1.2.23"}, OAI)
    assert ci["client"] == "OpenCode" and ci["version"] == "0.15.3" and ci["session"] == "u-42" and ci["session_source"] == "user"
    ci = client_info({"user-agent": "opencode/latest/2.0.17/cli", "x-opencode-session": "ses_f1", "x-session-id": "ses_f1"}, {})
    assert (ci["client"], ci["version"], ci["session"], ci["session_source"]) == ("OpenCode", "2.0.17", "ses_f1", "header")
    ci = client_info({"user-agent": "ai-sdk/openai-compatible/1.0.19 ai-sdk/provider-utils/3.0.9 runtime/bun/1.2.23"}, {})
    assert ci["client"] == "AI SDK app"
    assert client_info({"user-agent": "OpenAI/Python 1.99.0"}, {})["client"] == "OpenAI Python SDK"
    assert client_info({}, {})["client"] == "unknown client"
    assert client_info({"user-agent": "mytool/1"}, {"metadata": {"user_id": "plain-id"}})["session"] == "plain-id"
