from kvprof.request import Request
from kvprof.sections import assign_tokens, build_sections, iter_nodes, section_at
from kvprof.util import char_to_byte_table

R = ("<|im_start|>system\nSYS TEXT\n\n# Tools\n\nYou have:\n\n<tools>\n<function>\n<name>Read</name>\n<description>d</description>\n</function>\n"
     "<function>\n<name>Edit</name>\n<description>e</description>\n</function>\n</tools>\n\nInstructions<|im_end|>\n"
     "<|im_start|>user\nhello\nreminder<|im_end|>\n<|im_start|>assistant\nok<tool_call>\n<function=Read>\n</function>\n</tool_call><|im_end|>\n"
     "<|im_start|>user\n<tool_response>\nRESULT\n</tool_response><|im_end|>\n<|im_start|>assistant\n")
BODY = {"system": "SYS TEXT", "tools": [{"name": "Read", "description": "d", "input_schema": {}}, {"name": "Edit", "description": "e", "input_schema": {}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "reminder"}]},
                     {"role": "assistant", "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "c", "name": "Read", "input": {}}]},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": "RESULT"}]}]}


def _build(rendered, req):
    msgs, _, origins, _ = req.to_openai()
    return build_sections("chatml_qwen", rendered, req, msgs, origins, req.render_units())


def test_selftest_port():
    req = Request(BODY, "anthropic")
    tree, ok, reason = _build(R, req)
    assert ok, reason
    ids = [n["id"] for n in iter_nodes(tree)]
    assert "tool:Read" in ids and "msg[0].block[1]" in ids and "msg[2].block[0]" in ids and "generation_prompt" in ids
    c2b = char_to_byte_table(R)
    n = len(R.encode())
    bstart = list(range(n + 1))  # one token per byte
    leaves = assign_tokens(tree, c2b, bstart, n)
    assert sum(x["tokens"] for x in tree) == n
    assert section_at(leaves, R.find("RESULT")) == "msg[2].block[0]"
    assert section_at(leaves, R.find("<name>Edit")) == "tool:Edit"
    assert section_at(leaves, R.find("SYS")) == "system.text"
    assert section_at(leaves, R.find("reminder")) == "msg[0].block[1]"


def test_openai_shape_same_sections():
    body = {"messages": [{"role": "system", "content": "SYS TEXT"},
                         {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "reminder"}]},
                         {"role": "assistant", "content": "ok", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]},
                         {"role": "tool", "tool_call_id": "c", "content": "RESULT"}],
            "tools": [{"type": "function", "function": {"name": "Read", "description": "d", "parameters": {}}},
                      {"type": "function", "function": {"name": "Edit", "description": "e", "parameters": {}}}]}
    req = Request(body, "openai")
    tree, ok, reason = _build(R, req)
    assert ok, reason
    ids = [n["id"] for n in iter_nodes(tree)]
    assert "system.text" in ids and "tool:Edit" in ids and "msg[1].part[1]" in ids and "msg[2].tool_call[0]" in ids and "msg[3].tool_result" in ids
    leaves = assign_tokens(tree, char_to_byte_table(R), list(range(len(R) + 1)), len(R))
    assert section_at(leaves, R.find("RESULT")) == "msg[3].tool_result"
    assert section_at(leaves, R.find("<tool_call>")) == "msg[2].tool_call[0]"


def test_no_markers_single_prompt_node():
    req = Request({"messages": [{"role": "user", "content": "x"}]}, "anthropic")
    tree, ok, reason = _build("plain text prompt", req)
    assert not ok and tree[0]["id"] == "prompt"


def test_turn_level_fallback():
    req = Request({"system": "S", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}, "anthropic")
    rendered = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>weird\nyo<|im_end|>\n"
    tree, ok, reason = _build(rendered, req)
    assert not ok and "assistant turn" in reason
    msgs = next(n for n in tree if n["id"] == "messages")
    assert [c["id"] for c in msgs["children"]] == ["turn[1]", "turn[2]"]


def test_tools_without_system_message():
    body = {"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "Read", "description": "d", "parameters": {}}}]}
    req = Request(body, "openai")
    rendered = ("<|im_start|>system\n\n\n# Tools\n\n<tools>\n<function>\n<name>Read</name>\n</function>\n</tools>\n\nInstr<|im_end|>\n"
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
    tree, ok, _ = _build(rendered, req)
    assert ok and [n["id"] for n in tree] == ["tools", "messages", "generation_prompt"]
    assert tree[0]["char"][0] == 0
