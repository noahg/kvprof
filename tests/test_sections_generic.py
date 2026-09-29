from fake_upstream import render_chatml, render_llama3, tokenize
from kvprof.request import Request
from kvprof.sections import assign_tokens, build_sections, iter_nodes, section_at
from kvprof.sections.profiles import detect_profile
from kvprof.util import byte_starts, char_to_byte_table, piece_bytes

BODY = {"system": "You are a coding agent. Follow the rules.",
        "tools": [{"name": "Read", "description": "Read a file from the local disk.", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}},
                  {"name": "Bash", "description": "Run a shell command and return its output.", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Please summarize the notes file for me."},
                                         {"type": "text", "text": "  <system-reminder>today is Tuesday</system-reminder>\n"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "I will read the notes file first."},
                                              {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/work/notes/todo-list.txt"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "buy milk\nfix the flaky build on main\n"}]},
            {"role": "assistant", "content": "The notes say: buy milk, fix the build."},
            {"role": "user", "content": "Thanks, now run the tests please."}]}


def analyze(render, body=BODY, api="anthropic"):
    req = Request(body, api)
    msgs, tools, origins, _ = req.to_openai()
    R = render(msgs, tools)
    toks = [(i, piece_bytes(p)) for i, p in tokenize(R)]
    bs = byte_starts(toks)
    tree, ok, reason = build_sections("generic", R, req, msgs, origins, req.render_units())
    leaves = assign_tokens(tree, char_to_byte_table(R), bs, len(toks))
    return R, tree, ok, reason, leaves, bs, len(toks)


def tok_of(R, bs, s, nth=0):
    i = -1
    for _ in range(nth + 1):
        i = R.find(s, i + 1)
    b = len(R[:i].encode())
    import bisect
    return bisect.bisect_right(bs, b) - 1


def test_profile_detection():
    assert detect_profile("... <|im_start|> ... <tools> ... <tool_call> ...") == "chatml_qwen"
    assert detect_profile("<|start_header_id|>") == "generic" and detect_profile(None) == "generic"
    assert detect_profile("<|im_start|> only chatml, no tools") == "generic"


def test_llama3_all_located():
    R, tree, ok, reason, leaves, bs, n = analyze(render_llama3)
    assert ok, reason
    assert sum(x["tokens"] for x in tree) == n
    assert [x["id"] for x in tree] == ["system", "tools", "messages", "generation_prompt"]
    ids = [x["id"] for x in iter_nodes(tree)]
    for want in ("system.text", "tools.preamble", "tool:Read", "tool:Bash", "msg[0].block[0]", "msg[0].block[1]", "msg[1].block[0]",
                 "msg[1].block[1]", "msg[2].block[0]", "msg[3].text", "msg[4].text"):
        assert want in ids, want
    assert section_at(leaves, tok_of(R, bs, "coding agent")) == "system.text"
    assert section_at(leaves, tok_of(R, bs, "Run a shell")) == "tool:Bash"
    assert section_at(leaves, tok_of(R, bs, "summarize the notes")) == "msg[0].block[0]"
    assert section_at(leaves, tok_of(R, bs, "today is Tuesday")) == "msg[0].block[1]"   # found via strip(): the template trims
    assert section_at(leaves, tok_of(R, bs, "todo-list")) == "msg[1].block[1]"         # args re-serialized: name + value
    assert section_at(leaves, tok_of(R, bs, "flaky build")) == "msg[2].block[0]"
    assert section_at(leaves, tok_of(R, bs, "run the tests")) == "msg[4].text"
    gp = tree[-1]
    assert R[gp["char"][0]:].endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_chatml_via_generic_matches_leaves():
    R, tree, ok, _, leaves, bs, n = analyze(render_chatml)
    assert ok and sum(x["tokens"] for x in tree) == n
    assert section_at(leaves, tok_of(R, bs, "flaky build")) == "msg[2].block[0]"
    assert section_at(leaves, tok_of(R, bs, "<name>Bash")) == "tool:Bash"


def test_openai_shape():
    body = {"messages": [{"role": "system", "content": "You are OpenCode, an agent."}, {"role": "user", "content": "list the files in this directory"},
                         {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"ls -la /work\"}"}}]},
                         {"role": "tool", "tool_call_id": "1", "content": "a.txt\nb.txt"}],
            "tools": [{"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object"}}}]}
    R, tree, ok, reason, leaves, bs, n = analyze(render_llama3, body, "openai")
    assert ok, reason
    assert section_at(leaves, tok_of(R, bs, "ls -la")) == "msg[2].tool_call[0]"
    assert section_at(leaves, tok_of(R, bs, "b.txt")) == "msg[3].tool_result"


def mangle(msgs, tools):
    """A template that rewrites message text (upper-cases it), so only turn markers survive."""
    out = render_llama3([dict(m, content=m["content"].upper()) if m["role"] not in ("system",) and isinstance(m.get("content"), str) else m for m in msgs], tools)
    return out


def test_turn_level_fallback():
    body = {"system": "sys prompt here", "messages": [
        {"role": "user", "content": "hello there, this is the first message"},
        {"role": "assistant", "content": "a reply that will be mangled"},
        {"role": "user", "content": "another question that gets mangled"},
        {"role": "assistant", "content": "second reply also mangled"}]}

    def partly(msgs, tools):
        # keep the first message intact, upper-case the others
        m2 = [msgs[0], msgs[1]] + [dict(m, content=m["content"].upper()) for m in msgs[2:]]
        return render_llama3(m2, tools)
    R, tree, ok, reason, leaves, bs, n = analyze(partly, body)
    assert not ok and "turn-level" in reason
    msgs = next(x for x in tree if x["id"] == "messages")
    assert [c["id"] for c in msgs["children"]] == ["turn[0]"]
    assert sum(x["tokens"] for x in tree) == n


def test_first_message_missing_single_prompt():
    R, tree, ok, reason, leaves, bs, n = analyze(mangle, {"messages": [{"role": "user", "content": "lowercase text only"}]})
    assert not ok and [x["id"] for x in tree] == ["prompt"] and tree[0]["tokens"] == n
