"""fake llama-server -> tap -> traces -> Pipeline (render, tokenize, sections, diff, classify, log pairing) -> serve."""
import json
import os
import threading
import time
import urllib.request

import pytest

from conftest import post
from fake_upstream import SERVER_VARIANT
from kvprof.analyze import Pipeline
from kvprof.config import Settings
from kvprof.convert import anthropic_to_openai

TOOLS = [{"name": "Read", "description": "Read a file from disk.",
          "input_schema": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Absolute path"}}, "required": ["file_path"]}},
         {"name": "Bash", "description": "Run a shell command.",
          "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
SYSTEM = [{"type": "text", "text": "You are a coding agent."}, {"type": "text", "text": "Be concise."}]
TURN1 = [{"role": "user", "content": [{"type": "text", "text": "Read README.md and summarize it."}]}]
TURN2 = TURN1 + [
    {"role": "assistant", "content": [{"type": "text", "text": "Let me read it."},
                                      {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/repo/README.md"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                  "content": [{"type": "text", "text": "# Project\nA tool that does things.\n"}]}]}]
OAI_TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a command",
                                               "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
OAI_MSGS = [{"role": "system", "content": "You are OpenCode, a coding agent."}, {"role": "user", "content": "list the files here"}]
UA_OPENCODE = "opencode/0.15.3 ai-sdk/provider-utils/3.0.9 runtime/bun/1.2.23"


def _wait_traces(d, n, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        ids = [x for x in os.listdir(d) if x.endswith(".json") and not x.endswith(".analysis.json") and not x.startswith(".")]
        if len(ids) >= n:
            done = True
            for x in ids:
                with open(os.path.join(d, x)) as f:
                    if json.load(f)["outcome"] == "inflight":
                        done = False
            if done:
                return
        time.sleep(0.05)
    raise AssertionError("traces not written")


@pytest.fixture
def session(tap_proc, fake_upstream):
    url = tap_proc["url"]
    anth = {"model": "test-model", "max_tokens": 256, "stream": True, "system": SYSTEM, "tools": TOOLS}
    post(url + "/v1/messages", dict(anth, messages=TURN1), raw=True)
    # the tool round-trip reuses everything the first request prefilled
    body1 = next(b for p, _, b in fake_upstream.requests if p == "/v1/messages")
    m, t, _, _ = anthropic_to_openai(body1, SERVER_VARIANT)
    first_total = len(fake_upstream.prompt_ids(m, t))
    post(url + "/v1/messages", dict(anth, messages=TURN2), headers={"x-fake-cached": str(first_total)}, raw=True)
    oai = {"model": "test-model", "tools": OAI_TOOLS, "messages": OAI_MSGS}
    post(url + "/v1/chat/completions", dict(oai, stream=True, stream_options={"include_usage": True}), headers={"user-agent": UA_OPENCODE}, raw=True)
    msgs2 = OAI_MSGS + [{"role": "assistant", "content": "Hello from the fake server."}, {"role": "user", "content": "thanks"}]
    post(url + "/v1/chat/completions", dict(oai, messages=msgs2), headers={"user-agent": UA_OPENCODE})
    _wait_traces(tap_proc["trace_dir"], 4)
    s = Settings(upstream=fake_upstream.url, trace_dir=tap_proc["trace_dir"], server_log=fake_upstream.log_path, log_anchor=fake_upstream.t0)
    p = Pipeline(s)
    p.refresh_all()
    return p


def test_pipeline(session):
    p = session
    a = p.analyses
    assert [x["api"] for x in a] == ["anthropic", "anthropic", "openai", "openai"]
    assert p.profile == "chatml_qwen" and p.info.n_slots == 2 and p.info.n_ctx == 8192
    cold, rt, ocold, oturn = a
    assert cold["kind"] == "cold" and cold["exact"]["usage"] is True and cold["log_match"] == "exact"
    assert rt["kind"] == "tool_roundtrip" and rt["diff"]["end_extension"] and rt["log_match"] == "exact"
    assert abs(rt["log"]["residual_pp"]) <= 5 and rt["exact"]["usage"] is True
    assert rt["usage"]["cached"] == cold["n_tokens"] and rt["prefilled_usage"] == rt["n_tokens"] - cold["n_tokens"]
    assert all(x["mapping_ok"] for x in a), [x["mapping_reason"] for x in a]
    # every section tree sums to the prompt
    for x in a:
        assert sum(n["tokens"] for n in x["sections"]) == x["n_tokens"]
    ids = {n["id"] for n in _iter(rt["sections"])}
    assert {"system.text", "tool:Read", "tool:Bash", "msg[1].block[1]", "msg[2].block[0]", "generation_prompt"} <= ids
    assert ocold["exact"]["usage"] is True and oturn["exact"]["usage"] is True
    assert ocold["client_name"] == "OpenCode" and cold["client_name"] == "Claude Code"
    assert oturn["kind"] == "main_turn" and oturn["log_match"] == "exact" and oturn["usage"]["decode_rate"] == 50.0
    assert ocold["kind"] == "cold" and ocold["usage"]["total"] == ocold["n_tokens"]
    assert oturn["session_id"] == ocold["session_id"] and oturn["session_inferred"]
    # cached analyses are reused on the next pass
    assert p.refresh_all() is False
    p2 = Pipeline(p.settings)
    p2.refresh_all()
    assert [x["kind"] for x in p2.analyses] == [x["kind"] for x in a]


def test_summary_and_dump(session, capsys):
    from kvprof.summarize import dump_request, print_session, summarize
    print_session(session)
    out = capsys.readouterr().out
    assert "template chatml_qwen" in out and "tool_roundtrip" in out and "totals:" in out
    dump_request(session, "2")
    out = capsys.readouterr().out
    assert "⟦tool:Read" in out and "msg[2].block[0]" in out
    row = summarize(session.analyses[2])
    assert row["api"] == "openai" and row["exact"] is True and "from_claude_code" not in row


def test_no_log(session):
    s = Settings(upstream=session.settings.upstream, trace_dir=session.settings.trace_dir, use_log=False)
    p = Pipeline(s)
    p.refresh_all()
    assert all(x["log"] is None for x in p.analyses)
    from kvprof.summarize import summary_line
    assert "prefilled per usage" in summary_line(p.analyses[1])


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def test_serve(session):
    from http.server import ThreadingHTTPServer

    from kvprof.serve import Handler, make_handler
    handler = make_handler(session)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        st, h, body = _get(base + "/")
        assert st == 200 and b"<html" in body and "text/html" in h["Content-Type"]
        st, h, body = _get(base + "/api/requests")
        d = json.loads(body)
        assert st == 200 and d["n_slots"] == 2 and d["n_ctx"] == 8192 and d["package"] == "kvprof" and d["profile"] == "chatml_qwen"
        assert d["backend"] == "llamacpp" and d["api_counts"] == {"anthropic": 2, "openai": 2} and d["build"] == "b10909-test"
        assert len(d["requests"]) == 4 and d["trace_dir"]
        st, _, _ = _get(base + "/api/requests", {"If-None-Match": h["ETag"]})
        assert st == 304
        rid = d["requests"][1]["id"]
        st, _, body = _get(base + f"/api/requests/{rid}")
        det = json.loads(body)
        assert st == 200 and det["summary_row"]["kind"] == "tool_roundtrip" and "ids" not in det
        st, _, body = _get(base + f"/api/requests/{rid}/tokens?section=tools")
        tk = json.loads(body)
        assert st == 200 and tk["end"] > tk["start"] and "".join(t[1] for t in tk["tokens"]).count("<function>\n<name>") == 2
        st, _, body = _get(base + f"/api/requests/{rid}/diff?other={d['requests'][0]['id']}")
        assert st == 200 and json.loads(body)["end_extension"] is True
        st, _, body = _get(base + f"/api/requests/{rid}/body")
        assert st == 200 and json.loads(body)["messages"][2]["content"][0]["type"] == "tool_result"
        st, _, body = _get(base + f"/api/requests/{rid}/sse")
        assert st == 200 and json.loads(body)[0] == "event: message_start"
        st, _, _ = _get(base + "/api/requests/nope")
        assert st == 404
        assert Handler is not None
    finally:
        httpd.shutdown()
        httpd.server_close()


def _iter(nodes):
    for n in nodes:
        yield n
        yield from _iter(n["children"])
