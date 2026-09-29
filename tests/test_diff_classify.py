from kvprof.diff import KINDS, classify, diff_analysis
from kvprof.request import Request

TOOLS = [{"id": "tools", "tokens": 100}]
NO_TOOLS = [{"id": "messages", "tokens": 100}]


def A(sections=TOOLS, n=1000, cached=None):
    return {"sections": sections, "n_tokens": n, "usage": {"cached": cached}}


def D(lcp, parent_len, cur_len, sec=None, changed=None):
    return {"lcp": lcp, "parent_len": parent_len, "cur_len": cur_len, "end_extension": lcp >= parent_len - 2,
            "first_divergent": {"tok": lcp, "cur_section": sec} if sec else None, "changed_text": changed}


def R(last, n_messages=2, tools=True):
    msgs = [{"role": "user", "content": "q"}] * (n_messages - 1) + [last]
    return Request({"system": "You are a title generator.", "messages": msgs,
                    **({"tools": [{"name": "Read", "input_schema": {}}]} if tools else {})}, "anthropic")


TEXT = {"role": "user", "content": "next question"}
RESULT = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}


def test_all_kinds():
    seen = set()
    cases = [
        ((A(), R(TEXT), None, None), "cold"),
        ((A(), R(TEXT), D(1000, 1000, 1000), {}), "retry"),
        ((A(), R(TEXT), D(900, 900, 1000), {}), "main_turn"),
        ((A(), R(RESULT), D(899, 900, 1000), {}), "tool_roundtrip"),
        ((A(), R(TEXT), D(100, 900, 1000, "msg[0].text"), {}), "cold"),
        ((A(), R(TEXT, 2), D(800, 900, 1000, "msg[1].text"), {"n_messages": 5}), "compaction"),
        ((A(), R(TEXT, 2), D(800, 900, 1000, "msg[1].text", {"added": "x"}), {"n_messages": 5}), "rewrite"),
        ((A(cached=900), R(TEXT), None, None), "warm"),
        ((A(NO_TOOLS), R(TEXT, tools=False), None, None), "side"),
    ]
    for (a, req, diff, parent), want in cases:
        kind, subkind, hint = classify(a, req, diff, parent)
        assert kind == want, (want, kind)
        seen.add(kind)
    assert seen == set(KINDS)


def test_side_hint_and_subkind():
    kind, _, hint = classify(A(NO_TOOLS), R(TEXT, tools=False), None, None)
    assert kind == "side" and hint.startswith("You are a title generator")
    kind, sub, _ = classify(A(), R(TEXT), D(800, 900, 1000, "msg[1].text", {"added": "x"}), {"n_messages": 1})
    assert kind == "rewrite" and sub == "msg[1].text"


def test_openai_tool_roundtrip():
    req = Request({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": []},
                                {"role": "tool", "tool_call_id": "1", "content": "r"}],
                   "tools": [{"type": "function", "function": {"name": "f"}}]}, "openai")
    assert classify(A(), req, D(899, 900, 1000), {})[0] == "tool_roundtrip"


def _an(aid, text, leaves_split):
    b = text.encode()
    return {"id": aid, "ids": list(b), "bstart": list(range(len(b) + 1)), "rendered": text,
            "leaves": [[0, leaves_split, "system.text"], [leaves_split, len(b), "msg[0].text"]]}


def test_diff_analysis_changed_text():
    pa = _an("p", "SYSTEM|hello world", 7)
    ca = _an("c", "SYSTEM|hello brave world", 7)
    preq = Request({"system": "SYSTEM", "messages": [{"role": "user", "content": "hello world"}]})
    creq = Request({"system": "SYSTEM", "messages": [{"role": "user", "content": "hello brave world"}]})
    d = diff_analysis(ca, pa, creq, preq)
    assert d["lcp"] == 13 and not d["end_extension"] and d["appended"] == len(ca["ids"]) - 13
    fd = d["first_divergent"]
    assert fd["cur_section"] == fd["parent_section"] == "msg[0].text" and fd["cur_piece"] == "b" and fd["parent_piece"] == "w"
    ct = d["changed_text"]
    assert ct["added"] == "brave " and ct["removed"] == "" and ct["at_char"] == 6


def test_diff_analysis_extension():
    pa = _an("p", "SYSTEM|hello", 7)
    ca = _an("c", "SYSTEM|hello again", 7)
    d = diff_analysis(ca, pa, None, None)
    assert d["end_extension"] and d["lcp"] == 12 and d["first_divergent"]["parent_section"] is None


def test_parent_prefers_extension_over_longer_prefix(fake_upstream, tmp_path):
    """A new session's tool round-trip must pair with its own previous prompt (which it extends), not with a
    different conversation that happens to share a few more tokens."""
    import json

    from conftest import load_fixture_trace
    from kvprof.analyze import Pipeline
    from kvprof.config import Settings
    d = tmp_path / "t"
    d.mkdir()
    cold, rt = load_fixture_trace("anthropic_cold"), load_fixture_trace("anthropic_tool_roundtrip")
    seq = [("0001", cold, "s1"), ("0002", rt, "s1"), ("0003", cold, "s2"), ("0004", rt, "s2")]
    for tag, tr, sess in seq:
        tr = json.loads(json.dumps(tr))
        tr["id"] = f"20260101-000000.000-{tag}"
        tr["seq"] = int(tag)
        tr["request"]["headers"]["x-claude-code-session-id"] = sess
        if tag == "0004":  # the second session read a different file: shares more with #2 than #3's whole prompt
            tr["request"]["body"]["messages"][2]["content"][0]["content"] = "buy milk\nfix the build\nand something else entirely"
        (d / (tr["id"] + ".json")).write_text(json.dumps(tr))
    p = Pipeline(Settings(upstream=fake_upstream.url, trace_dir=str(d), use_log=False))
    p.refresh_all()
    a = p.analyses
    assert [x["kind"] for x in a] == ["cold", "tool_roundtrip", "retry", "tool_roundtrip"]
    assert a[2]["parent_id"] == a[0]["id"] and a[3]["parent_id"] == a[2]["id"]
