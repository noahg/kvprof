"""Stored traces (as the tap writes them) through the analyzer, against the fake server, without a log."""
import json

from conftest import load_fixture_trace
from kvprof.analyze import Pipeline
from kvprof.config import Settings
from kvprof.summarize import summary_line


def _dir(tmp_path, names):
    d = tmp_path / "traces"
    d.mkdir()
    for n in names:
        tr = load_fixture_trace(n)
        (d / (tr["id"] + ".json")).write_text(json.dumps(tr))
    return str(d)


def test_session_kinds(fake_upstream, tmp_path):
    d = _dir(tmp_path, ["anthropic_cold", "anthropic_tool_roundtrip", "anthropic_rewrite", "openai_cold"])
    p = Pipeline(Settings(upstream=fake_upstream.url, trace_dir=d, use_log=False))
    p.refresh_all()
    kinds = [a["kind"] for a in p.analyses]
    assert kinds == ["cold", "tool_roundtrip", "rewrite", "cold"]
    rw = p.analyses[2]
    assert rw["subkind"] == "msg[0].block[1]" and rw["parent_id"] == p.analyses[1]["id"]
    ct = rw["diff"]["changed_text"]
    assert ct["removed"] == "29" and ct["added"] == "30"
    assert 'text changed: "29" -> "30"' in summary_line(rw)
    assert all(a["exact"]["usage"] is True for a in p.analyses)
    assert p.analyses[0]["session_id"] == "fixture-session" and not p.analyses[0]["session_inferred"]
    assert p.analyses[3]["client_name"] == "OpenCode" and p.analyses[3]["api"] == "openai"
    assert all(a["concurrent_with"] == [] for a in p.analyses)


def test_schema1_trace_loads(fake_upstream, tmp_path):
    d = _dir(tmp_path, ["anthropic_cold_schema1"])
    p = Pipeline(Settings(upstream=fake_upstream.url, trace_dir=d, use_log=False))
    p.refresh_all()
    (a,) = p.analyses
    assert a["api"] == "anthropic" and a["kind"] == "cold" and a["exact"]["usage"] is True
    assert a["usage"]["total"] == a["n_tokens"] and a["usage_raw"]["api"] == "anthropic"


def test_template_profile_override(fake_upstream, tmp_path):
    d = _dir(tmp_path, ["anthropic_cold", "anthropic_tool_roundtrip"])
    p = Pipeline(Settings(upstream=fake_upstream.url, trace_dir=d, use_log=False, template_profile="generic"))
    p.refresh_all()
    assert p.profile == "generic" and all(a["profile"] == "generic" for a in p.analyses)
    assert [a["kind"] for a in p.analyses] == ["cold", "tool_roundtrip"]
