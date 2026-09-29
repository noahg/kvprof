import json

import pytest

from conftest import load_fixture_trace
from kvprof.cli import build_parser, main
from kvprof.config import DEFAULT_TRACE_DIR, DEFAULT_UPSTREAM, resolve


def test_defaults_and_env():
    a = build_parser().parse_args(["analyze"])
    s = resolve(a, environ={})
    assert (s.upstream, s.trace_dir, s.use_log, s.template_profile) == (DEFAULT_UPSTREAM, DEFAULT_TRACE_DIR, False, "auto")
    s = resolve(a, environ={"KVPROF_UPSTREAM": "http://h:1", "KVPROF_TRACE_DIR": "/t", "KVPROF_SERVER_LOG": "/l"})
    assert (s.upstream, s.trace_dir, s.server_log, s.use_log) == ("http://h:1", "/t", "/l", True)
    a = build_parser().parse_args(["analyze", "--upstream", "http://x:2", "--server-log", "/l", "--no-log", "--log-anchor", "12.5"])
    s = resolve(a, environ={"KVPROF_UPSTREAM": "http://h:1"})
    assert (s.upstream, s.use_log, s.log_anchor) == ("http://x:2", False, 12.5)
    a = build_parser().parse_args(["tap", "--record-paths", "/v1/messages, /x"])
    assert resolve(a, environ={}).record_paths == ("/v1/messages", "/x")


def test_help_mentions_prog(capsys):
    main([])
    assert "kvprof tap" in capsys.readouterr().out


def _trace_dir(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    for n in ("anthropic_cold", "anthropic_tool_roundtrip"):
        tr = load_fixture_trace(n)
        (d / (tr["id"] + ".json")).write_text(json.dumps(tr))
    return str(d)


def test_analyze_cli(fake_upstream, tmp_path, capsys):
    d = _trace_dir(tmp_path)
    main(["analyze", "--upstream", fake_upstream.url, "--trace-dir", d])
    out = capsys.readouterr().out
    assert "no server log (pass --server-log" in out and "#01 cold" in out and "#02 tool_roundtrip" in out
    main(["analyze", "--upstream", fake_upstream.url, "--trace-dir", d, "--json", "--last", "1"])
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1 and rows[0]["kind"] == "tool_roundtrip"
    main(["analyze", "--upstream", fake_upstream.url, "--trace-dir", d, "--dump", "1"])
    assert "⟦system.text" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["analyze", "--upstream", fake_upstream.url, "--trace-dir", d, "--dump", "99"])


def test_calibrate_cli(fake_upstream, tmp_path, capsys):
    main(["calibrate", "--upstream", fake_upstream.url, "--trace-dir", _trace_dir(tmp_path)])
    out = capsys.readouterr().out
    assert "tool_result_join       -> ''" in out and "ok rendered" in out


def test_unreachable_upstream(tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["analyze", "--upstream", "http://127.0.0.1:9", "--trace-dir", str(tmp_path)])
    assert "cannot reach" in str(e.value.code)
