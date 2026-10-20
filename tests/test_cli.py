import json

import pytest

from conftest import load_fixture_trace
from kvprof.cli import build_parser, main
from kvprof.config import DEFAULT_TRACE_DIR, DEFAULT_UPSTREAM, default_data_dir, resolve
from kvprof.store import model_dir


def test_defaults_and_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = build_parser().parse_args(["analyze"])
    s = resolve(a, environ={"KVPROF_DATA_DIR": str(tmp_path / "data")})
    assert (s.upstream, s.use_log, s.template_profile) == (DEFAULT_UPSTREAM, False, "auto")
    s = resolve(a, environ={"KVPROF_UPSTREAM": "http://h:1", "KVPROF_TRACE_DIR": "/t", "KVPROF_SERVER_LOG": "/l"})
    assert (s.upstream, s.trace_dir, s.server_log, s.use_log) == ("http://h:1", "/t", "/l", True)
    a = build_parser().parse_args(["analyze", "--upstream", "http://x:2", "--server-log", "/l", "--no-log", "--log-anchor", "12.5"])
    s = resolve(a, environ={"KVPROF_UPSTREAM": "http://h:1"})
    assert (s.upstream, s.use_log, s.log_anchor) == ("http://x:2", False, 12.5)
    a = build_parser().parse_args(["tap", "--record-paths", "/v1/messages, /x"])
    assert resolve(a, environ={}).record_paths == ("/v1/messages", "/x")


def test_trace_dir_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = {"KVPROF_DATA_DIR": str(tmp_path / "data")}
    parse = build_parser().parse_args
    where = lambda argv, environ=env: (lambda s: (s.trace_dir, s.per_model))(resolve(parse(argv), environ=environ))  # noqa: E731
    root = str(tmp_path / "data" / "traces")
    # nothing chosen: the per-user root, one subdirectory per model
    for cmd in ("start", "analyze", "serve", "calibrate"):
        assert where([cmd]) == (root, True), cmd
    # tap alone cannot ask the server which model it runs
    assert where(["tap"]) == (DEFAULT_TRACE_DIR, False)
    # a chosen directory is used as it is
    assert where(["start", "--trace-dir", "x"]) == ("x", False)
    assert where(["start"], dict(env, KVPROF_TRACE_DIR="/t")) == ("/t", False)
    # an existing ./.kvprof keeps working
    (tmp_path / DEFAULT_TRACE_DIR).mkdir()
    assert where(["start"]) == (DEFAULT_TRACE_DIR, False)
    assert default_data_dir({}).endswith("kvprof")


def test_model_dir():
    a, b = model_dir("/r", "/models/Qwen3 Coder (30B).Q4_K_M.gguf"), model_dir("/r", "/other/Qwen3 Coder (30B).Q4_K_M.gguf")
    assert a.startswith("/r/Qwen3-Coder-30B-.Q4_K_M-") and a != b and a == model_dir("/r", "/models/Qwen3 Coder (30B).Q4_K_M.gguf")
    assert model_dir("/r", None).startswith("/r/model-")


def test_start_flags():
    a = build_parser().parse_args(["start", "--listen", ":9001", "--port", "9002", "--server-log", "/l", "--no-open", "--exit-with-parent"])
    s = resolve(a, environ={"KVPROF_TRACE_DIR": "/t"})
    assert (s.listen, s.port, s.server_log, s.use_log, a.no_open, a.exit_with_parent) == (":9001", 9002, "/l", True, True, True)


def test_help_mentions_prog(capsys):
    main([])
    out = capsys.readouterr().out
    assert "kvprof start" in out and "kvprof tap" in out


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
