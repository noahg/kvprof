import os

from conftest import FIXTURES
from kvprof.backends.llamacpp.serverlog import LOG_PATTERNS, log_anchor, parse_log_ts, parse_server_log, scan_log_format

LOG = os.path.join(FIXTURES, "server.log")


def test_ts():
    assert abs(parse_log_ts("24.08.912.329") - 1448.912329) < 1e-9
    assert abs(parse_log_ts("5.02.257.960 I slot") - 302.25796) < 1e-9


def test_select_patterns():
    m = LOG_PATTERNS["select"].search("id  0 | task -1 | selected slot by LCP similarity, f_sim_best = 0.979 (> 0.100 thold), f_keep = 1.000")
    assert m and m.group(2) == "LCP" and m.group(4) == "1.000"
    m = LOG_PATTERNS["select"].search("id  1 | task -1 | selected slot by LRU, t_last = -1")
    assert m and m.group(5) == "LRU"


def test_parse_fixture_log():
    d = parse_server_log(LOG)
    assert d["segments"] == 2
    tasks = d["tasks"]
    assert [t["task_id"] for t in tasks] == [0, 25, 40, 0]
    t0, t25, t40, restart = tasks
    assert t0["select"] == {"by": "LRU", "t_last": -1} and t0["slot"] == 1
    assert t0["pp_tokens"] == 4000 and t0["pp_ms"] == 4100.0 and t0["tg_tokens"] == 20 and t0["n_tokens_final"] == 4019
    assert len(t0["pp_samples"]) == 1 and t0["pp_samples"][0][1] == 2048
    assert t0["reused_from_log"] == 0
    assert t25["select"]["by"] == "LCP" and t25["select"]["f_keep"] == 1.0
    assert t25["prev_slot_n_tokens"] == 4019 and t25["reused_from_log"] == 4019
    assert t40["cancelled"] and t40["evictions_mib"] == [512.25]
    assert restart["segment"] == 1 and restart["select"]["by"] == "LRU" and restart["prev_slot_n_tokens"] is None
    assert len(d["errors"]) == 1 and "exceeds the available context size" in d["errors"][0]
    assert abs(d["last_ts"] - 0.5001) < 1e-6


def test_missing_log():
    assert parse_server_log("/nonexistent/server.log")["tasks"] == []
    assert parse_server_log(None)["tasks"] == []


def test_anchor(tmp_path):
    assert log_anchor(override=123.5) == (123.5, "arg")
    p = tmp_path / "server.log"
    p.write_text("x")
    os.utime(p, (1000.0, 1000.0))
    assert log_anchor(None, str(p), last_ts=100.0) == (900.0, "mtime")
    assert log_anchor(str(tmp_path / "missing.pid"), str(p), last_ts=100.0) == (900.0, "mtime")
    assert log_anchor(None, None, None) == (None, "none")


def test_scan_format():
    r = scan_log_format(LOG)
    assert r["counts"]["launch"] == 4 and r["counts"]["release"] == 2 and r["counts"]["select"] == 4
    assert r["unmatched"] == 0


def test_scan_format_drift(tmp_path):
    p = tmp_path / "server.log"
    p.write_text("1.00.000.000 I slot launch_slot_: id  0 | task 3 | started processing (new format)\n")
    r = scan_log_format(str(p))
    assert r["unmatched"] == 1 and r["counts"]["launch"] == 0
