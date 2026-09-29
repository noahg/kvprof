from kvprof.backends.llamacpp.pairing import mark_concurrency, pair_with_log
from kvprof.backends.llamacpp.serverlog import new_task

ANCHOR = 1_000_000.0


def task(tid, slot, t_launch, pp, tg, final, cancelled=False, select=None):
    t = new_task(tid, slot, t_launch, 0)
    t.update({"pp_tokens": pp, "pp_ms": pp * 1.0 if pp is not None else None, "pp_rate": 1000.0, "tg_tokens": tg,
              "tg_ms": tg * 20.0 if tg is not None else None, "tg_rate": 50.0, "n_tokens_final": final, "cancelled": cancelled,
              "select": select or {"by": "LRU", "t_last": -1}, "t_pp_done": t_launch + 1.0, "reused_from_log": 0})
    return t


def an(aid, t_req, n, cached, out, outcome="ok", t_last=None):
    return {"id": aid, "n_tokens": n, "outcome": outcome,
            "timing": {"t_request": t_req, "t_request_body_complete": t_req + 0.01, "t_first_delta": t_req + 2, "t_last_byte": t_last or t_req + 3},
            "usage": {"total": n if outcome == "ok" else None, "cached": cached, "output_tokens": out}}


def test_exact_and_cancelled():
    tasks = [task(0, 0, 10.5, 1000, 20, 1019), task(5, 0, 20.4, 100, 10, 1109, select={"by": "LCP", "f_sim": 0.9, "f_keep": 1.0}),
             task(9, 1, 30.2, None, None, None, cancelled=True)]
    tasks[2]["pp_samples"] = [[31.0, 512, 0.5, 0.8, 640.0]]
    logdata = {"tasks": tasks, "segments": 1}
    a1 = an("a1", ANCHOR + 10.0, 1000, 0, 20)
    a2 = an("a2", ANCHOR + 20.0, 1100, 1000, 10)
    a3 = an("a3", ANCHOR + 30.0, 5000, None, None, outcome="client_disconnect", t_last=ANCHOR + 31.5)
    pair_with_log([a1, a2, a3], logdata, ANCHOR)
    assert a1["log_match"] == "exact" and a1["log"]["task_id"] == 0 and a1["log"]["residual_pp"] == 0
    assert a2["log_match"] == "exact" and a2["log"]["select"]["by"] == "LCP" and a2["log"]["residual_total"] == -1
    assert a3["log_match"] == "cancelled" and a3["log"]["task_id"] == 9
    assert logdata["anchor_shift"] == 0.0


def test_anchor_refined_when_log_is_early():
    # ps lstart truncates to whole seconds: the log appears to launch before the request arrived
    logdata = {"tasks": [task(0, 0, 9.5, 1000, 20, 1019)], "segments": 1}
    a = an("a", ANCHOR + 10.0, 1000, 0, 20)
    pair_with_log([a], logdata, ANCHOR)
    assert a["log_match"] == "exact" and logdata["anchor_shift"] > 0.5
    assert a["log"]["queue_wait_s"] >= 0


def test_no_anchor_no_pairing():
    a = an("a", ANCHOR, 10, 0, 1)
    pair_with_log([a], {"tasks": [task(0, 0, 0.0, 10, 1, 10)], "segments": 1}, None)
    assert a["log"] is None and a["log_match"] == "none"


def test_implausible_task_rejected():
    logdata = {"tasks": [task(0, 0, 10.5, 5000, 20, 5019)], "segments": 1}
    a = an("a", ANCHOR + 10.0, 1000, 0, 20)
    pair_with_log([a], logdata, ANCHOR)
    assert a["log"] is None


def test_old_segment_ignored():
    t = task(0, 0, 10.5, 1000, 20, 1019)
    t["segment"] = 0
    a = an("a", ANCHOR + 10.0, 1000, 0, 20)
    pair_with_log([a], {"tasks": [t], "segments": 2}, ANCHOR)
    assert a["log"] is None


def test_concurrency():
    a = an("a", 100.0, 10, 0, 1, t_last=110.0)
    b = an("b", 105.0, 10, 0, 1, t_last=106.0)
    c = an("c", 111.0, 10, 0, 1, t_last=112.0)
    mark_concurrency([a, b, c])
    assert a["concurrent_with"] == ["b"] and b["concurrent_with"] == ["a"] and c["concurrent_with"] == []
