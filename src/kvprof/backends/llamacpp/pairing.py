"""Pair recorded requests with llama-server log tasks (by token counts first, then by time)."""
import time


def _plausible(t, n_in, n_out):
    """Can log task t be the server-side of a request with n_in prompt tokens and n_out output tokens?"""
    if t["pp_tokens"] is not None and t["pp_tokens"] > n_in + 16:
        return False
    if any(smp[1] > n_in + 16 for smp in t["pp_samples"]):
        return False
    if n_out is not None and t["n_tokens_final"] is not None and not t["cancelled"] and t["n_tokens_final"] > n_in + n_out + 2:
        return False
    return True


def pair_with_log(analyses, logdata, anchor):
    """Two passes: (1) requests with a complete response matched by token counts (exact), which also refines
    the wall-clock anchor (ps lstart is truncated to whole seconds); (2) the rest by plausibility and time.

    Reads each analysis's n_tokens, timing, outcome and normalized usage (total/cached/output_tokens)."""
    tasks = [dict(t) for t in logdata["tasks"]]
    last_seg = logdata["segments"] - 1
    used = set()
    for a in analyses:
        a["log"], a["log_match"], a["concurrent_with"] = None, "none", []
    if anchor is None:
        return analyses

    def times(a):
        tm = a.get("timing") or {}
        t_req = tm.get("t_request")
        t_body = tm.get("t_request_body_complete") or t_req
        t_end = tm.get("t_first_delta") or tm.get("t_last_byte") or (t_req + 3600 if t_req else None)
        return t_req, t_body, t_end

    def keys(a):
        u = a.get("usage") or {}
        n_in = a["n_tokens"]
        n_out = u.get("output_tokens")
        pp_exp = (n_in - (u.get("cached") or 0)) if u.get("total") is not None else None
        return n_in, n_out, pp_exp

    def attach(a, t, match, anc):
        t_req, t_body, _ = times(a)
        n_in, n_out, pp_exp = keys(a)
        used.add(t["task_id"])
        a["log_match"] = match
        a["log"] = {
            "task_id": t["task_id"], "slot": t["slot"], "select": t["select"], "pp_tokens": t["pp_tokens"],
            "pp_s": round(t["pp_ms"] / 1000, 2) if t["pp_ms"] is not None else None, "pp_rate": t["pp_rate"],
            "tg_tokens": t["tg_tokens"], "tg_s": round(t["tg_ms"] / 1000, 2) if t["tg_ms"] is not None else None,
            "tg_rate": t["tg_rate"], "n_tokens_final": t["n_tokens_final"], "truncated": t["truncated"],
            "cancelled": t["cancelled"], "reused_from_log": t.get("reused_from_log"), "prev_slot_n_tokens": t["prev_slot_n_tokens"],
            "evictions_mib": t["evictions_mib"], "t_launch": anc + t["t_launch"],
            "queue_wait_s": round(anc + t["t_launch"] - t_body, 3),
            "server_ttft_s": round(t["t_pp_done"] - t["t_launch"], 3) if t["t_pp_done"] else None,
            "pp_samples": [[round(anc + smp[0], 3), smp[1], smp[2], smp[3], smp[4]] for smp in t["pp_samples"]],
            "residual_pp": (t["pp_tokens"] - pp_exp) if (pp_exp is not None and t["pp_tokens"] is not None) else None,
            "residual_total": (t["n_tokens_final"] - (n_in + n_out)) if (n_out is not None and t["n_tokens_final"] is not None) else None,
        }

    # pass 1: complete responses, content keys, generous window
    lags = []
    for a in analyses:
        t_req, t_body, t_end = times(a)
        if not t_req or a.get("outcome") != "ok":
            continue
        n_in, n_out, pp_exp = keys(a)
        if pp_exp is None:
            continue
        best = None
        for t in tasks:
            if t["segment"] != last_seg or t["task_id"] in used or t.get("is_child") or t["cancelled"]:
                continue
            tl = anchor + t["t_launch"]
            if not (t_body - 3.0 <= tl <= t_end + 3.0) or not _plausible(t, n_in, n_out):
                continue
            s_total = 0 if (n_out is not None and t["n_tokens_final"] is not None and abs(t["n_tokens_final"] - (n_in + n_out)) <= 2) else 1
            d_pp = abs(t["pp_tokens"] - pp_exp) if t["pp_tokens"] is not None else 10 ** 6
            d_tg = abs(t["tg_tokens"] - n_out) if (n_out is not None and t["tg_tokens"] is not None) else 10 ** 6
            if s_total == 0 or (d_pp <= 16 and d_tg <= 2):
                score = (s_total, d_pp, d_tg, abs(tl - t_body))
                if best is None or score < best[0]:
                    best = (score, t)
        if best:
            score, t = best
            attach(a, t, "exact" if score[0] == 0 else "approx", anchor)
            lags.append(anchor + t["t_launch"] - t_body)
    shift = 0.0
    if lags:
        m = min(lags)
        if m < 0:
            shift = min(1.5, -m) + 0.01
    anc = anchor + shift
    if shift:
        for a in analyses:
            if a.get("log"):
                a["log"]["t_launch"] += shift
                a["log"]["queue_wait_s"] = round(a["log"]["queue_wait_s"] + shift, 3)
                for smp in a["log"]["pp_samples"]:
                    smp[0] = round(smp[0] + shift, 3)
    # pass 2: everything else (cancelled, inflight, mismatched) with the refined anchor and a tight window
    for a in analyses:
        if a.get("log"):
            continue
        t_req, t_body, t_end = times(a)
        if not t_req:
            continue
        n_in, n_out, pp_exp = keys(a)
        cancelled_req = a.get("outcome") not in ("ok", None)
        best = None
        for t in tasks:
            if t["segment"] != last_seg or t["task_id"] in used or t.get("is_child"):
                continue
            tl = anc + t["t_launch"]
            if not (t_body - 0.35 <= tl <= t_end + 1.0) or not _plausible(t, n_in, n_out):
                continue
            d_pp = abs(t["pp_tokens"] - pp_exp) if (pp_exp is not None and t["pp_tokens"] is not None) else 10 ** 6
            score = (0 if t["cancelled"] == cancelled_req else 1, d_pp if d_pp < 10 ** 6 else 0, abs(tl - t_body))
            if best is None or score < best[0]:
                best = (score, t)
        if best:
            score, t = best
            if cancelled_req:
                match = "cancelled" if t["cancelled"] else "time-only"
            elif score[1] <= 16 and pp_exp is not None and t["pp_tokens"] is not None:
                match = "approx"
            elif score[1] <= 64 and pp_exp is not None and t["pp_tokens"] is not None:
                match = "loose"
            else:
                match = "time-only"
            attach(a, t, match, anc)
    mark_concurrency(analyses)
    logdata["anchor_shift"] = shift
    return analyses


def mark_concurrency(analyses):
    """concurrent_with: ids of requests whose request..last-byte interval overlaps this one's."""
    now = time.time()
    for a in analyses:
        a["concurrent_with"] = []
        tm = a.get("timing") or {}
        if not tm.get("t_request"):
            continue
        s1, e1 = tm["t_request"], tm.get("t_last_byte") or now
        a["concurrent_with"] = [b["id"] for b in analyses if b is not a and (b.get("timing") or {}).get("t_request")
                                and max(s1, b["timing"]["t_request"]) < min(e1, b["timing"].get("t_last_byte") or now)]
