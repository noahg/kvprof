"""Human-readable output: one summary dict / line per request, the session report, a per-request dump."""
import json
import os
import sys
import time

from .sections import find_node, iter_nodes
from .util import fmt_n


def summarize(a):
    u = a.get("usage") or {}
    tm = a.get("timing") or {}
    lg = a.get("log") or {}
    d = a.get("diff") or {}
    fd = d.get("first_divergent") or {}
    groups = {n["id"]: n["tokens"] for n in a["sections"]}
    sel = lg.get("select") or {}
    prefilled = lg.get("pp_tokens") if lg.get("pp_tokens") is not None else a.get("prefilled_usage")
    return {
        "id": a["id"], "seq": a.get("seq"), "api": a.get("api"), "kind": a["kind"], "subkind": a.get("subkind"), "hint": a.get("hint"),
        "outcome": a.get("outcome"), "stage": a.get("stage"), "stop_reason": a.get("stop_reason"),
        "t_request": tm.get("t_request"), "t_first_delta": tm.get("t_first_delta"), "t_last_byte": tm.get("t_last_byte"),
        "ttft_s": tm.get("ttft_s"), "total_s": tm.get("total_s"),
        "prompt_tokens": a["n_tokens"], "input_tokens": u.get("raw_prompt"), "cache_read": u.get("cached"),
        "prefilled": prefilled, "reused": a.get("reused_usage"), "output_tokens": u.get("output_tokens"),
        "output_estimated": u.get("output_estimated"),
        "lcp": d.get("lcp"), "appended": d.get("appended"), "parent_len": d.get("parent_len"),
        "diverge_tok": fd.get("tok"), "diverge_section": fd.get("cur_section"),
        "changed_added": (d.get("changed_text") or {}).get("added"), "changed_removed": (d.get("changed_text") or {}).get("removed"),
        "pp_s": lg.get("pp_s"), "pp_rate": lg.get("pp_rate"), "tg_rate": lg.get("tg_rate"), "tg_tokens": lg.get("tg_tokens"),
        "slot": lg.get("slot"), "task_id": lg.get("task_id"), "select": sel.get("by"), "f_keep": sel.get("f_keep"),
        "reused_from_log": lg.get("reused_from_log"), "queue_wait_s": lg.get("queue_wait_s"), "evictions_mib": lg.get("evictions_mib"),
        "exact": (a.get("exact") or {}).get("usage"), "usage_semantics": (a.get("exact") or {}).get("usage_semantics"),
        "log_match": a.get("log_match"), "residual_pp": lg.get("residual_pp"),
        "parent_id": a.get("parent_id"), "prev_id": a.get("prev_id"), "concurrent_with": a.get("concurrent_with") or [],
        "sys_tokens": groups.get("system", 0), "tools_tokens": groups.get("tools", 0), "msgs_tokens": groups.get("messages", 0),
        "gen_tokens": groups.get("generation_prompt", 0), "raw_total": (a.get("raw") or {}).get("total"),
        "n_messages": a.get("n_messages"), "n_tools": a.get("n_tools"), "mapping_ok": a.get("mapping_ok"), "profile": a.get("profile"),
        "client": a.get("client"), "client_name": a.get("client_name"), "client_version": a.get("client_version"),
        "session_id": a.get("session_id"), "session_inferred": a.get("session_inferred"),
        "summary": summary_line(a),
    }


def summary_line(a):
    u = a.get("usage") or {}
    tm = a.get("timing") or {}
    lg = a.get("log") or {}
    d = a.get("diff") or {}
    n = fmt_n
    kind = a["kind"] + (f"({a['hint'][:18]})" if a.get("hint") else "")
    if a.get("outcome") not in ("ok", None):
        kind = a["outcome"]
    groups = {x["id"]: x["tokens"] for x in a["sections"]}
    if a["kind"] == "cold" or not d:
        what = (f"{n(a['n_tokens'])}-tok prompt (system {n(groups.get('system', 0))} / tools {n(groups.get('tools', 0))}"
                f" / messages {n(groups.get('messages', 0))})")
    elif d.get("end_extension"):
        what = f"+{n(d['appended'])} tok appended after a {n(d['lcp'])}-tok shared prefix"
    elif a["kind"] == "retry":
        what = f"identical to #{a['parent_id'][-4:]} ({n(a['n_tokens'])} tok)"
    else:
        fd = d.get("first_divergent") or {}
        ch = d.get("changed_text")
        detail = ""
        if ch:
            if ch["added"] and not ch["removed"]:
                detail = f' text changed: +"{_short(ch["added"])}"'
            elif ch["removed"] and not ch["added"]:
                detail = f' text changed: -"{_short(ch["removed"])}"'
            else:
                detail = f' text changed: "{_short(ch["removed"])}" -> "{_short(ch["added"])}"'
        what = f"diverged at tok {n(fd.get('tok'))} ({fd.get('cur_section')}{detail}); {n(a['n_tokens'])}-tok prompt, {n(d['lcp'])} shared"
    # server side
    if lg:
        sel = lg.get("select") or {}
        sel_s = f"f_keep {sel['f_keep']:.3f}" if sel.get("by") == "LCP" else ("LRU" if sel.get("by") == "LRU" else "slot?")
        if lg.get("pp_tokens") is not None:
            srv = f"{n(lg['pp_tokens'])} prefilled in {lg['pp_s']} s @ {(lg['pp_rate'] or 0):.0f} t/s (slot {lg['slot']}, {sel_s}"
        else:
            last_pp = lg["pp_samples"][-1][1] if lg.get("pp_samples") else None
            srv = f"prefill interrupted after {n(last_pp)} tok (slot {lg['slot']}, {sel_s}"
        if a.get("concurrent_with"):
            srv += f"; concurrent with {', '.join('#' + c[-4:] for c in a['concurrent_with'])}"
        srv += ")"
        if lg.get("tg_tokens") is not None:
            srv += f"; decode {n(lg['tg_tokens'])} tok @ {(lg['tg_rate'] or 0):.1f} t/s"
    elif u.get("total") is not None:
        srv = f"{n(a.get('prefilled_usage'))} prefilled per usage (cached {n(u.get('cached'))}; no log match)"
        if u.get("output_tokens") is not None:
            srv += f"; decode {n(u['output_tokens'])} tok"
            if u.get("decode_rate"):
                srv += f" @ {u['decode_rate']:.1f} t/s"
    else:
        srv = "no response recorded"
    ttft = f"TTFT {tm['ttft_s']:.1f} s" if tm.get("ttft_s") is not None else "TTFT -"
    ex = a.get("exact") or {}
    if ex.get("usage") is True:
        exs = "exact"
    elif ex.get("usage") is False:
        exs = f"MISMATCH rendered {n(ex['n_rendered'])} vs usage {n(ex['n_usage'])}"
    else:
        exs = "exact?"
    lm = a.get("log_match")
    if lm not in ("exact", "none", None):
        exs += f"; log {lm}" + (f" (pp {lg.get('residual_pp'):+d})" if lg.get("residual_pp") is not None else "")
    if a.get("outcome") == "client_disconnect":
        exs += f"; client disconnected after {n(u.get('output_tokens'))} tok"
    return f"#{(a.get('seq') or 0):02d} {kind:15s} {what} -> {srv}; {ttft}; {exs}"


def _short(s, n=48):
    s = s.replace("\n", "⏎")
    return s[:n] + ("…" if len(s) > n else "")


def _rel(path):
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def print_session(p, last=None, as_json=False, out=None):
    out = out or sys.stdout
    an = p.analyses
    if last:
        an = an[-last:]
    if as_json:
        print(json.dumps([summarize(a) for a in an], indent=1, ensure_ascii=False), file=out)
        return
    info = p.info
    build = (info.build or "?").split("-")[0]
    if p.telemetry is None:
        anchor = "no server log (pass --server-log for server-side prefill numbers)"
    else:
        anchor = f"log anchor from {p.anchor_src}" + (f" ({time.strftime('%H:%M:%S', time.localtime(p.anchor))})" if p.anchor else "")
    print(f"session: {len(p.analyses)} requests in {_rel(p.store.trace_dir)}, model {info.model_name} ({p.backend.name} {build}, "
          f"template {p.profile}), {anchor}", file=out)
    for a in an:
        print(summary_line(a), file=out)
    # footer
    by_kind = {}
    for a in an:
        lg = a.get("log") or {}
        k = by_kind.setdefault(a["kind"], {"n": 0, "pp_tok": 0, "pp_s": 0.0, "tg_tok": 0, "tg_s": 0.0})
        k["n"] += 1
        if lg.get("pp_tokens") is not None:
            k["pp_tok"] += lg["pp_tokens"]; k["pp_s"] += lg["pp_s"] or 0
        if lg.get("tg_tokens") is not None:
            k["tg_tok"] += lg["tg_tokens"]; k["tg_s"] += lg["tg_s"] or 0
    if by_kind and p.telemetry is not None:
        print("totals:", file=out)
        for kind, k in sorted(by_kind.items(), key=lambda kv: -kv[1]["pp_s"]):
            print(f"  {kind:15s} {k['n']:3d} req   prefill {fmt_n(k['pp_tok']):>8} tok in {k['pp_s']:7.1f} s   decode {fmt_n(k['tg_tok']):>6} tok in {k['tg_s']:6.1f} s", file=out)
    sessions = {}
    for a in p.analyses:
        k = a.get("session_id") or "?"
        e = sessions.setdefault(k, {"client": f"{a.get('client_name')} {a.get('client_version') or ''}".strip(), "n": 0,
                                    "first": a.get("seq") or 0, "inferred": a.get("session_inferred")})
        e["n"] += 1
    if len(sessions) > 1 or any(v["inferred"] for v in sessions.values()):
        print("sessions:", file=out)
        for k, v in sessions.items():
            print(f"  {k[:8]:8s} {v['client']:28s} {v['n']:3d} req from #{v['first']:02d}{'  (inferred from prompt lineage)' if v['inferred'] else ''}", file=out)
    if an:
        a = an[-1]
        toolsn = find_node(a["sections"], "tools")
        if toolsn:
            top = sorted([c for c in toolsn["children"] if c["kind"] == "tool"], key=lambda c: -c["tokens"])[:8]
            print(f"largest tools in #{a.get('seq') or 0:02d} ({fmt_n(toolsn['tokens'])} tok for {a['n_tools']} tools): "
                  + ", ".join(f"{c['label']} {fmt_n(c['tokens'])}" for c in top), file=out)


def find_analysis(p, which):
    if which.isdigit():
        n = int(which)
        return next((x for x in p.analyses if x.get("seq") == n), None) or (p.analyses[n - 1] if 0 < n <= len(p.analyses) else None)
    return p.by_id.get(which)


def dump_request(p, which, as_json=False, out=None):
    out = out or sys.stdout
    a = find_analysis(p, which)
    if a is None:
        raise KeyError(f"no request {which}")
    if as_json:
        print(json.dumps({k: v for k, v in a.items() if k not in ("ids", "bstart", "rendered")}, indent=1, ensure_ascii=False), file=out)
        return
    print(summary_line(a), file=out)
    print(f"sections ({a['n_tokens']} tokens; raw approximation {fmt_n(a['raw']['total'])}, template overhead {fmt_n(a['raw']['overhead'])}"
          f"; profile {a.get('profile')}):", file=out)
    for n in iter_nodes(a["sections"]):
        depth = n["id"].count(".") + (1 if n["id"].startswith(("tool:", "msg[")) else 0)
        raw = f" raw {fmt_n(n['raw_tokens'])}" if n["raw_tokens"] is not None else ""
        print(f"  {'  ' * depth}{n['id']:34s} {fmt_n(n['tokens']):>8} tok  {n['pct']:5.1f}%  @{fmt_n(n['cum'])}{raw}{' ~' if n['approx'] else ''}", file=out)
    print("rendered prompt with section markers:", file=out)
    marks = sorted(((n["char"][0], i, n) for i, n in enumerate(iter_nodes(a["sections"])) if not n["children"]), key=lambda x: (x[0], x[1]))
    R = a["rendered"]
    parts, pos = [], 0
    for cs, _, n in marks:
        parts.append(R[pos:cs])
        parts.append(f"⟦{n['id']} tok {fmt_n(n['tok'][0])}–{fmt_n(n['tok'][1])}⟧")
        pos = cs
    parts.append(R[pos:])
    print("".join(parts), file=out)
