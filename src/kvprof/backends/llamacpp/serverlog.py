"""llama-server log parsing: per-task slot selection, prefill and decode timings, release sizes.

The log's timestamps count from server start (`MM.SS.mmm.uuu`, first field in total minutes), so pairing
log tasks with recorded requests needs a wall-clock anchor (`log_anchor`).
"""
import os
import re
import subprocess
import time

# Log formats this parser was verified against (build_info prefix from /props).
TESTED_BUILDS = ("b10909",)

# KEEP IN SYNC with tests/fixtures/server.log and TESTED_BUILDS: a format change must fail a test.
LOG_PATTERNS = {
    "ts":      re.compile(r"^(\d+)\.(\d\d)\.(\d{3})\.(\d{3}) ([IWE]) "),
    "select":  re.compile(r"id\s+(\d+) \| task -1 \| selected slot by (?:(LCP) similarity, f_sim_best = ([0-9.]+).*?f_keep = ([0-9.]+)|(LRU), t_last = (-?\d+))"),
    "launch":  re.compile(r"id\s+(\d+) \| task (\d+) \| processing task, is_child = (\d)"),
    "pp":      re.compile(r"id\s+(\d+) \| task (\d+) \| prompt processing, n_tokens =\s*(\d+), progress = ([0-9.]+), t =\s*([0-9.]+) s / ([0-9.]+) tokens per second"),
    "tg":      re.compile(r"id\s+(\d+) \| task (\d+) \| (?:n_decoded|n_gen) =\s*(\d+), tg =\s*([0-9.]+) t/s, tg_3s =\s*([0-9.]+)"),
    "pp_done": re.compile(r"id\s+(\d+) \| task (\d+) \| prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens \(.*?([0-9.]+) tokens per second"),
    "tg_done": re.compile(r"id\s+(\d+) \| task (\d+) \|\s+eval time =\s*([0-9.]+) ms /\s*(\d+) tokens \(.*?([0-9.]+) tokens per second"),
    "release": re.compile(r"id\s+(\d+) \| task (\d+) \| stop processing: n_tokens = (\d+), truncated = (\d)"),
    "cancel":  re.compile(r"cancel task, id_task = (\d+)"),
    "alloc":   re.compile(r"making room for prompt cache entry, removing oldest entry \(size = ([0-9.]+) MiB\)"),
    "schema_warn": re.compile(r"^WARNING: JSON schema conversion was incomplete"),
    "err":     re.compile(r" E .*?(error: .*|failed to .*)$"),
}
# Slot lines that carry no information the analyzer uses; `doctor` does not count them as format drift.
IGNORED_SLOT_LINES = re.compile(r"\| task -?\d+ \|\s+(?:total time =|graphs reused =)")


def _ts_from_match(m):
    return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1e3 + int(m.group(4)) / 1e6


def parse_log_ts(s):
    """'24.08.912.329' -> 1448.912329 seconds since server start (first field is total minutes)."""
    m = LOG_PATTERNS["ts"].match(s if s.endswith(" ") else s + " I ")
    if not m:
        raise ValueError(s)
    return _ts_from_match(m)


def new_task(task_id, slot, t, segment):
    return {"task_id": task_id, "slot": slot, "t_launch": t, "t_pp_done": None, "t_tg_done": None, "t_release": None,
            "select": None, "pp_samples": [], "pp_ms": None, "pp_tokens": None, "pp_rate": None,
            "tg_ms": None, "tg_tokens": None, "tg_rate": None, "n_tokens_final": None, "truncated": None,
            "cancelled": False, "t_cancel": None, "evictions_mib": [], "prev_slot_n_tokens": None,
            "segment": segment, "is_child": 0, "line": None}


def parse_server_log(path):
    """-> {"tasks": [task dicts in launch order], "last_ts": float, "segments": int, "schema_warnings": int,
           "errors": [str]}. A timestamp that goes backwards starts a new segment (server restart)."""
    tasks, by_id = [], {}
    pending_select = {}
    last_release = {}
    last_task = None
    prev_ts, segment, schema_warnings, errors = -1.0, 0, 0, []
    P = LOG_PATTERNS
    try:
        f = open(path, errors="replace")
    except (OSError, TypeError):
        return {"tasks": [], "last_ts": None, "segments": 0, "schema_warnings": 0, "errors": []}
    with f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if P["schema_warn"].match(line):
                schema_warnings += 1
                continue
            m = P["ts"].match(line)
            if not m:
                continue
            t = _ts_from_match(m)
            if t < prev_ts - 1.0:
                segment += 1
                pending_select, last_release, last_task = {}, {}, None
            prev_ts = t
            rest = line[m.end():]
            mm = P["launch"].search(rest)
            if mm:
                slot, tid = int(mm.group(1)), int(mm.group(2))
                task = new_task(tid, slot, t, segment)
                task["is_child"] = int(mm.group(3))
                task["line"] = lineno
                task["select"] = pending_select.pop(slot, None)
                task["prev_slot_n_tokens"] = last_release.get(slot)
                tasks.append(task); by_id[(segment, tid)] = task; last_task = task
                continue
            mm = P["select"].search(rest)
            if mm:
                slot = int(mm.group(1))
                if mm.group(2):
                    pending_select[slot] = {"by": "LCP", "f_sim": float(mm.group(3)), "f_keep": float(mm.group(4))}
                else:
                    pending_select[slot] = {"by": "LRU", "t_last": int(mm.group(6))}
                continue
            mm = P["pp"].search(rest)
            if mm:
                task = by_id.get((segment, int(mm.group(2))))
                if task:
                    task["pp_samples"].append([t, int(mm.group(3)), float(mm.group(4)), float(mm.group(5)), float(mm.group(6))])
                continue
            mm = P["tg"].search(rest)
            if mm:
                continue
            mm = P["pp_done"].search(rest)
            if mm:
                task = by_id.get((segment, int(mm.group(2))))
                if task:
                    task["pp_ms"], task["pp_tokens"], task["pp_rate"] = float(mm.group(3)), int(mm.group(4)), float(mm.group(5))
                    task["t_pp_done"] = t
                continue
            mm = P["tg_done"].search(rest)
            if mm:
                task = by_id.get((segment, int(mm.group(2))))
                if task:
                    task["tg_ms"], task["tg_tokens"], task["tg_rate"] = float(mm.group(3)), int(mm.group(4)), float(mm.group(5))
                    task["t_tg_done"] = t
                continue
            mm = P["release"].search(rest)
            if mm:
                slot, tid = int(mm.group(1)), int(mm.group(2))
                task = by_id.get((segment, tid))
                n = int(mm.group(3))
                last_release[slot] = n
                if task:
                    task["n_tokens_final"], task["truncated"], task["t_release"] = n, int(mm.group(4)), t
                continue
            mm = P["cancel"].search(rest)
            if mm:
                task = by_id.get((segment, int(mm.group(1))))
                if task:
                    task["cancelled"], task["t_cancel"] = True, t
                continue
            mm = P["alloc"].search(rest)
            if mm:
                if last_task:
                    last_task["evictions_mib"].append(float(mm.group(1)))
                continue
            mm = P["err"].search(line)
            if mm:
                errors.append(f"{lineno}: {mm.group(1)[:200]}")
    for task in tasks:
        sel = task["select"]
        if sel and sel.get("by") == "LCP" and task["prev_slot_n_tokens"]:
            task["reused_from_log"] = round(sel["f_keep"] * task["prev_slot_n_tokens"])
        else:
            task["reused_from_log"] = 0 if sel else None
    return {"tasks": tasks, "last_ts": prev_ts if prev_ts >= 0 else None, "segments": segment + 1,
            "schema_warnings": schema_warnings, "errors": errors}


def scan_log_format(path, max_examples=5):
    """Format-drift check for `doctor`: how many lines each key pattern matched, and slot-looking lines
    (containing ' | task ') that matched no pattern. -> {"counts": {...}, "unmatched": n, "examples": [...]}"""
    keys = ("launch", "select", "pp", "pp_done", "tg_done", "release")
    counts = {k: 0 for k in keys}
    unmatched, examples = 0, []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = LOG_PATTERNS["ts"].match(line)
            if not m or " | task " not in line:
                continue
            rest = line[m.end():]
            hit = False
            for k in keys:
                if LOG_PATTERNS[k].search(rest):
                    counts[k] += 1
                    hit = True
                    break
            if not hit and not LOG_PATTERNS["tg"].search(rest) and not IGNORED_SLOT_LINES.search(rest):
                unmatched += 1
                if len(examples) < max_examples:
                    examples.append(rest[:160])
    return {"counts": counts, "unmatched": unmatched, "examples": examples}


def log_anchor(pid_file=None, log_path=None, last_ts=None, override=None):
    """Epoch seconds corresponding to log timestamp 0.0 -> (epoch, source).

    Sources in order: an explicit override; `ps -o lstart` of the pid in pid_file (only when given; whole
    seconds, refined later by pairing); the log file's mtime minus its last timestamp."""
    if override is not None:
        return float(override), "arg"
    if pid_file:
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                return time.mktime(time.strptime(out, "%a %b %d %H:%M:%S %Y")), "ps"
        except Exception:
            pass
    try:
        if last_ts is not None and log_path:
            return os.path.getmtime(log_path) - last_ts, "mtime"
    except OSError:
        pass
    return None, "none"
