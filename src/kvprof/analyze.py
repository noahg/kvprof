"""Analysis of recorded requests: render, tokenize, map sections, diff against earlier prompts, classify,
pair with server telemetry. Analyses are cached next to the traces as <id>.analysis.json."""
import sys
import threading
import time

from . import __version__
from .backends import get_backend
from .convert import load_variant
from .diff import classify, diff_analysis
from .request import Request, client_info, detect_api
from .sections import add_raw_counts, assign_tokens, build_sections
from .sections.profiles import PROFILES, detect_profile
from .store import RawCache, TraceStore
from .usage import upgrade_schema1_usage
from .util import byte_starts, char_to_byte_table, lcp_len

ANALYSIS_SCHEMA = 7


def trace_api(trace):
    """Wire shape of a recorded request; schema-1 traces predate the field and were all Anthropic."""
    return trace.get("api") or detect_api(trace.get("path")) or "anthropic"


def trace_raw_usage(trace):
    u = trace.get("usage")
    if not u:
        return None
    if trace.get("schema", 1) < 2:
        return upgrade_schema1_usage(u, trace.get("stop_reason"))
    return u


def exactness(usage, n_rendered):
    """Does our rendering have exactly the server's prompt token count? usage: normalized Usage dict."""
    raw, cached = usage.get("raw_prompt"), usage.get("cached")
    ex = {"n_rendered": n_rendered, "n_usage": raw, "cache_read": cached,
          "usage": None, "usage_semantics": None, "count_tokens": None, "n_count_tokens": None}
    if raw is not None:
        if raw == n_rendered:
            ex["usage"], ex["usage_semantics"] = True, "total"
        elif (cached or 0) + raw == n_rendered:
            ex["usage"], ex["usage_semantics"] = True, "uncached"
        else:
            ex["usage"] = False
            ex["delta"] = raw - n_rendered
    return ex


def _apply_usage(a, trace, backend):
    raw = trace_raw_usage(trace)
    usage = backend.usage.normalize(raw, a["n_tokens"]).to_dict()
    a["usage_raw"] = raw
    a["usage"] = usage
    a["stop_reason"] = usage.get("stop_reason") or trace.get("stop_reason")
    a["exact"] = exactness(usage, a["n_tokens"])
    if usage.get("total") is not None:
        a["prefilled_usage"], a["reused_usage"] = usage.get("uncached"), usage.get("cached") or 0
    else:
        a["prefilled_usage"] = a["reused_usage"] = None


def analyze_trace(trace, prev, backend, request, raw, request_of, profile):
    """trace: loaded trace json; prev: earlier analyses (oldest..newest, each with ids/leaves/rendered);
    request_of(id) -> Request of an earlier trace, for diffing against its body."""
    body = request.body
    t0 = time.time()
    headers = trace.get("request", {}).get("headers") or {}
    ci = client_info(headers, body)
    messages, tools, origins, flags = request.to_openai()
    units = request.render_units()
    rendered = backend.renderer.render(messages, tools)
    toks = backend.tokenizer.tokenize(rendered)
    ids = [t for t, _ in toks]
    bstart = byte_starts(toks)
    rb = rendered.encode("utf-8")
    mapping_ok = bstart[-1] == len(rb)
    reason = None if mapping_ok else f"pieces concatenate to {bstart[-1]} bytes, rendered is {len(rb)}"
    c2b = char_to_byte_table(rendered)
    tree, sec_ok, sec_reason = build_sections(profile, rendered, request, messages, origins, units)
    if not sec_ok:
        mapping_ok, reason = False, sec_reason
    leaves = assign_tokens(tree, c2b, bstart, len(ids))
    add_raw_counts(tree, request.unit_text, raw)
    a = {
        "schema": ANALYSIS_SCHEMA, "id": trace["id"], "seq": trace.get("seq"), "api": request.api,
        "stage": 2 if trace.get("outcome") != "inflight" else 1,
        "outcome": trace.get("outcome"), "timing": trace.get("timing"), "model": trace.get("model"),
        "n_tokens": len(ids), "n_messages": request.n_messages, "n_tools": request.n_tools,
        "max_tokens": request.max_tokens, "flags": flags, "mapping_ok": mapping_ok, "mapping_reason": reason,
        "profile": profile, "client": headers.get("user-agent"),
        "client_name": ci["client"], "client_version": ci["version"], "session_id": ci["session"], "session_source": ci["session_source"],
        "session_inferred": False,
        "sections": tree, "leaves": leaves, "ids": ids, "bstart": bstart, "rendered": rendered,
        "analysis_s": None,
    }
    _apply_usage(a, trace, backend)
    a["raw"] = {"total": sum(n["raw_tokens"] or 0 for n in tree), "by_group": {n["id"]: n["raw_tokens"] for n in tree},
                "overhead": len(ids) - sum(n["raw_tokens"] or 0 for n in tree)}
    # diff against the closest earlier prompt
    diff, parent = None, None
    if prev:
        # candidates: every earlier request of the same explicit session, plus the last 8 overall
        cands = [p for p in prev if ci["session"] and p.get("session_id") == ci["session"]]
        for p in prev[-8:]:
            if p not in cands:
                cands.append(p)
        # prefer a prompt this one extends (what the server can reuse whole), then the longest shared prefix,
        # then the same session; later requests win ties
        best = None
        for p in cands:
            n = lcp_len(ids, p["ids"])
            key = (n >= len(p["ids"]) - 2, n, bool(ci["session"]) and p.get("session_id") == ci["session"])
            if best is None or key >= best[0]:
                best = (key, p)
        parent = best[1]
        diff = diff_analysis(a, parent, request, request_of(parent["id"]))
        a["prev_id"] = prev[-1]["id"]
        a["parent_id"] = parent["id"]
        parent_out = (parent.get("usage") or {}).get("output_tokens") or 0
        diff["expected_prefill_max"] = len(ids) - diff["lcp"]
        diff["expected_prefill_min"] = max(0, len(ids) - diff["lcp"] - parent_out)
    else:
        a["prev_id"] = a["parent_id"] = None
    a["diff"] = diff
    if not a["session_id"]:
        # no session id from the client: inherit the parent's session when most of the prompt is shared, else start one
        if parent and diff and diff["lcp"] >= 0.5 * len(ids) and parent.get("client_name") == ci["client"]:
            a["session_id"], a["session_inferred"] = parent.get("session_id"), True
        else:
            a["session_id"], a["session_inferred"] = trace["id"], True
    a["kind"], a["subkind"], a["hint"] = classify(a, request, diff, parent)
    a["prompt_total"] = len(ids)
    a["log"] = None
    a["log_match"] = "none"
    a["analysis_s"] = round(time.time() - t0, 3)
    return a


def update_from_trace(old, trace, backend):
    """Stage-1 analysis whose trace is now complete: refresh outcome, timing and usage without re-rendering."""
    a = dict(old)
    a.update({"stage": 2 if trace.get("outcome") != "inflight" else 1, "outcome": trace.get("outcome"),
              "timing": trace.get("timing")})
    _apply_usage(a, trace, backend)
    return a


def analysis_valid(meta, trace_mtime, trace_size, model_path, variant, profile, backend):
    if not meta or meta.get("schema") != ANALYSIS_SCHEMA:
        return False
    if meta.get("model_path") != model_path or meta.get("variant") != variant:
        return False
    if meta.get("profile") != profile or meta.get("backend") != backend:
        return False
    if meta.get("trace_mtime") != trace_mtime or meta.get("trace_size") != trace_size:
        return False
    return True


class UpstreamUnavailable(RuntimeError):
    pass


class Pipeline:
    """Loads traces, keeps analyses (cached on disk), pairs them with server telemetry."""

    def __init__(self, settings, backend=None, store=None, verbose=False):
        self.settings = settings
        self.verbose = verbose
        self.backend = backend or get_backend(settings.backend, upstream=settings.upstream, api_key=settings.api_key)
        self.store = store or TraceStore(settings.trace_dir)
        self.info = self.backend.server_info()
        if "props_error" in self.info.extra and not self.info.model_path:
            raise UpstreamUnavailable(f"cannot reach {settings.upstream}: {self.info.extra['props_error']}")
        self.model_path = self.info.model_path
        self.model = settings.model or self.backend.default_model()
        tp = settings.template_profile or "auto"
        if tp != "auto" and tp not in PROFILES:
            raise ValueError(f"unknown template profile {tp!r} (choose from auto, {', '.join(PROFILES)})")
        self.profile = detect_profile(self.info.chat_template) if tp == "auto" else tp
        self.variant = load_variant(self.backend, self.store, self.model, verbose=verbose)
        self.raw = RawCache(self.store, self.backend.tokenizer)
        self.telemetry = self.backend.telemetry(settings.server_log, settings.server_pid) if settings.use_log else None
        self.analyses = []          # in trace order
        self.by_id = {}
        self.requests = {}          # id -> Request
        self.anchor, self.anchor_src = None, "none"
        self.logdata = None
        self.lock = threading.Lock()
        self.last_fingerprint = None

    def _meta(self, st, trace):
        return {"schema": ANALYSIS_SCHEMA, "version": __version__, "trace_mtime": st.st_mtime, "trace_size": st.st_size,
                "model_path": self.model_path, "variant": self.variant, "profile": self.profile, "backend": self.backend.name,
                "body_sha256": trace["request"].get("body_sha256"), "analyzed_at": time.time()}

    def _valid(self, a, st):
        return a is not None and analysis_valid(a.get("meta"), st.st_mtime, st.st_size, self.model_path, self.variant,
                                                self.profile, self.backend.name)

    def _request(self, tid, trace=None):
        r = self.requests.get(tid)
        if r is None:
            try:
                trace = trace or self.store.load_trace(tid)
            except Exception:
                return None
            r = Request(trace["request"]["body"], trace_api(trace), self.variant)
            self.requests[tid] = r
        return r

    def request(self, tid):
        return self._request(tid)

    def refresh_all(self):
        """Analyze new/changed traces, re-pair with the log. Returns True if anything changed."""
        changed = False
        prev = []
        new_list = []
        for tid in self.store.list_ids():
            try:
                st = self.store.stat(tid)
            except OSError:
                continue
            a = self.by_id.get(tid)
            if not self._valid(a, st):
                a = None if self.settings.refresh else self.store.load_analysis(tid)
                if not self._valid(a, st):
                    a = None
                    try:
                        trace = self.store.load_trace(tid)
                    except Exception:
                        continue
                    body = trace.get("request", {}).get("body")
                    if not isinstance(body, dict) or "messages" not in body:
                        continue
                    req = Request(body, trace_api(trace), self.variant)
                    self.requests[tid] = req
                    old = self.by_id.get(tid)
                    if old and old.get("meta", {}).get("body_sha256") == trace["request"].get("body_sha256") and old.get("stage") == 1:
                        a = update_from_trace(old, trace, self.backend)
                    else:
                        try:
                            a = analyze_trace(trace, prev, self.backend, req, self.raw, self._request, self.profile)
                        except Exception as e:
                            if self.verbose:
                                print(f"analysis failed for {tid}: {e!r}", file=sys.stderr)
                            continue
                    a["meta"] = self._meta(st, trace)
                    self._save(a)
                changed = True
            new_list.append(a)
            prev.append(a)
        self.raw.save()
        with self.lock:
            self.analyses = new_list
            self.by_id = {a["id"]: a for a in new_list}
        if self.telemetry is not None:
            logdata = self.telemetry.load()
            anchor, src = self.telemetry.anchor(logdata, self.settings.log_anchor)
            before = [(a.get("log") or {}).get("task_id") for a in new_list]
            with self.lock:
                self.logdata, self.anchor, self.anchor_src = logdata, anchor, src
                self.telemetry.pair(new_list, logdata, anchor)
            after = [(a.get("log") or {}).get("task_id") for a in new_list]
            if before != after:
                changed = True
                for a, b0, a0 in zip(new_list, before, after):
                    if b0 != a0:
                        self._save(a)
        else:
            from .backends.llamacpp.pairing import mark_concurrency
            with self.lock:
                for a in new_list:
                    a["log"], a["log_match"] = None, "none"
                mark_concurrency(new_list)
        return changed

    def _save(self, a):
        try:
            self.store.save_analysis(a)
        except OSError as e:
            if self.verbose:
                print(f"could not save analysis: {e}", file=sys.stderr)

    def fingerprint(self):
        return self.store.fingerprint() + "-" + (self.telemetry.fingerprint() if self.telemetry else "")
