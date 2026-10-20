"""Per-connection recorder: parses both directions of a relayed connection and writes one trace per request."""
import hashlib
import itertools
import json
import os
import threading
import time
from collections import deque

from ..backends.llamacpp.usage import LlamaCppUsageNormalizer
from ..config import DEFAULT_RECORD_PATHS
from ..request import detect_api
from ..sse import SSEParser
from ..usage import StreamWatcher, extract_raw_usage
from ..util import display_path, fmt_n, log, now_hms, save_json_atomic
from .httpparser import HttpParser

TRACE_SCHEMA = 2
RECORD_PATHS = set(DEFAULT_RECORD_PATHS)
DROP_HEADERS = {"authorization", "x-api-key", "cookie", "proxy-authorization"}
MAX_BODY_KEEP = 1 << 20
_normalizer = LlamaCppUsageNormalizer()


class TapContext:
    """State shared by every connection of one tap: output dir, record paths, the request counter."""

    def __init__(self, out_dir, record_paths=None, quiet=False, locate=None):
        self.out_dir = out_dir
        self.record_paths = set(record_paths or RECORD_PATHS)
        self.quiet = quiet
        self.locate = locate   # () -> directory for a new connection, or None to stay (`start`, per-model roots)
        self._seq = itertools.count(self._next_seq(out_dir))
        self._lock = threading.Lock()

    def connection_dir(self):
        """Where a connection that starts now records."""
        if self.locate:
            try:
                d = self.locate()
            except Exception:
                d = None
            with self._lock:
                if d and d != self.out_dir:
                    os.makedirs(d, exist_ok=True)
                    self.out_dir = d
                    self._seq = itertools.count(self._next_seq(d))
        return self.out_dir

    @staticmethod
    def _next_seq(out_dir):
        """Continue numbering after the traces already in the directory."""
        best = 0
        try:
            for n in os.listdir(out_dir):
                if n.endswith(".json") and not n.endswith(".analysis.json") and not n.startswith("."):
                    tail = n[:-5].rsplit("-", 1)[-1]
                    if tail.isdigit():
                        best = max(best, int(tail))
        except OSError:
            pass
        return best + 1

    def next_seq(self):
        with self._lock:
            return next(self._seq)


class Recorder:
    def __init__(self, conn_id, ctx):
        self.conn_id, self.ctx = conn_id, ctx
        self.out_dir, self.quiet = ctx.connection_dir(), ctx.quiet
        self.lock = threading.Lock()
        self.disabled = False
        self.pending = deque()
        self.cur_req = None
        self.cur_resp = None
        self.req_parser = HttpParser(False, self._req_head, self._req_body, self._req_end)
        self.resp_parser = HttpParser(True, self._resp_head, self._resp_body, self._resp_end)

    # -- request side
    def _req_head(self, head, t):
        self.cur_req = {"head": head, "body": bytearray(), "t_request": t, "t_body_complete": None}
        self.resp_parser.set_no_body(head["method"] == "HEAD")

    def _req_body(self, data, t):
        if self.cur_req is not None and len(self.cur_req["body"]) < MAX_BODY_KEEP * 8:
            self.cur_req["body"] += data

    def _req_end(self, t):
        r = self.cur_req
        self.cur_req = None
        if r is None:
            return
        r["t_body_complete"] = t
        rec = self._new_record(r)
        self.pending.append(rec)
        if rec["recorded"]:
            self._write(rec)

    # -- response side
    def _resp_head(self, head, t):
        rec = self.pending[0] if self.pending else None
        is_sse = "text/event-stream" in head["headers"].get("content-type", "")
        self.cur_resp = {"rec": rec, "head": head, "t_headers": t, "raw": bytearray(), "bytes": 0,
                         "sse": SSEParser() if is_sse else None, "watch": StreamWatcher((rec or {}).get("api") or "anthropic"),
                         "linebuf": b"", "sse_lines": [], "t_last_byte": t}

    def _resp_body(self, data, t):
        cr = self.cur_resp
        if cr is None:
            return
        cr["bytes"] += len(data)
        cr["t_last_byte"] = t
        if cr["sse"] is not None:
            cr["linebuf"] += data
            while True:
                i = cr["linebuf"].find(b"\n")
                if i < 0:
                    break
                line, cr["linebuf"] = cr["linebuf"][:i], cr["linebuf"][i + 1:]
                line = line.decode("utf-8", "replace").rstrip("\r")
                cr["sse_lines"].append(line)
                for ev, d in cr["sse"].feed_line(line):
                    cr["watch"].observe(ev, d, t)
        elif len(cr["raw"]) < MAX_BODY_KEEP:
            cr["raw"] += data

    def _resp_end(self, t):
        cr = self.cur_resp
        self.cur_resp = None
        if cr is None:
            return
        rec = self.pending.popleft() if self.pending else None
        if rec is None:
            return
        self._finalize(rec, cr, "ok" if cr["head"]["status"] < 400 else "http_error", t)

    # -- connection end
    def on_close(self, who, t):
        with self.lock:
            if self.disabled:
                return
            try:
                self.resp_parser.close(t)
                cr = self.cur_resp
                self.cur_resp = None
                outcome = "client_disconnect" if who == "client" else "upstream_closed"
                if cr is not None and self.pending:
                    rec = self.pending.popleft()
                    self._finalize(rec, cr, outcome, t, complete=False)
                while self.pending:
                    rec = self.pending.popleft()
                    if rec["recorded"]:
                        rec["response"] = None
                        rec["outcome"] = outcome
                        rec["timing"]["t_last_byte"] = t
                        self._write(rec)
                        self._line(rec, None)
            except Exception as e:  # never let recording break the relay
                log(f"{now_hms()}  tap: recorder error on close (conn {self.conn_id}): {e!r}")

    # -- byte taps (called from the pump threads)
    def on_request_bytes(self, data, t):
        if self.disabled:
            return
        with self.lock:
            try:
                self.req_parser.feed(data, t)
            except Exception as e:
                self.disabled = True
                log(f"{now_hms()}  tap: request parser error (conn {self.conn_id}), recording disabled for this connection: {e!r}")

    def on_response_bytes(self, data, t):
        if self.disabled:
            return
        with self.lock:
            try:
                self.resp_parser.feed(data, t)
            except Exception as e:
                self.disabled = True
                log(f"{now_hms()}  tap: response parser error (conn {self.conn_id}), recording disabled for this connection: {e!r}")

    # -- records
    def _new_record(self, r):
        head = r["head"]
        recorded = head["path"] in self.ctx.record_paths and head["method"] == "POST"
        rec = {"recorded": recorded, "method": head["method"], "path": head["path"], "query": head["query"],
               "t_request": r["t_request"], "seq": None, "id": None}
        if not recorded:
            return rec
        seq = self.ctx.next_seq()
        t0 = r["t_request"]
        rec["seq"] = seq
        rec["id"] = time.strftime("%Y%m%d-%H%M%S", time.localtime(t0)) + f".{int((t0 % 1) * 1000):03d}-{seq:04d}"
        body_bytes = bytes(r["body"])
        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            body = {"_unparsed": body_bytes[:4096].decode("utf-8", "replace")}
        headers = {k: v for k, v in head["headers"].items() if k not in DROP_HEADERS}
        rec.update({
            "schema": TRACE_SCHEMA, "api": detect_api(head["path"]) or "anthropic", "conn": self.conn_id, "tap_pid": os.getpid(),
            "request": {"headers": headers, "body": body, "body_bytes": len(body_bytes),
                        "body_sha256": hashlib.sha256(body_bytes).hexdigest()},
            "response": None,
            "timing": {"t_request": t0, "t_request_body_complete": r["t_body_complete"], "t_response_headers": None,
                       "t_first_event": None, "t_message_start": None, "t_first_delta": None, "t_first_text_delta": None,
                       "t_last_byte": None, "ttft_s": None, "total_s": None},
            "usage": None, "stop_reason": None, "outcome": "inflight",
            "model": body.get("model") if isinstance(body, dict) else None,
            "stream": bool(body.get("stream")) if isinstance(body, dict) else None,
        })
        return rec

    def _finalize(self, rec, cr, outcome, t, complete=True):
        if not rec["recorded"]:
            self._line(rec, cr)
            return
        head = cr["head"]
        sse_lines = cr["sse_lines"]
        body_text = None
        if cr["sse"] is not None:
            if cr["linebuf"]:
                tail = cr["linebuf"].decode("utf-8", "replace")
                sse_lines.append(tail)
                cr["sse"].feed_line(tail)
            cr["sse"].flush()
            usage = extract_raw_usage(rec["api"], events=cr["sse"].events)
        else:
            body_text = bytes(cr["raw"]).decode("utf-8", "replace") if cr["raw"] else None
            usage = extract_raw_usage(rec["api"], body_text=body_text) if head["status"] < 400 else None
        rec["response"] = {"status": head["status"], "headers": dict(head["headers"]),
                           "sse_lines": sse_lines if cr["sse"] is not None else None,
                           "body_text": body_text, "bytes": cr["bytes"], "complete": complete, "outcome": outcome}
        tm = rec["timing"]
        tm.update({"t_response_headers": cr["t_headers"], "t_last_byte": cr["t_last_byte"]})
        tm.update(cr["watch"].timing())
        t_first = tm["t_first_delta"] or tm["t_first_event"]
        tm["ttft_s"] = round(t_first - tm["t_request"], 4) if t_first else None
        tm["total_s"] = round(tm["t_last_byte"] - tm["t_request"], 4) if tm["t_last_byte"] else None
        rec["outcome"] = outcome
        if usage:
            rec["stop_reason"] = usage.get("stop_reason")
            rec["usage"] = usage
        self._write(rec)
        self._line(rec, cr)

    def _write(self, rec):
        out = {k: v for k, v in rec.items() if k != "recorded"}
        path = os.path.join(self.out_dir, rec["id"] + ".json")
        try:
            save_json_atomic(path, out)
        except Exception as e:
            log(f"{now_hms()}  tap: could not write {path}: {e!r}")

    def _line(self, rec, cr):
        if self.quiet and not rec["recorded"]:
            return
        status = cr["head"]["status"] if cr else "-"
        if not rec["recorded"]:
            ms = ((cr["t_last_byte"] if cr else time.time()) - rec["t_request"]) * 1000
            q = ("?" + rec["query"]) if rec["query"] else ""
            log(f"{now_hms(rec['t_request'])}  {rec['method']} {rec['path']}{q} {status} {ms:.0f} ms")
            return
        u = _normalizer.normalize(rec.get("usage"))
        tm = rec["timing"]
        parts = [f"#{rec['seq']:04d} {rec['method']} {rec['path']} {status}",
                 f"prompt={fmt_n(u.total)} cached={fmt_n(u.cached)} out={fmt_n(u.output_tokens)}"]
        if tm.get("ttft_s") is not None:
            parts.append(f"ttft={tm['ttft_s']:.1f}s total={tm['total_s']:.1f}s")
        if rec["outcome"] != "ok":
            parts.append(rec["outcome"].upper())
        path = display_path(os.path.join(self.out_dir, rec["id"] + ".json"))
        log(f"{now_hms(rec['t_request'])}  " + "  ".join(parts) + f"  -> {path}")
