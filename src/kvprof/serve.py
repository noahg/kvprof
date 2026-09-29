"""Live web GUI: a small JSON API over the Pipeline plus one static page (web/index.html)."""
import hashlib
import json
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from . import PACKAGE_NAME, __version__
from .diff import diff_analysis
from .sections import find_node
from .summarize import summarize
from .util import display_path

_SLIM_DROP = ("ids", "bstart", "rendered")


def index_html():
    return resources.files(__package__).joinpath("web/index.html").read_bytes()


def meta(p):
    """Everything the page needs besides the request rows (server facts, where the traces are)."""
    info = p.info
    api_counts = {}
    for a in p.analyses:
        api_counts[a.get("api") or "anthropic"] = api_counts.get(a.get("api") or "anthropic", 0) + 1
    trace_dir = display_path(p.store.trace_dir)
    return {"package": PACKAGE_NAME, "version": __version__, "backend": p.backend.name, "build": info.build,
            "model": info.model_name, "model_id": p.model, "n_slots": info.n_slots, "n_ctx": info.n_ctx,
            "lanes": p.backend.topology.lanes(info), "profile": p.profile, "trace_dir": trace_dir,
            "has_log": p.telemetry is not None, "anchor": p.anchor, "anchor_src": p.anchor_src, "variant": p.variant,
            "fingerprint": p.last_fingerprint, "api_counts": api_counts}


class Handler(BaseHTTPRequestHandler):
    pipeline = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", etag=None):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        if etag:
            self.send_header("ETag", etag)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, obj, etag=None):
        self._send(200, json.dumps(obj, ensure_ascii=False), etag=etag)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        parts = [x for x in u.path.split("/") if x]
        p = self.pipeline
        try:
            if not parts or parts == ["index.html"]:
                return self._send(200, index_html(), "text/html; charset=utf-8")
            if parts[0] != "api" or len(parts) < 2:
                return self._send(404, '{"error":"not found"}')
            if parts[1:] == ["requests"]:
                with p.lock:
                    payload = meta(p)
                    payload["requests"] = [summarize(a) for a in p.analyses]
                etag = '"' + hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest() + '"'
                if self.headers.get("If-None-Match") == etag:
                    self.send_response(304); self.send_header("ETag", etag); self.send_header("Content-Length", "0"); self.end_headers()
                    return
                return self._json(payload, etag)
            if len(parts) >= 3 and parts[1] == "requests":
                with p.lock:
                    a = p.by_id.get(parts[2])
                if a is None:
                    return self._send(404, '{"error":"unknown request"}')
                if len(parts) == 3:
                    slim = {k: v for k, v in a.items() if k not in _SLIM_DROP}
                    slim["summary_row"] = summarize(a)
                    return self._json(slim)
                if parts[3] == "tokens":
                    if "section" in q:
                        n = find_node(a["sections"], q["section"][0])
                        if n is None:
                            return self._send(404, '{"error":"unknown section"}')
                        start, end = n["tok"]
                    else:
                        start, end = int(q.get("start", ["0"])[0]), int(q.get("end", [str(a["n_tokens"])])[0])
                    start = max(0, min(start, a["n_tokens"]))
                    end = max(start, min(end, a["n_tokens"], start + 4096))
                    rb = a["rendered"].encode("utf-8")
                    bs = a["bstart"]
                    toks = [[a["ids"][i], rb[bs[i]:bs[i + 1]].decode("utf-8", "replace")] for i in range(start, end)]
                    return self._json({"start": start, "end": end, "total": a["n_tokens"], "tokens": toks})
                if parts[3] == "diff":
                    other = q.get("other", [None])[0]
                    with p.lock:
                        b = p.by_id.get(other)
                    if b is None:
                        return self._send(404, '{"error":"unknown other"}')
                    return self._json(diff_analysis(a, b, p.request(a["id"]), p.request(b["id"])))
                if parts[3] == "body":
                    r = p.request(a["id"])
                    return self._json(r.body if r else None)
                if parts[3] == "sse":
                    try:
                        tr = p.store.load_trace(a["id"])
                        resp = tr.get("response") or {}
                        return self._json(resp.get("sse_lines") or ([resp["body_text"]] if resp.get("body_text") else []))
                    except Exception:
                        return self._json([])
            return self._send(404, '{"error":"not found"}')
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._send(500, json.dumps({"error": repr(e)}))
            except Exception:
                pass


def make_handler(pipeline):
    return type("BoundHandler", (Handler,), {"pipeline": pipeline})


def scan_loop(p, stop_event=None, interval=1.0):
    while not (stop_event and stop_event.is_set()):
        try:
            fp = p.fingerprint()
            if fp != p.last_fingerprint:
                p.refresh_all()
                p.last_fingerprint = fp
        except Exception as e:
            print(f"scan error: {e!r}", file=sys.stderr)
        time.sleep(interval)


def serve(p, host, port):
    threading.Thread(target=scan_loop, args=(p,), daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), make_handler(p))
    httpd.daemon_threads = True
    trace_dir = display_path(p.store.trace_dir)
    print(f"{PACKAGE_NAME} GUI: http://{host}:{httpd.server_address[1]}/   (traces in {trace_dir}; Ctrl-C to stop)", flush=True)
    if p.telemetry is None:
        print("  no --server-log given: server-side prefill/decode numbers and slot choice will be missing", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
