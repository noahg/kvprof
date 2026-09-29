"""Command line: tap | analyze | serve | calibrate | doctor."""
import argparse
import sys

from . import PACKAGE_NAME, __version__
from .config import (DEFAULT_GUI_HOST, DEFAULT_GUI_PORT, DEFAULT_LISTEN, DEFAULT_RECORD_PATHS, DEFAULT_TRACE_DIR,
                     DEFAULT_UPSTREAM, env_name, resolve)

DESCRIPTION = f"""\
Prompt-cache profiler for llama.cpp's llama-server: why did it re-prefill my prompt?

  %(prog)s tap        record requests between your client and llama-server
  %(prog)s analyze    one line per request: what changed, what it cost, why
  %(prog)s serve      the same, live, in a browser
  %(prog)s doctor     check the server exposes everything the analyzer needs

Typical session:
  %(prog)s tap --upstream {DEFAULT_UPSTREAM}
  ANTHROPIC_BASE_URL=http://{DEFAULT_LISTEN} claude                     # or any OpenAI client at http://{DEFAULT_LISTEN}/v1
  %(prog)s analyze --upstream {DEFAULT_UPSTREAM} --server-log server.log
"""


def _common(p, upstream=True, trace_dir=True):
    if upstream:
        p.add_argument("--upstream", metavar="URL", help=f"llama-server base URL (default {DEFAULT_UPSTREAM}; env {env_name('upstream')})")
        p.add_argument("--api-key", metavar="KEY", help=f"sent as Bearer / x-api-key if the server needs one (env {env_name('api_key')})")
    if trace_dir:
        p.add_argument("--trace-dir", metavar="DIR", help=f"where traces live (default ./{DEFAULT_TRACE_DIR}; env {env_name('trace_dir')})")


def _analysis_opts(p):
    _common(p)
    p.add_argument("--server-log", metavar="PATH", help=f"llama-server's log file, for server-side prefill/decode numbers and slot choice (env {env_name('server_log')})")
    p.add_argument("--server-pid", metavar="PATH", help="pid file of llama-server; its start time anchors the log's timestamps (else the log's mtime)")
    p.add_argument("--log-anchor", type=float, metavar="EPOCH", help="epoch seconds of the log's time 0 (overrides --server-pid)")
    p.add_argument("--no-log", dest="use_log", action="store_false", default=None, help="ignore the server log")
    p.add_argument("--model", metavar="NAME", help="model name for the analyzer's own count_tokens requests (default: from /v1/models)")
    p.add_argument("--template-profile", choices=["auto", "chatml_qwen", "generic"], help="section builder (default: auto-detected from the chat template)")
    p.add_argument("--refresh", action="store_true", default=None, help="ignore cached analyses")


def build_parser():
    ap = argparse.ArgumentParser(prog=PACKAGE_NAME, description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", metavar="COMMAND")

    p = sub.add_parser("tap", help="record requests passing between a client and llama-server",
                       description="A passive byte relay: bytes pass through unchanged; POSTs to the recorded paths are written "
                                   "to the trace directory as one JSON file per request.")
    _common(p)
    p.add_argument("--listen", metavar="HOST:PORT", help=f"address the client connects to (default {DEFAULT_LISTEN})")
    p.add_argument("--record-paths", metavar="PATHS", help=f"comma-separated request paths to record (default {','.join(DEFAULT_RECORD_PATHS)})")
    p.add_argument("--quiet", action="store_true", default=None, help="do not log pass-through requests")

    p = sub.add_parser("analyze", help="explain each recorded request's prefill cost")
    _analysis_opts(p)
    p.add_argument("--last", type=int, metavar="N", help="only the last N requests")
    p.add_argument("--dump", metavar="N|ID", help="one request: sections and the rendered prompt with section markers")
    p.add_argument("--json", action="store_true", help="summaries as JSON (with --dump: the full analysis)")

    p = sub.add_parser("serve", help="live web GUI over the trace directory")
    _analysis_opts(p)
    p.add_argument("--host", help=f"GUI address (default {DEFAULT_GUI_HOST})")
    p.add_argument("--port", type=int, help=f"GUI port (default {DEFAULT_GUI_PORT})")

    p = sub.add_parser("calibrate", help="(re)derive the Anthropic->OpenAI conversion details from the server")
    _common(p)
    p.add_argument("--model", metavar="NAME", help="model name for count_tokens (default: from /v1/models)")

    p = sub.add_parser("doctor", help="check the server exposes what the analyzer needs")
    _common(p, trace_dir=False)
    p.add_argument("--server-log", metavar="PATH", help="also check the log format against the parser")
    p.add_argument("--model", metavar="NAME", help="model name for count_tokens (default: from /v1/models)")
    return ap


def _pipeline(s, verbose):
    from .analyze import Pipeline, UpstreamUnavailable
    try:
        return Pipeline(s, verbose=verbose)
    except UpstreamUnavailable as e:
        sys.exit(f"{PACKAGE_NAME}: {e}\n  (is llama-server running? pass --upstream URL)")


def cmd_analyze(s, a):
    from .summarize import dump_request, print_session
    p = _pipeline(s, verbose=not a.json)
    p.refresh_all()
    if a.dump:
        try:
            dump_request(p, a.dump, a.json)
        except KeyError as e:
            sys.exit(str(e.args[0]))
    else:
        print_session(p, a.last, a.json)


def cmd_serve(s, a):
    from .serve import serve
    p = _pipeline(s, verbose=False)
    serve(p, s.host, s.port)


def cmd_calibrate(s, a):
    from .backends import get_backend
    from .convert import calibrate
    from .store import TraceStore
    backend = get_backend(s.backend, upstream=s.upstream, api_key=s.api_key)
    store = TraceStore(s.trace_dir)
    extra = []
    for tid in store.list_ids()[:1]:
        try:
            tr = store.load_trace(tid)
            if tr.get("api", "anthropic") == "anthropic":
                extra.append((tid[-4:], tr["request"]["body"]))
        except Exception:
            pass
    try:
        calibrate(backend, store, s.model or backend.default_model(), extra)
    except OSError as e:
        sys.exit(f"{PACKAGE_NAME}: cannot reach {s.upstream}: {e}")


def cmd_tap(s, a):
    from .tap import run_tap
    run_tap(s)


def cmd_doctor(s, a):
    from .doctor import run_doctor
    sys.exit(run_doctor(s))


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return
    s = resolve(a)
    {"tap": cmd_tap, "analyze": cmd_analyze, "serve": cmd_serve, "calibrate": cmd_calibrate, "doctor": cmd_doctor}[a.cmd](s, a)
