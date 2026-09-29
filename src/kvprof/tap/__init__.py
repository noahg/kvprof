"""The tap: a passive recorder between a client (Claude Code, OpenCode, ...) and llama-server.

A raw TCP relay: every byte the client sends goes to the upstream unchanged and every byte the upstream sends
comes back unchanged (same framing, same headers, same chunking). A passive HTTP/1.1 parser watches the
bytes and, for POSTs to the recorded paths (/v1/messages, /v1/chat/completions), writes one JSON file per
request: the parsed request body (credentials removed), the raw SSE lines or JSON body of the response,
timestamps, and the usage block. Other paths pass through with one log line.
"""
import os

from ..util import display_path, log, parse_addr
from .recorder import TapContext
from .relay import serve_relay


def run_tap(settings, ready_event=None, stop_event=None, info=None):
    listen = parse_addr(settings.listen, 8081)
    upstream = parse_addr(settings.upstream, 8080)
    os.makedirs(settings.trace_dir, exist_ok=True)
    ctx = TapContext(settings.trace_dir, settings.record_paths, settings.quiet)
    info = {} if info is None else info

    def on_listen(addr):
        info["host"], info["port"] = addr[0], addr[1]
        out = display_path(settings.trace_dir)
        base = f"http://{addr[0]}:{addr[1]}"
        log(f"tap: listening on {addr[0]}:{addr[1]} -> {upstream[0]}:{upstream[1]}; recording {', '.join(sorted(ctx.record_paths))} to {out}")
        log(f"tap: Claude Code:    ANTHROPIC_BASE_URL={base} claude")
        log(f"tap: OpenAI clients: OPENAI_BASE_URL={base}/v1   (OpenCode: provider.<name>.options.baseURL = \"{base}/v1\")")
        log("tap: traces contain full prompts (your code, pasted secrets); keep the trace directory out of version control")

    try:
        serve_relay(listen, upstream, ctx, ready_event=ready_event, stop_event=stop_event, on_listen=on_listen)
    except KeyboardInterrupt:
        log("tap: stopped")
