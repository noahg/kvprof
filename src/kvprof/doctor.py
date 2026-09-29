"""`doctor`: check that the server exposes everything the analyzer needs, and that its log still parses."""
import os

from . import PACKAGE_NAME
from .backends import get_backend
from .backends.llamacpp.serverlog import TESTED_BUILDS, scan_log_format
from .convert import DEFAULT_VARIANT, rendered_count
from .sections.profiles import detect_profile
from .util import byte_starts, fmt_n

PROBE_TEXT = "kvprof probe: naïve café – ünïcødé ✓"
PROBE_MESSAGES = [{"role": "system", "content": "PROBE_SYSTEM_7f3a"}, {"role": "user", "content": "PROBE_USER_7f3a"},
                  {"role": "assistant", "content": "PROBE_ASSISTANT_7f3a"}, {"role": "user", "content": "PROBE_AGAIN_7f3a"}]
PROBE_BODY = {"system": "You are terse.", "messages": [{"role": "user", "content": "Say hi."}]}


class Report:
    def __init__(self):
        self.failed = False
        self.warned = False

    def line(self, status, msg):
        if status == "FAIL":
            self.failed = True
        elif status == "WARN":
            self.warned = True
        print(f"  {status:5s} {msg}")


def run_doctor(settings):
    """Print a checklist; return the process exit code (1 on any hard failure)."""
    r = Report()
    print(f"{PACKAGE_NAME} doctor: {settings.upstream}")
    backend = get_backend(settings.backend, upstream=settings.upstream, api_key=settings.api_key, timeout=60)
    client = backend.client
    try:
        props = client.props()
    except Exception as e:
        r.line("FAIL", f"GET /props: {e}  (is llama-server running at {settings.upstream}?)")
        return 1
    info = backend.server_info()
    build = info.build or "?"
    tested = build.split("-")[0] in TESTED_BUILDS
    r.line("OK" if tested else "WARN", f"/props: llama.cpp build {build} ({'tested' if tested else 'untested; tested: ' + ', '.join(TESTED_BUILDS)}), "
                                        f"{fmt_n(info.n_slots)} slot(s), n_ctx {fmt_n(info.n_ctx)} per slot, model {info.model_name}")
    if not props.get("chat_template"):
        r.line("WARN", "/props has no chat_template; template profile falls back to generic")
    profile = detect_profile(info.chat_template)
    r.line("OK", f"chat template: {len(info.chat_template or ''):,} chars -> section profile {profile}"
                 + (" (boundaries approximate to turn markers)" if profile == "generic" else ""))
    if info.model_id:
        r.line("OK", f"/v1/models: id {info.model_id!r}" + (f" (alias {info.model_alias!r})" if info.model_alias and info.model_alias != info.model_id else ""))
    else:
        r.line("WARN", "/v1/models returned no model id; pass --model NAME to analyze/calibrate")

    # tokenizer: pieces must concatenate to the exact bytes, or token ranges cannot be mapped to text
    try:
        toks = client.tokenize(PROBE_TEXT, pieces=True)
        joined = b"".join(t.piece for t in toks)
        ok = joined == PROBE_TEXT.encode("utf-8") and byte_starts(toks)[-1] == len(joined)
        r.line("OK" if ok else "FAIL", f"/tokenize with pieces: {len(toks)} tokens" + ("; pieces concatenate to the input" if ok else f"; pieces give {joined!r}"))
    except Exception as e:
        r.line("FAIL", f"/tokenize with pieces: {e}")

    # renderer
    rendered = None
    try:
        rendered = client.apply_template(PROBE_MESSAGES)
        missing = [m["content"] for m in PROBE_MESSAGES if m["content"] not in rendered]
        r.line("FAIL" if missing else "OK", "/apply-template: " + (f"probe texts missing from the render: {missing}" if missing
                                                                    else f"renders all probe messages ({len(rendered)} chars)"))
    except Exception as e:
        r.line("FAIL", f"/apply-template: {e}")

    # the server's own count for an Anthropic body vs our render + tokenize
    model = settings.model or backend.default_model()
    try:
        target = client.count_tokens(PROBE_BODY, model)
        n = rendered_count(backend, PROBE_BODY, DEFAULT_VARIANT)[0]
        r.line("OK" if n == target else "FAIL", f"/v1/messages/count_tokens: {target} tokens; rendered + tokenized here: {n}"
                                                + ("" if n == target else " (Anthropic traces will not be exact)"))
    except Exception as e:
        r.line("WARN", f"/v1/messages/count_tokens: {e} (Anthropic calibration unavailable; OpenAI traces are unaffected)")

    try:
        sl = client.slots()
        r.line("OK", f"/slots: {len(sl) if isinstance(sl, list) else '?'} slot(s)")
    except Exception as e:
        r.line("WARN", f"/slots: {e} (optional; start llama-server with --slots for it)")

    if settings.server_log:
        path = settings.server_log
        if not os.path.exists(path):
            r.line("FAIL", f"server log {path}: not found")
        else:
            scan = scan_log_format(path)
            c = scan["counts"]
            counts = ", ".join(f"{k} {fmt_n(v)}" for k, v in c.items())
            if not any(c.values()):
                r.line("FAIL", f"server log {path}: no slot lines recognized ({counts}); is it llama-server's log (-v not needed, --log-file or stdout)?")
            elif c["launch"] == 0 or c["release"] == 0:
                r.line("FAIL", f"server log: {counts}; task launch/release lines not recognized (format changed?)")
            else:
                r.line("OK", f"server log: {counts}")
            if scan["unmatched"]:
                r.line("WARN", f"server log: {fmt_n(scan['unmatched'])} slot line(s) matched no pattern (log format drift?), e.g.:")
                for ex in scan["examples"]:
                    print(f"          {ex}")
    else:
        r.line("INFO", "no --server-log: skipping the log-format check (server-side prefill numbers need it)")
    print("result: " + ("FAIL" if r.failed else "OK with warnings" if r.warned else "all OK"))
    return 1 if r.failed else 0
