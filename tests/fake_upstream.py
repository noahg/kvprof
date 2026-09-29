"""A fake llama-server for tests: the endpoints kvprof uses, a toy tokenizer, two chat templates, and a
server log in llama.cpp's b10909 format.

Caching behaves like a plain-attention llama-server with two slots: a request reuses the longest common
token prefix with the best slot's previous prompt. The header `x-fake-cached: N` overrides the reused count.
"""
import json
import re
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kvprof.convert import DEFAULT_VARIANT, anthropic_to_openai

# What the fake server's own Anthropic->OpenAI converter does (mirrors llama.cpp b10909: tool_result parts joined with "").
SERVER_VARIANT = dict(DEFAULT_VARIANT, tool_result_join="")
REPLY = "Hello from the fake server."

CHATML_TEMPLATE = ("{%- for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor %}"
                   "<tools></tools><tool_call></tool_call><tool_response></tool_response>")
LLAMA3_TEMPLATE = "{{- bos_token }}<|start_header_id|>{{ role }}<|end_header_id|>\n\n{{ content | trim }}<|eot_id|>"

TOKEN_RE = re.compile(r"\w+|\s+|[^\w\s]")


# ----------------------------------------------------------------------------- tokenizer

def tokenize(text):
    """-> [(id, piece)] where piece is a str, or a list of byte values for a partial UTF-8 sequence."""
    out = []
    for p in TOKEN_RE.findall(text):
        b = p.encode("utf-8")
        if len(b) == len(p):
            out.append((zlib.crc32(b) % 50000 + 1, p))
        else:  # split non-ASCII pieces mid-character, like byte-level BPE does
            for part in (b[:1], b[1:]):
                out.append((zlib.crc32(part) % 50000 + 1, list(part)))
    return out


# ----------------------------------------------------------------------------- renderers

def _content_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _xml_params(schema):
    out = []
    for name, p in (schema.get("properties") or {}).items():
        out.append(f"<parameter>\n<name>{name}</name>\n<type>{p.get('type', 'string')}</type>\n"
                   + (f"<description>{p['description']}</description>\n" if p.get("description") else "") + "</parameter>")
    req = f"\n<required>{json.dumps(schema['required'])}</required>" if schema.get("required") else ""
    return "<parameters>\n" + "\n".join(out) + req + "\n</parameters>" if out or req else "<parameters>\n</parameters>"


def render_chatml(messages, tools):
    """Qwen3-Coder-style ChatML (the shape of the real template's output)."""
    R = []
    msgs = list(messages)
    sys_text = None
    if msgs and msgs[0].get("role") == "system":
        sys_text = _content_text(msgs.pop(0).get("content"))
    if sys_text is not None or tools:
        R.append("<|im_start|>system\n" + (sys_text or ""))
        if tools:
            R.append("\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>")
            for t in tools:
                fn = t.get("function") or {}
                R.append(f"\n<function>\n<name>{fn.get('name')}</name>\n<description>{fn.get('description', '')}</description>\n"
                         f"{_xml_params(fn.get('parameters') or {})}\n</function>")
            R.append("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                     "<tool_call>\n<function=example_function_name>\n</function>\n</tool_call>")
        R.append("<|im_end|>\n")
    i = 0
    while i < len(msgs):
        m = msgs[i]
        role = m.get("role")
        if role == "tool":
            R.append("<|im_start|>user")
            while i < len(msgs) and msgs[i].get("role") == "tool":
                R.append("\n<tool_response>\n" + _content_text(msgs[i].get("content")) + "\n</tool_response>")
                i += 1
            R.append("<|im_end|>\n")
            continue
        if role == "assistant":
            R.append("<|im_start|>assistant\n" + _content_text(m.get("content")))
            for c in m.get("tool_calls") or []:
                fn = c.get("function") or {}
                args = fn.get("arguments")
                args = json.loads(args) if isinstance(args, str) and args else (args or {})
                ps = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in args.items())
                R.append(f"\n\n<tool_call>\n<function={fn.get('name')}>\n{ps}</function>\n</tool_call>")
            R.append("<|im_end|>\n")
        else:
            R.append(f"<|im_start|>{role}\n" + _content_text(m.get("content")) + "<|im_end|>\n")
        i += 1
    R.append("<|im_start|>assistant\n")
    return "".join(R)


def render_llama3(messages, tools):
    """Llama-3.1-style: header markers, trimmed contents, tools as JSON, tool calls re-serialized as JSON."""
    H = lambda role: f"<|start_header_id|>{role}<|end_header_id|>\n\n"  # noqa: E731
    R = ["<|begin_of_text|>"]
    msgs = list(messages)
    sys_text = ""
    if msgs and msgs[0].get("role") == "system":
        sys_text = _content_text(msgs.pop(0).get("content")).strip()
    R.append(H("system") + "Cutting Knowledge Date: December 2023\n\n" + sys_text)
    if tools:
        R.append("\n\nYou have access to the following functions. To call a function, respond with JSON.\n\n")
        for t in tools:
            R.append(json.dumps(t, indent=4) + "\n\n")
    R.append("<|eot_id|>")
    for m in msgs:
        role = m.get("role")
        if role == "tool":
            R.append(H("ipython") + _content_text(m.get("content")).strip() + "<|eot_id|>")
        elif role == "assistant" and m.get("tool_calls"):
            R.append(H("assistant") + _content_text(m.get("content")).strip())
            for c in m["tool_calls"]:
                fn = c.get("function") or {}
                args = fn.get("arguments")
                args = json.loads(args) if isinstance(args, str) and args else (args or {})
                R.append(json.dumps({"name": fn.get("name"), "parameters": args}))
            R.append("<|eot_id|>")
        else:
            R.append(H(role) + _content_text(m.get("content")).strip() + "<|eot_id|>")
    R.append(H("assistant"))
    return "".join(R)


RENDERERS = {"chatml": (render_chatml, CHATML_TEMPLATE), "llama3": (render_llama3, LLAMA3_TEMPLATE)}


# ----------------------------------------------------------------------------- server

class FakeUpstream:
    def __init__(self, log_path, template="chatml", n_slots=2, n_ctx=8192):
        self.render, self.chat_template = RENDERERS[template]
        self.template = template
        self.log_path = log_path
        self.n_slots, self.n_ctx = n_slots, n_ctx
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.slots = [{"ids": [], "t_last": -1} for _ in range(n_slots)]
        self.task_id = 0
        self.requests = []   # (path, headers, body) of every request received
        open(log_path, "w").close()
        self._log(f"I srv    load_model: initializing, n_slots = {n_slots}, n_ctx_slot = {n_ctx}, kv_unified = 'false'")
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                fake.handle(self, "GET", None)

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                fake.handle(self, "POST", json.loads(self.rfile.read(n) or b"{}"))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    # -- log
    def _ts(self):
        e = time.time() - self.t0
        us = int(round(e * 1e6))
        m, us = divmod(us, 60_000_000)
        s, us = divmod(us, 1_000_000)
        ms, us = divmod(us, 1000)
        return f"{m}.{s:02d}.{ms:03d}.{us:03d}"

    def _log(self, rest):
        with open(self.log_path, "a") as f:
            f.write(f"{self._ts()} {rest}\n")

    # -- helpers
    def props(self):
        return {"total_slots": self.n_slots, "model_alias": "test-model", "model_path": "/models/test-model.gguf",
                "build_info": "b10909-test", "chat_template": self.chat_template,
                "default_generation_settings": {"n_ctx": self.n_ctx}}

    def prompt_ids(self, messages, tools):
        return [t for t, _ in tokenize(self.render(messages, tools))]

    def run_task(self, ids, n_out, cached_override, on_prefill_done=None):
        """Slot selection, caching and log lines for one completion. -> (cached, slot)"""
        with self.lock:
            best, best_l = None, 0
            for k, s in enumerate(self.slots):
                n = 0
                while n < min(len(ids), len(s["ids"])) and ids[n] == s["ids"][n]:
                    n += 1
                if n > best_l:
                    best, best_l = k, n
            if best is not None and best_l > 0:
                slot = best
                prev = self.slots[slot]["ids"]
                self._log(f"I slot get_availabl: id  {slot} | task -1 | selected slot by LCP similarity, "
                          f"f_sim_best = {best_l / max(1, len(ids)):.3f} (> 0.100 thold), f_keep = {best_l / max(1, len(prev)):.3f}")
            else:
                slot = min(range(self.n_slots), key=lambda k: self.slots[k]["t_last"])
                self._log(f"I slot get_availabl: id  {slot} | task -1 | selected slot by LRU, t_last = {self.slots[slot]['t_last']}")
                best_l = 0
            cached = min(best_l, len(ids) - 1) if cached_override is None else cached_override
            tid = self.task_id
            self.task_id += 5
            self._log(f"I slot launch_slot_: id  {slot} | task {tid} | processing task, is_child = 0")
            pp = len(ids) - cached
            self._log(f"I slot print_timing: id  {slot} | task {tid} | prompt eval time =  {pp * 1.0:9.2f} ms / {pp:5d} tokens "
                      f"(    1.00 ms per token,  1000.00 tokens per second)")
            self._log(f"I slot print_timing: id  {slot} | task {tid} |        eval time =  {n_out * 20.0:9.2f} ms / {n_out:5d} tokens "
                      f"(   20.00 ms per token,    50.00 tokens per second)")
            self._log(f"I slot print_timing: id  {slot} | task {tid} |       total time =  {pp + n_out * 20.0:9.2f} ms / {pp + n_out:5d} tokens")
            self._log(f"I slot      release: id  {slot} | task {tid} | stop processing: n_tokens = {len(ids) + n_out - 1}, truncated = 0")
            self.slots[slot] = {"ids": list(ids), "t_last": int(time.time() * 1000)}
            return cached, slot

    # -- dispatch
    def handle(self, h, method, body):
        path = h.path.split("?")[0]
        headers = {k.lower(): v for k, v in h.headers.items()}
        self.requests.append((path, headers, body))
        try:
            if method == "GET" and path == "/props":
                return self.send_json(h, self.props())
            if method == "GET" and path == "/v1/models":
                return self.send_json(h, {"object": "list", "data": [{"id": "test-model", "object": "model", "meta": {"n_ctx": self.n_ctx}}]})
            if method == "GET" and path == "/slots":
                return self.send_json(h, [{"id": k, "n_ctx": self.n_ctx} for k in range(self.n_slots)])
            if method == "GET" and path == "/health":
                return self.send_json(h, {"status": "ok"})
            if path == "/tokenize":
                toks = tokenize(body.get("content", ""))
                if body.get("with_pieces"):
                    return self.send_json(h, {"tokens": [{"id": i, "piece": p} for i, p in toks]})
                return self.send_json(h, {"tokens": [i for i, _ in toks]})
            if path == "/apply-template":
                return self.send_json(h, {"prompt": self.render(body.get("messages") or [], body.get("tools"))})
            if path == "/v1/messages/count_tokens":
                msgs, tools, _, _ = anthropic_to_openai(body, SERVER_VARIANT)
                return self.send_json(h, {"input_tokens": len(self.prompt_ids(msgs, tools))})
            if path == "/v1/messages":
                return self.messages(h, headers, body)
            if path == "/v1/chat/completions":
                return self.chat(h, headers, body)
            self.send_json(h, {"error": {"message": "not found"}}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, h, obj, status=200):
        data = json.dumps(obj).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json; charset=utf-8")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    def start_sse(self, h):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()

    def chunk(self, h, text):
        data = text.encode()
        h.wfile.write(f"{len(data):x};ext=1\r\n".encode() + data + b"\r\n")
        h.wfile.flush()

    def end_chunks(self, h):
        h.wfile.write(b"0\r\n\r\n")
        h.wfile.flush()

    def _reply_tokens(self):
        return [p for _, p in tokenize(REPLY)]

    def messages(self, h, headers, body):
        msgs, tools, _, _ = anthropic_to_openai(body, SERVER_VARIANT)
        ids = self.prompt_ids(msgs, tools)
        if len(ids) > self.n_ctx:
            return self.send_json(h, {"error": {"type": "invalid_request_error", "message": "exceeds the available context size"}}, 400)
        out = self._reply_tokens()
        override = int(headers["x-fake-cached"]) if headers.get("x-fake-cached") else None
        cached, _ = self.run_task(ids, len(out), override)
        usage = {"input_tokens": len(ids) - cached, "cache_read_input_tokens": cached}
        if not body.get("stream"):
            return self.send_json(h, {"id": "msg_fake", "type": "message", "role": "assistant", "model": "test-model",
                                      "content": [{"type": "text", "text": REPLY}], "stop_reason": "end_turn",
                                      "usage": dict(usage, output_tokens=len(out))})
        self.start_sse(h)
        ev = lambda name, d: self.chunk(h, f"event: {name}\ndata: {json.dumps(d)}\n\n")  # noqa: E731
        ev("message_start", {"type": "message_start", "message": {"id": "msg_fake", "type": "message", "role": "assistant",
                                                                  "model": "test-model", "content": [], "usage": dict(usage, output_tokens=0)}})
        ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        delay = float(headers.get("x-fake-delay") or 0)
        for p in out:
            ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": p}})
            time.sleep(delay)
        ev("content_block_stop", {"type": "content_block_stop", "index": 0})
        ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": len(out)}})
        ev("message_stop", {"type": "message_stop"})
        self.end_chunks(h)

    def chat(self, h, headers, body):
        msgs, tools = body.get("messages") or [], body.get("tools")
        ids = self.prompt_ids(msgs, tools)
        out = self._reply_tokens()
        override = int(headers["x-fake-cached"]) if headers.get("x-fake-cached") else None
        cached, _ = self.run_task(ids, len(out), override)
        usage = {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out),
                 "prompt_tokens_details": {"cached_tokens": cached}}
        timings = {"cache_n": cached, "prompt_n": len(ids) - cached, "prompt_ms": float(len(ids) - cached),
                   "predicted_n": len(out), "predicted_ms": 20.0 * len(out), "predicted_per_second": 50.0}
        base = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()), "model": "test-model"}
        if not body.get("stream"):
            return self.send_json(h, {"id": "chatcmpl-fake", "object": "chat.completion", "model": "test-model",
                                      "choices": [{"index": 0, "message": {"role": "assistant", "content": REPLY}, "finish_reason": "stop"}],
                                      "usage": usage, "timings": timings})
        self.start_sse(h)
        send = lambda d: self.chunk(h, "data: " + json.dumps(d) + "\n\n")  # noqa: E731
        send(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]))
        for p in out:
            send(dict(base, choices=[{"index": 0, "delta": {"content": p}, "finish_reason": None}]))
        final = dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}], timings=timings)
        if (body.get("stream_options") or {}).get("include_usage"):
            final["usage"] = usage
        send(final)
        self.chunk(h, "data: [DONE]\n\n")
        self.end_chunks(h)
