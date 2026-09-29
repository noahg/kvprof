"""Thin HTTP client for llama-server: /props, /v1/models, /slots, /apply-template, /tokenize, count_tokens."""
import json
import urllib.error
import urllib.request

from ... import PACKAGE_NAME, __version__
from ...util import piece_bytes
from ..base import Renderer, ServerInfo, Token, Tokenizer


class LlamaCppClient(Renderer, Tokenizer):
    def __init__(self, base, api_key=None, timeout=300):
        if "://" not in base:
            base = "http://" + base
        self.base = base.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.api_key = api_key
        self.timeout = timeout

    def _req(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        headers = {"content-type": "application/json", "user-agent": f"{PACKAGE_NAME}/{__version__}"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
            headers["x-api-key"] = self.api_key
        if path.startswith("/v1/messages"):
            headers["anthropic-version"] = "2023-06-01"
        req = urllib.request.Request(self.base + path, data=data, method="POST" if body is not None else "GET", headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    # -- server facts
    def props(self):
        return self._req("/props")

    def models(self):
        return self._req("/v1/models")

    def slots(self):
        return self._req("/slots")

    def model_path(self):
        try:
            return self.props().get("model_path")
        except Exception:
            return None

    def server_info(self):
        info = ServerInfo(backend="llamacpp")
        try:
            p = self.props()
        except Exception as e:
            info.extra["props_error"] = repr(e)
            p = {}
        info.build = p.get("build_info")
        info.model_alias = p.get("model_alias")
        info.model_path = p.get("model_path")
        info.n_slots = p.get("total_slots")
        info.chat_template = p.get("chat_template")
        info.n_ctx = (p.get("default_generation_settings") or {}).get("n_ctx")
        try:
            data = self.models().get("data") or []
            if data:
                info.model_id = data[0].get("id")
                if info.n_ctx is None:
                    info.n_ctx = (data[0].get("meta") or {}).get("n_ctx")
        except Exception:
            pass
        if info.n_ctx is None or info.n_slots is None:
            try:
                sl = self.slots()
                if isinstance(sl, list) and sl:
                    info.n_ctx = info.n_ctx or sl[0].get("n_ctx")
                    info.n_slots = info.n_slots or len(sl)
            except Exception:
                pass
        return info

    # -- Renderer / Tokenizer
    def apply_template(self, messages, tools=None):
        body = {"messages": messages}
        if tools:
            body["tools"] = tools
        return self._req("/apply-template", body)["prompt"]

    def render(self, messages, tools=None):
        return self.apply_template(messages, tools)

    def tokenize(self, text, pieces=True):
        """-> list of Token(id, piece_bytes). add_special=False: the rendered prompt already has its own markers."""
        r = self._req("/tokenize", {"content": text, "with_pieces": pieces, "add_special": False})
        if not pieces:
            return [Token(t if isinstance(t, int) else t["id"], None) for t in r["tokens"]]
        return [Token(t["id"], piece_bytes(t["piece"])) for t in r["tokens"]]

    def count_tokens(self, anthropic_body, model):
        """The server's own prompt count for an Anthropic body (render + tokenize, server side)."""
        body = {k: anthropic_body[k] for k in ("model", "system", "tools", "messages") if k in anthropic_body}
        body["model"] = model or body.get("model") or "default"
        return self._req("/v1/messages/count_tokens", body)["input_tokens"]
