"""One recorded request body in either wire shape (Anthropic Messages or OpenAI chat completions).

`Request` converts the body to the OpenAI messages the server's chat template renders, and describes every
piece of content as a `RenderUnit` (system text, each tool definition, each message block / part / tool call)
with a stable id. Section builders locate units in the rendered prompt; diffs and raw token counts look
units up by id.
"""
import json
from dataclasses import dataclass

from .convert import DEFAULT_VARIANT, _json_args, anthropic_to_openai, tool_result_text

APIS = {"/v1/messages": "anthropic", "/v1/chat/completions": "openai"}


def detect_api(path):
    """Request path -> "anthropic" | "openai" | None."""
    return APIS.get((path or "").split("?", 1)[0].rstrip("/"))


@dataclass(slots=True)
class RenderUnit:
    id: str                  # system.text, tool:<name>, msg[i].text, msg[i].block[j], msg[i].part[j], msg[i].tool_call[k], ...
    kind: str                # system | tool | text | tool_use | tool_result | reasoning | image | document | ...
    text: str                # the text as it goes into the template (what a section builder searches for)
    label: str
    raw_text: str | None = None  # the client-side source text, for raw token counts (None: nothing to count)
    whole: bool = False      # the message content was a plain string: the unit covers the whole turn
    is_error: bool = False
    name: str | None = None  # tool name (tool, tool_use units)

    @property
    def preview(self):
        from .sections import _preview
        return _preview(self.text)


def _parts_text(content, join="\n"):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return join.join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return "" if content is None else str(content)


class Request:
    def __init__(self, body, api=None, variant=None):
        self.body = body if isinstance(body, dict) else {}
        self.api = api or "anthropic"   # schema-1 traces carry no api field; they were all Anthropic
        self.variant = dict(DEFAULT_VARIANT, **(variant or {}))
        self._conv = None
        self._units = None
        self._by_id = None

    # -- plain facts
    @property
    def model(self):
        return self.body.get("model")

    @property
    def stream(self):
        return bool(self.body.get("stream"))

    @property
    def messages(self):
        return self.body.get("messages") or []

    @property
    def n_messages(self):
        return len(self.messages)

    @property
    def n_tools(self):
        return len(self.body.get("tools") or [])

    @property
    def has_tools(self):
        return self.n_tools > 0

    @property
    def max_tokens(self):
        return self.body.get("max_tokens", self.body.get("max_completion_tokens"))

    @property
    def system_text(self):
        """System prompt as one string (blocks joined by spaces), '' when absent."""
        if self.api == "anthropic":
            s = self.body.get("system")
            return s if isinstance(s, str) else " ".join(b.get("text", "") for b in (s or []) if isinstance(b, dict) and b.get("type") == "text")
        msgs = self.messages
        if msgs and msgs[0].get("role") in ("system", "developer"):
            return _parts_text(msgs[0].get("content"), " ")
        return ""

    def hint_text(self):
        """System text, else the first message's text: what a request without tools is about."""
        st = self.system_text
        if not st and self.messages:
            st = _parts_text(self.messages[0].get("content"), " ")
        return st or ""

    @property
    def last_message_kinds(self):
        """Content kinds of the last message: {"tool_result"} for a tool round-trip in either shape."""
        msgs = self.messages
        if not msgs:
            return set()
        last = msgs[-1]
        if self.api == "openai":
            return {"tool_result"} if last.get("role") == "tool" else {"text"}
        lc = last.get("content")
        return {b.get("type") for b in lc if isinstance(b, dict)} if isinstance(lc, list) else {"text"}

    @property
    def metadata_session(self):
        md = self.body.get("metadata")
        uid = md.get("user_id") if isinstance(md, dict) else None
        if isinstance(uid, str) and uid:
            try:
                j = json.loads(uid)
                return (j.get("session_id") if isinstance(j, dict) else None) or None
            except ValueError:
                return uid[:64]
        u = self.body.get("user")
        return u[:64] if isinstance(u, str) and u else None

    def message_role(self, i):
        if i is None or not (0 <= i < len(self.messages)):
            return None
        return self.messages[i].get("role")

    # -- conversion
    def to_openai(self):
        """-> (messages, tools, origins, flags) in OpenAI shape, as the server's template sees them."""
        if self._conv is None:
            if self.api == "anthropic":
                self._conv = anthropic_to_openai(self.body, self.variant)
            else:
                msgs = list(self.messages)
                origins = []
                for i, m in enumerate(msgs):
                    origins.append({"msg": None, "kind": "system"} if i == 0 and m.get("role") in ("system", "developer")
                                   else {"msg": i, "blocks": None})
                self._conv = (msgs, list(self.body.get("tools") or []), origins, {"unsupported_blocks": 0, "thinking_blocks": 0})
        return self._conv

    def tool_units(self):
        out = []
        for t in self.body.get("tools") or []:
            if self.api == "anthropic":
                name, desc, schema = t.get("name") or "", t.get("description", ""), t.get("input_schema", {})
            else:
                fn = t.get("function") or {}
                name, desc, schema = fn.get("name") or "", fn.get("description", ""), fn.get("parameters", {})
            text = f"{name}\n{desc}\n{json.dumps(schema, ensure_ascii=False)}"
            out.append(RenderUnit("tool:" + name, "tool", text, name, raw_text=text, name=name))
        return out

    def render_units(self):
        """-> list, per converted message, of its RenderUnits in render order."""
        if self._units is None:
            messages, _, origins, _ = self.to_openai()
            self._units = [self._units_anthropic(m, o) if self.api == "anthropic" else self._units_openai(m, o)
                           for m, o in zip(messages, origins)]
        return self._units

    def _units_anthropic(self, m, org):
        if org["msg"] is None:
            return [RenderUnit("system.text", "system", m["content"], "system text", raw_text=m["content"], whole=True)]
        i = org["msg"]
        mid = f"msg[{i}]"
        content = self.messages[i].get("content")
        if org["blocks"] is None:
            return [RenderUnit(f"{mid}.text", "text", content, "text", raw_text=content, whole=True)]
        blocks = content
        role = m["role"]
        if role == "tool":
            j = org["blocks"][0]
            b = blocks[j]
            return [RenderUnit(f"{mid}.block[{j}]", "tool_result", m.get("content", ""), f"tool_result {b.get('tool_use_id', '')}",
                               raw_text=tool_result_text(dict(b, is_error=False), self.variant), is_error=bool(b.get("is_error")))]
        if role == "assistant":
            out = [RenderUnit(f"{mid}.block[{j}]", "reasoning", b.get("thinking", ""), "thinking", raw_text=b.get("thinking", ""))
                   for j, b in enumerate(blocks) if b.get("type") == "thinking"]
            for j in org.get("text_blocks", []):
                t = blocks[j].get("text", "")
                out.append(RenderUnit(f"{mid}.block[{j}]", "text", t, "text", raw_text=t))
            for j in org.get("tool_blocks", []):
                b = blocks[j]
                args = _json_args(b.get("input", {}))
                out.append(RenderUnit(f"{mid}.block[{j}]", "tool_use", args, f"tool_use {b.get('name', '')}", raw_text=args, name=b.get("name")))
            return out
        out = []
        for j in org["blocks"]:
            b = blocks[j]
            t = b.get("type", "text")
            if t == "text":
                out.append(RenderUnit(f"{mid}.block[{j}]", "text", b.get("text", ""), "text", raw_text=b.get("text", "")))
            else:
                out.append(RenderUnit(f"{mid}.block[{j}]", t, "[" + str(t) + "]", t))
        return out

    def _units_openai(self, m, org):
        if org["msg"] is None:
            t = _parts_text(m.get("content"))
            return [RenderUnit("system.text", "system", t, "system text", raw_text=t, whole=True)]
        i = org["msg"]
        mid = f"msg[{i}]"
        role = m.get("role")
        content = m.get("content")
        if role == "tool":
            t = _parts_text(content)
            return [RenderUnit(f"{mid}.tool_result", "tool_result", t, f"tool_result {m.get('tool_call_id', '')}", raw_text=t)]
        out = []
        if m.get("reasoning_content"):
            out.append(RenderUnit(f"{mid}.reasoning", "reasoning", m["reasoning_content"], "reasoning", raw_text=m["reasoning_content"]))
        calls = m.get("tool_calls") or []
        if isinstance(content, list):
            for j, p in enumerate(content):
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    out.append(RenderUnit(f"{mid}.part[{j}]", "text", p.get("text", ""), "text", raw_text=p.get("text", "")))
                else:
                    t = str(p.get("type"))
                    out.append(RenderUnit(f"{mid}.part[{j}]", t, "[" + t + "]", t))
        elif content or not calls:
            t = content if isinstance(content, str) else ""
            out.append(RenderUnit(f"{mid}.text", "text", t, "text", raw_text=t, whole=not calls and not out))
        for k, c in enumerate(calls):
            fn = c.get("function") or {}
            args = fn.get("arguments")
            args = args if isinstance(args, str) else _json_args(args or {})
            out.append(RenderUnit(f"{mid}.tool_call[{k}]", "tool_use", args, f"tool_use {fn.get('name', '')}", raw_text=args, name=fn.get("name")))
        return out

    def unit(self, nid):
        if self._by_id is None:
            self._by_id = {u.id: u for u in self.tool_units()}
            for mu in self.render_units():
                for u in mu:
                    self._by_id[u.id] = u
        return self._by_id.get(nid)

    def unit_text(self, nid):
        """The client-side text a section leaf came from (None for template overhead)."""
        u = self.unit(nid)
        return u.raw_text if u else None


# ----------------------------------------------------------------------------- client identification

KNOWN_CLIENTS = [("claude-cli", "Claude Code"), ("opencode", "OpenCode"), ("ai-sdk", "AI SDK app"), ("openai-node", "OpenAI Node SDK"),
                 ("openai-python", "OpenAI Python SDK"), ("openai/python", "OpenAI Python SDK"), ("aider", "aider"),
                 ("continue", "Continue"), ("cline", "Cline"), ("curl", "curl"), ("python-urllib", "python (urllib script)"),
                 ("python-requests", "python requests"), ("python-httpx", "python httpx"), ("node", "node"), ("undici", "node (undici)"),
                 ("bun", "bun")]
# Clients that are coding agents (the GUI treats everything else as a script or test).
AGENT_CLIENTS = ("Claude Code", "OpenCode", "aider", "Continue", "Cline")
SESSION_HEADERS = ("x-claude-code-session-id", "x-session-id", "x-opencode-session", "x-opencode-session-id", "session_id", "x-conversation-id")


def client_info(headers, body):
    """Who sent a request and which conversation it belongs to.
    -> {"client": name, "version": str|None, "user_agent": str, "session": id|None, "session_source": "header"|"metadata"|"user"|None}"""
    h = {k.lower(): v for k, v in (headers or {}).items()}
    ua = h.get("user-agent", "") or ""
    first = ua.split(" ")[0]
    name, _, version = first.partition("/")
    client = None
    for prefix, label in KNOWN_CLIENTS:
        if first.lower().startswith(prefix) or (prefix == "ai-sdk" and "ai-sdk/" in ua.lower()):
            client = label
            break
    if client is None:
        client = name or "unknown client"
    if client == "AI SDK app" and "opencode" in ua.lower():
        client = "OpenCode"
    session, source = None, None
    for k in SESSION_HEADERS:
        if h.get(k):
            session, source = h[k], "header"
            break
    else:
        b = body if isinstance(body, dict) else {}
        md = b.get("metadata")
        uid = md.get("user_id") if isinstance(md, dict) else None
        if isinstance(uid, str):
            try:
                j = json.loads(uid)
                session = (j.get("session_id") if isinstance(j, dict) else None) or None
            except ValueError:
                session = uid[:64] if uid else None
            source = "metadata" if session else None
        elif isinstance(b.get("user"), str) and b["user"]:
            session, source = b["user"][:64], "user"
    return {"client": client, "version": version or None, "user_agent": ua, "session": session, "session_source": source}
