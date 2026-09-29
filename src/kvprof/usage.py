"""Raw usage extraction from recorded responses, for both wire shapes (streamed or not).

The result is the trace's `usage` record (schema 2):
  {"api", "fields": {usage block as the server sent it}, "timings": {llama.cpp timings}|None,
   "stop_reason", "model", "n_deltas", "n_text_deltas", "output_tokens_estimated"}
Normalization into prompt/cached/uncached is the backend's job (backends/*/usage.py).
"""
import json


def _new(api):
    return {"api": api, "fields": {}, "timings": None, "stop_reason": None, "model": None,
            "n_deltas": 0, "n_text_deltas": 0, "output_tokens_estimated": False}


def anthropic_usage_from_events(events):
    u = _new("anthropic")
    f = u["fields"]
    for ev, data in events:
        if not isinstance(data, dict):
            continue
        if ev == "message_start":
            msg = data.get("message", {})
            u["model"] = msg.get("model")
            f.update(msg.get("usage") or {})
        elif ev == "content_block_delta":
            u["n_deltas"] += 1
            if (data.get("delta") or {}).get("type") == "text_delta":
                u["n_text_deltas"] += 1
        elif ev == "message_delta":
            f.update(data.get("usage") or {})
            if (data.get("delta") or {}).get("stop_reason"):
                u["stop_reason"] = data["delta"]["stop_reason"]
        if isinstance(data.get("timings"), dict):
            u["timings"] = data["timings"]
    if f.get("output_tokens") is None and u["n_deltas"]:
        f["output_tokens"] = u["n_deltas"]
        u["output_tokens_estimated"] = True
    return u


def anthropic_usage_from_json(j):
    u = _new("anthropic")
    u["fields"] = dict(j.get("usage") or {})
    u["stop_reason"] = j.get("stop_reason")
    u["model"] = j.get("model")
    if isinstance(j.get("timings"), dict):
        u["timings"] = j["timings"]
    return u


def openai_usage_from_events(events):
    u = _new("openai")
    for _, data in events:
        if not isinstance(data, dict) or "_raw" in data:
            continue
        if data.get("model") and not u["model"]:
            u["model"] = data["model"]
        for ch in data.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                u["n_deltas"] += 1
            if d.get("content"):
                u["n_text_deltas"] += 1
            if ch.get("finish_reason"):
                u["stop_reason"] = ch["finish_reason"]
        if isinstance(data.get("usage"), dict):
            u["fields"] = dict(data["usage"])
        if isinstance(data.get("timings"), dict):
            u["timings"] = data["timings"]
    if u["fields"].get("completion_tokens") is None and u["n_deltas"]:
        u["fields"]["completion_tokens"] = u["n_deltas"]
        u["output_tokens_estimated"] = True
    return u


def openai_usage_from_json(j):
    u = _new("openai")
    u["fields"] = dict(j.get("usage") or {})
    u["model"] = j.get("model")
    for ch in j.get("choices") or []:
        if ch.get("finish_reason"):
            u["stop_reason"] = ch["finish_reason"]
    if isinstance(j.get("timings"), dict):
        u["timings"] = j["timings"]
    return u


def extract_raw_usage(api, events=None, body_text=None):
    """From parsed SSE events (streamed) or a JSON body (not streamed). None when there is nothing to read."""
    if events is not None:
        return anthropic_usage_from_events(events) if api == "anthropic" else openai_usage_from_events(events)
    if not body_text:
        return None
    try:
        j = json.loads(body_text)
    except ValueError:
        return None
    if not isinstance(j, dict) or "error" in j:
        return None
    return anthropic_usage_from_json(j) if api == "anthropic" else openai_usage_from_json(j)


def upgrade_schema1_usage(usage, stop_reason=None):
    """A schema-1 trace's flat Anthropic usage dict -> the schema-2 record."""
    if not usage:
        return None
    if "fields" in usage and "api" in usage:
        return usage
    u = _new("anthropic")
    u["fields"] = {k: usage[k] for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")
                   if usage.get(k) is not None}
    u["stop_reason"] = stop_reason
    for k in ("n_deltas", "n_text_deltas", "output_tokens_estimated"):
        if k in usage:
            u[k] = usage[k]
    return u


class StreamWatcher:
    """Timestamps of the first stream event, first delta and first text delta, for either wire shape."""

    def __init__(self, api):
        self.api = api
        self.t_first_event = None
        self.t_first_delta = None
        self.t_first_text_delta = None

    def observe(self, event, data, t):
        if self.t_first_event is None:
            self.t_first_event = t
        if self.api == "anthropic":
            if event == "content_block_delta":
                if self.t_first_delta is None:
                    self.t_first_delta = t
                if self.t_first_text_delta is None and isinstance(data, dict) and (data.get("delta") or {}).get("type") == "text_delta":
                    self.t_first_text_delta = t
            return
        if not isinstance(data, dict):
            return
        for ch in data.get("choices") or []:
            d = ch.get("delta") or {}
            if (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")) and self.t_first_delta is None:
                self.t_first_delta = t
            if d.get("content") and self.t_first_text_delta is None:
                self.t_first_text_delta = t

    def timing(self):
        return {"t_first_event": self.t_first_event, "t_message_start": self.t_first_event,
                "t_first_delta": self.t_first_delta, "t_first_text_delta": self.t_first_text_delta}
