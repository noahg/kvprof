"""Anthropic Messages -> OpenAI chat conversion, and its calibration against the server.

llama-server renders an Anthropic request by converting it to OpenAI chat messages first. A few details
of that conversion are not documented (how several text blocks are joined, where tool results go relative
to user text in the same message). VARIANT_AXES lists them; `calibrate` picks, for each axis, the value
whose rendered-and-tokenized count equals the server's own /v1/messages/count_tokens.
"""
import json
import time

# Axes that calibration explores. Values listed first are the defaults.
VARIANT_AXES = {
    "system_join": ["\n\n", "\n", ""],          # join of system[] text blocks
    "user_text_join": ["\n", "\n\n", ""],       # join of a user message's text blocks (llama.cpp joins typed parts with \n)
    "assistant_text_join": ["", "\n", "\n\n"],  # join of an assistant message's text blocks
    "tool_result_join": ["\n", "", "\n\n"],     # join of a tool_result's content[] text blocks
    "is_error_prefix": ["", "Error: "],         # what is_error: true adds to a tool_result
    "tool_first": [True, False],                # tool messages before or after the user text in the same message
}
DEFAULT_VARIANT = {k: v[0] for k, v in VARIANT_AXES.items()}


def _json_args(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def anthropic_to_openai(body, variant=None):
    """Return (messages, tools, origins, flags).

    origins[i] describes where messages[i] came from: {"msg": None, "kind": "system"} for the system message,
    {"msg": k, "blocks": [j, ...]} for the k-th Anthropic message (blocks=None when its content was a string).
    """
    v = dict(DEFAULT_VARIANT)
    v.update(variant or {})
    messages, origins = [], []
    flags = {"unsupported_blocks": 0, "thinking_blocks": 0}

    system = body.get("system")
    if system is not None:
        if isinstance(system, str):
            text = system
        else:
            text = v["system_join"].join(b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text")
        messages.append({"role": "system", "content": text})
        origins.append({"msg": None, "kind": "system"})

    tools = []
    for t in body.get("tools") or []:
        tools.append({"type": "function", "function": {
            "name": t.get("name"), "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}})

    for i, m in enumerate(body.get("messages") or []):
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            origins.append({"msg": i, "blocks": None})
            continue
        content = content or []
        if role == "assistant":
            text_parts, text_blocks, tool_calls, tool_blocks, reasoning = [], [], [], [], []
            for j, b in enumerate(content):
                t = b.get("type")
                if t == "text":
                    text_parts.append(b.get("text", "")); text_blocks.append(j)
                elif t == "tool_use":
                    tool_calls.append({"id": b.get("id"), "type": "function",
                                       "function": {"name": b.get("name"), "arguments": _json_args(b.get("input", {}))}})
                    tool_blocks.append(j)
                elif t == "thinking":
                    reasoning.append(b.get("thinking", "")); flags["thinking_blocks"] += 1
                elif t == "redacted_thinking":
                    flags["thinking_blocks"] += 1
                else:
                    flags["unsupported_blocks"] += 1
            msg = {"role": "assistant", "content": v["assistant_text_join"].join(text_parts)}
            if reasoning:
                msg["reasoning_content"] = "".join(reasoning)
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            origins.append({"msg": i, "blocks": text_blocks + tool_blocks, "text_blocks": text_blocks, "tool_blocks": tool_blocks})
        else:
            tool_msgs, user_texts, user_blocks = [], [], []
            for j, b in enumerate(content):
                t = b.get("type")
                if t == "tool_result":
                    text = tool_result_text(b, v)
                    tool_msgs.append(({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": text}, j))
                elif t == "text":
                    user_texts.append(b.get("text", "")); user_blocks.append(j)
                elif t in ("image", "document"):
                    user_texts.append("[" + t + "]"); user_blocks.append(j); flags["unsupported_blocks"] += 1
                else:
                    flags["unsupported_blocks"] += 1
            seq = [(mm, [j]) for mm, j in tool_msgs]
            if user_blocks:
                um = ({"role": role, "content": v["user_text_join"].join(user_texts)}, user_blocks)  # role passes through ("system" mid-conversation stays "system")
                seq = seq + [um] if v["tool_first"] else [um] + seq
            for mm, blocks in seq:
                messages.append(mm)
                origins.append({"msg": i, "blocks": blocks})
    return messages, tools, origins, flags


def tool_result_text(block, variant):
    """The text a tool_result block becomes (without the is_error prefix unless the variant adds one)."""
    c = block.get("content", "")
    if isinstance(c, str):
        text = c
    else:
        text = variant["tool_result_join"].join(x.get("text", "") for x in (c or []) if isinstance(x, dict) and x.get("type") == "text")
    if block.get("is_error"):
        text = variant["is_error_prefix"] + text
    return text


# ----------------------------------------------------------------------------- calibration

def _probe_bodies():
    tool = {"name": "Read", "description": "Read a file from disk.",
            "input_schema": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Absolute path"},
                                                              "limit": {"type": "integer"}}, "required": ["file_path"]}}
    tu = {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {"file_path": "/tmp/a.txt", "limit": 3, "opts": {"x": [1, "two"]}}}
    return [
        ("system_join", {"system": [{"type": "text", "text": "You are terse."}, {"type": "text", "text": "Answer briefly."}],
                         "messages": [{"role": "user", "content": "hi"}]}),
        ("user_text_join", {"system": "S", "messages": [{"role": "user", "content": [{"type": "text", "text": "first block"},
                                                                                     {"type": "text", "text": "second block"}]}]}),
        ("assistant_text_join", {"system": "S", "tools": [tool], "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "text", "text": "Let me look."}, {"type": "text", "text": "Reading now."}, tu]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "line one\nline two"}]}]}),
        ("tool_result_join", {"system": "S", "tools": [tool], "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [tu]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01",
                                          "content": [{"type": "text", "text": "part A"}, {"type": "text", "text": "part B"}]}]}]}),
        ("is_error_prefix", {"system": "S", "tools": [tool], "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [tu]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "is_error": True, "content": "no such file"}]}]}),
        ("tool_first", {"system": "S", "tools": [tool], "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [tu, dict(tu, id="toolu_02", input={"file_path": "/tmp/b.txt"})]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "A"},
                                         {"type": "tool_result", "tool_use_id": "toolu_02", "content": "B"},
                                         {"type": "text", "text": "<system-reminder>reminder text</system-reminder>"}]}]}),
        ("thinking", {"system": "S", "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": "x"}, {"type": "text", "text": "done"}]},
            {"role": "user", "content": "and?"}]}),
    ]


def rendered_count(backend, body, variant):
    """Tokens of an Anthropic body rendered through the server's template with a given conversion variant."""
    msgs, tools, _, _ = anthropic_to_openai(body, variant)
    rendered = backend.renderer.render(msgs, tools)
    return len(backend.tokenizer.tokenize(rendered, pieces=False)), rendered


def calibrate(backend, store, model, extra_bodies=(), verbose=True):
    """Pick the conversion variant whose rendered count equals the server's count_tokens, axis by axis."""
    variant = dict(DEFAULT_VARIANT)
    report = []
    for axis, body in _probe_bodies():
        target = backend.client.count_tokens(body, model)
        base = rendered_count(backend, body, variant)[0]
        if base == target:
            report.append(f"  {axis:22s} ok with default ({target} tok)")
            continue
        if axis not in VARIANT_AXES:
            report.append(f"  {axis:22s} MISMATCH {base} vs count_tokens {target} (no axis to adjust)")
            continue
        found = None
        for val in VARIANT_AXES[axis]:
            if val == variant[axis]:
                continue
            n = rendered_count(backend, body, dict(variant, **{axis: val}))[0]
            if n == target:
                found = val
                break
        if found is not None:
            variant[axis] = found
            report.append(f"  {axis:22s} -> {found!r} ({target} tok; default gave {base})")
        else:
            report.append(f"  {axis:22s} NO MATCH: count_tokens {target}, default {base}; keeping default")
    for label, body in extra_bodies:
        try:
            target = backend.client.count_tokens(body, model)
            n = rendered_count(backend, body, variant)[0]
            report.append(f"  {label:22s} {'ok' if n == target else 'MISMATCH'} rendered {n} vs count_tokens {target}")
        except Exception as e:
            report.append(f"  {label:22s} error: {e}")
    info = {"model_path": backend.server_info().model_path, "variant": variant, "verified_at": time.time(), "report": report}
    store.save_conversion(info)
    if verbose:
        print("conversion calibration (rendered-and-tokenized vs /v1/messages/count_tokens):")
        print("\n".join(report))
        print(f"  variant: {variant}")
    return variant


def load_variant(backend, store, model, verbose=False):
    """The calibrated variant for the loaded model; calibrates on first use or after a model change."""
    info = store.load_conversion()
    if info and info.get("model_path") == backend.server_info().model_path and isinstance(info.get("variant"), dict):
        return dict(DEFAULT_VARIANT, **info["variant"])
    try:
        return calibrate(backend, store, model, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"calibration failed ({e!r}); using the default conversion variant")
        return dict(DEFAULT_VARIANT)
