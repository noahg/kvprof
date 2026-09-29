"""Section builder for ChatML templates with Qwen-style tool markup (<tools>, <tool_call>, <tool_response>).

Walks the rendered prompt turn by turn (`<|im_start|>role ... <|im_end|>`), placing the system text, each
tool definition, and every content block of every message with exact character ranges.
"""
from . import _make_contiguous, _preview, _validate, node

MARK = {
    "im_start": "<|im_start|>", "im_end": "<|im_end|>",
    "tools_hdr": "\n\n# Tools", "tools_open": "<tools>", "tools_close": "</tools>",
    "fn_open": "<function>", "fn_close": "</function>",
    "tool_call_open": "<tool_call>", "tool_call_close": "</tool_call>",
    "tool_resp_open": "<tool_response>", "tool_resp_close": "</tool_response>",
}


def _find_block(rendered, text, start, end):
    """Locate a content block's text inside rendered[start:end]. -> (pos, approx)"""
    if not text:
        return start, True
    i = rendered.find(text, start, end)
    if i >= 0:
        return i, False
    s = text.strip()
    if s:
        i = rendered.find(s, start, end)
        if i >= 0:
            return i, True
        i = rendered.find(s[:64], start, end)
        if i >= 0:
            return i, True
    return start, True


def _past_im_end(R, pos):
    """If rendered[pos:] starts with <|im_end|> (optionally followed by a newline), return the offset after it."""
    if R.startswith(MARK["im_end"], pos):
        pos += len(MARK["im_end"])
        if R.startswith("\n", pos):
            pos += 1
    return pos


def build_sections_chatml(rendered, request, messages, origins, units):
    """-> (tree list, mapping_ok, reason). Char ranges only; token ranges are filled in later.

    messages/origins: the converted OpenAI messages and where each came from; units[k]: the RenderUnits of
    messages[k] in render order; request.tools_units: one unit per tool definition."""
    R = rendered
    starts = []
    i = R.find(MARK["im_start"])
    while i >= 0:
        starts.append(i)
        i = R.find(MARK["im_start"], i + 1)
    if not starts:
        return [node("prompt", "prompt", "prompt", 0, len(R))], False, "no <|im_start|> markers"
    turns = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(R)
        nl = R.find("\n", s)
        role = R[s + len(MARK["im_start"]):nl] if nl >= 0 else "?"
        turns.append((s, e, role, (nl + 1) if nl >= 0 else s))
    tree = []
    ti = 0
    tool_units = request.tool_units()

    # -- system turn (the system message, and/or the tool definitions the template puts there)
    has_sys_msg = bool(messages) and messages[0]["role"] in ("system", "developer")
    if turns[0][2] == "system" and (has_sys_msg or tool_units):
        s, e, _, cstart = turns[0]
        tools_open = R.find(MARK["tools_open"], cstart, e)
        if tools_open >= 0 and tool_units:
            if has_sys_msg:
                pre = R.rfind(MARK["tools_hdr"], cstart, tools_open)
                pre = pre if pre >= 0 else tools_open
                tree.append(node("system", "system", "system", s, pre, children=[node("system.text", "text", "system text", s, pre)]))
            else:
                pre = s
            tn = node("tools", "tools", "tools", pre, e)
            tn["children"].append(node("tools.preamble", "template", "tools preamble", pre, tools_open))
            cur = tools_open
            for tu in tool_units:
                name = tu.name
                p = R.find(MARK["fn_open"] + "\n<name>" + name + "</name>", cur, e)
                approx = False
                if p < 0:
                    p = R.find("<name>" + name + "</name>", cur, e)
                    approx = True
                if p < 0:
                    tn["children"].append(node(tu.id, "tool", name, cur, cur, approx=True))
                    continue
                q = R.find(MARK["fn_close"], p, e)
                q = (q + len(MARK["fn_close"])) if q >= 0 else p
                tn["children"].append(node(tu.id, "tool", name, p, q, approx=approx))
                cur = q
            close = R.find(MARK["tools_close"], cur, e)
            tn["children"].append(node("tools.instructions", "template", "tool-call instructions", close if close >= 0 else cur, e))
            tree.append(tn)
        elif has_sys_msg:
            tree.append(node("system", "system", "system", s, e, children=[node("system.text", "text", "system text", s, e)]))
        else:
            tree.append(node("system", "system", "system", s, e, children=[node("system.template", "template", "template default system prompt", s, e)]))
        ti = 1
        oi = 1 if has_sys_msg else 0
    else:
        oi = 0

    # -- message turns: walk the converted messages with a cursor
    msgs_node = node("messages", "messages", "messages", turns[ti][0] if ti < len(turns) else len(R), len(R))
    tree.append(msgs_node)
    cursor = turns[ti][0] if ti < len(turns) else len(R)
    per_msg = {}
    reason = None
    ok = True
    for k in range(oi, len(messages)):
        m, org = messages[k], origins[k]
        ai = org["msg"]
        mid = f"msg[{ai}]"
        mnode = per_msg.get(ai)
        if mnode is None:
            role0 = request.message_role(ai) or m["role"]
            mnode = node(mid, "message", f"{role0} #{ai}", cursor, cursor, role=role0)
            per_msg[ai] = mnode
            msgs_node["children"].append(mnode)
        role = m["role"]
        mu = units[k]
        if role == "tool":
            p = R.find(MARK["tool_resp_open"], cursor)
            if p < 0:
                ok, reason = False, f"tool_response for {mid} not found"
                break
            hdr = MARK["im_start"] + "user\n"
            if R.startswith(hdr, p - len(hdr)):
                p -= len(hdr)
            q = R.find(MARK["tool_resp_close"], p)
            q = (q + len(MARK["tool_resp_close"])) if q >= 0 else p
            u = mu[0]
            mnode["children"].append(node(u.id, "tool_result", u.label, p, q, preview=_preview(m.get("content", "")), is_error=u.is_error))
            cursor = _past_im_end(R, q)  # a run of tool responses shares one user turn; the last one owns the <|im_end|>
        elif role in ("user", "system"):
            hdr = MARK["im_start"] + role + "\n"
            p = R.find(hdr, cursor)
            if p < 0:
                ok, reason = False, f"user turn for {mid} not found"
                break
            e = R.find(MARK["im_end"], p)
            e = e if e >= 0 else len(R)
            cstart = p + len(hdr)
            if len(mu) == 1 and mu[0].whole:
                mnode["children"].append(node(mu[0].id, "text", "text", p, e, preview=_preview(mu[0].text)))
            else:
                c = cstart
                for u in mu:
                    pos, approx = _find_block(R, u.text, c, e)
                    mnode["children"].append(node(u.id, u.kind, u.label, pos, min(pos + len(u.text), e), preview=_preview(u.text), approx=approx))
                    c = min(pos + len(u.text), e)
            cursor = _past_im_end(R, e)
        elif role == "assistant":
            hdr = MARK["im_start"] + "assistant\n"
            p = R.find(hdr, cursor)
            if p < 0:
                ok, reason = False, f"assistant turn for {mid} not found"
                break
            e = R.find(MARK["im_end"], p)
            e = e if e >= 0 else len(R)
            cstart = p + len(hdr)
            if len(mu) == 1 and mu[0].whole:
                mnode["children"].append(node(mu[0].id, "text", "text", p, e, preview=_preview(mu[0].text)))
            else:
                c = cstart
                for u in mu:
                    if u.kind == "tool_use":
                        pos = R.find(MARK["tool_call_open"], c, e)
                        approx = pos < 0
                        pos = pos if pos >= 0 else c
                        q = R.find(MARK["tool_call_close"], pos, e)
                        q = (q + len(MARK["tool_call_close"])) if q >= 0 else e
                        mnode["children"].append(node(u.id, "tool_use", u.label, pos, q, preview=_preview(u.text), approx=approx))
                        c = q
                    elif u.kind == "reasoning":
                        continue  # the template decides where (and whether) reasoning is rendered; it stays in the turn
                    else:
                        pos, approx = _find_block(R, u.text, c, e)
                        mnode["children"].append(node(u.id, "text", "text", pos, min(pos + len(u.text), e), preview=_preview(u.text), approx=approx))
                        c = min(pos + len(u.text), e)
            cursor = _past_im_end(R, e)
        else:
            ok, reason = False, f"unexpected role {role}"
            break

    # -- generation prompt: the trailing empty assistant turn
    gen_start = R.rfind(MARK["im_start"] + "assistant\n")
    if gen_start >= 0 and R.find(MARK["im_end"], gen_start) < 0:
        msgs_node["char"][1] = gen_start
        tree.append(node("generation_prompt", "template", "generation prompt", gen_start, len(R)))
    else:
        msgs_node["char"][1] = len(R)
    if not ok:
        # keep turn-level ranges only
        msgs_node["children"] = [node(f"turn[{k}]", "turn", f"{turns[k][2]} turn", turns[k][0], turns[k][1]) for k in range(ti, len(turns))]
    _make_contiguous(tree, len(R))
    _validate(tree)
    return tree, ok, reason
