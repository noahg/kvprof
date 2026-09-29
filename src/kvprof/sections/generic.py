"""Section builder for any chat template: locate each RenderUnit's text in the rendered prompt.

Units are searched in render order (system text, tool definitions, then every message's units) with a
monotonic cursor. A unit's range runs from where it was found to where the next unit was found, so the
template's markers between two units (turn headers, closing tags) attach to the earlier unit: boundaries
are approximate to the template's turn markers. Token counts still sum exactly to the prompt.

Considered and rejected: learning per-role markers from probe renders; it fails when the closing marker
is identical across roles (ChatML).
"""
import json
import re

from . import _make_contiguous, _preview, _validate, node

SHORT = 16          # needles shorter than this are only searched near the cursor
NEAR = 4096         # ... within this many characters
CALL_WINDOW = 2048  # a tool call's argument value must follow its name within this many characters


def _longest_line(text, min_len=SHORT):
    lines = [ln.strip() for ln in text.splitlines()]
    best = max(lines, key=len, default="")
    return best if len(best) >= min_len else None


def _string_values(obj, out):
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _string_values(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _string_values(v, out)
    return out


class _Locator:
    def __init__(self, rendered):
        self.R = rendered
        self.n = len(rendered)

    def find(self, needle, c):
        if not needle:
            return -1
        limit = self.n if len(needle) >= SHORT else min(self.n, c + NEAR + len(needle))
        return self.R.find(needle, c, limit)

    def find_word(self, word, c, limit=None):
        if not word:
            return None
        m = re.compile(r"(?<![\w-])" + re.escape(word) + r"(?![\w-])").search(self.R, c, limit or self.n)
        return m

    def locate(self, u, c):
        """-> (start, needle_end, exact) or None. start is where the unit's text begins (estimated for partial needles)."""
        t = u.text or ""
        if u.kind == "tool":
            return self._locate_tool(u, c)
        if t:
            p = self.find(t, c)
            if p >= 0:
                return p, p + len(t), True
        if u.kind == "tool_use":
            hit = self._locate_call(u, c)
            if hit:
                return hit
        s = t.strip()
        cands = []
        if s:
            off = t.find(s)
            cands.append((s, off))
            head = s[:48] if u.kind == "tool_result" else s[:64]
            cands.append((head, off))
            line = _longest_line(t)
            if line:
                cands.append((line, t.find(line)))
        for needle, off in cands:
            p = self.find(needle, c)
            if p >= 0:
                return max(c, p - max(0, off)), p + len(needle), False
        return None

    def _locate_tool(self, u, c):
        name = u.name or ""
        for needle in (f"<name>{name}</name>", f'"name": "{name}"', f'"name":"{name}"'):
            p = self.find(needle, c)
            if p >= 0:
                return p, p + len(needle), False
        m = self.find_word(name, c)
        if m:
            return m.start(), m.end(), False
        return None

    def _locate_call(self, u, c):
        """The tool name, followed within CALL_WINDOW by the longest string value of the arguments."""
        try:
            args = json.loads(u.text) if u.text else {}
        except ValueError:
            args = {}
        values = sorted(_string_values(args, []), key=len, reverse=True)
        value = values[0] if values and values[0] else None
        pos = c
        for _ in range(64):
            m = self.find_word(u.name or "", pos)
            if not m:
                return None
            if value is None:
                return m.start(), m.end(), False
            end = min(self.n, m.end() + CALL_WINDOW + len(value) * 2)
            for v in (value, json.dumps(value, ensure_ascii=False)[1:-1], json.dumps(value)[1:-1]):
                q = self.R.find(v, m.end(), end)
                if q >= 0:
                    return m.start(), q + len(v), False
            pos = m.end()
        return None


def build_sections_generic(rendered, request, messages, origins, units):
    """-> (tree, mapping_ok, reason). Same node ids as the ChatML builder, positions approximate."""
    R = rendered
    L = _Locator(R)
    has_sys = bool(messages) and origins[0].get("msg") is None
    order = []  # (group, unit, message index or None)
    if has_sys:
        for u in units[0]:
            order.append(("system", u, None))
    for u in request.tool_units():
        order.append(("tools", u, None))
    for k in range(1 if has_sys else 0, len(messages)):
        for u in units[k]:
            order.append(("messages", u, origins[k]["msg"]))

    c = 0
    placed = []  # (group, unit, msg, start, end, exact, found)
    for group, u, mi in order:
        hit = L.locate(u, c)
        if hit is None:
            placed.append((group, u, mi, c, c, False, False))
            continue
        start, end, exact = hit
        start = max(start, c)
        placed.append((group, u, mi, start, end, exact, True))
        c = max(c, end)

    msg_units = [p for p in placed if p[0] == "messages"]
    n_found = sum(1 for p in msg_units if p[6])
    if msg_units and not msg_units[0][6]:
        return [node("prompt", "prompt", "prompt", 0, len(R), approx=True)], False, "first message not found in the rendered prompt"
    turn_level = bool(msg_units) and n_found < 0.5 * len(msg_units)

    # approx: a node is exact only if it and the unit after it were found by their full text
    approx = {}
    for i, p in enumerate(placed):
        nxt = placed[i + 1] if i + 1 < len(placed) else None
        approx[i] = not (p[5] and (nxt is None or nxt[5]))

    tree = []
    sys_p = [(i, p) for i, p in enumerate(placed) if p[0] == "system"]
    tool_p = [(i, p) for i, p in enumerate(placed) if p[0] == "tools"]
    if sys_p:
        i, p = sys_p[0]
        tree.append(node("system", "system", "system", 0, p[4],
                         children=[node(p[1].id, "text", "system text", 0, p[4], preview=_preview(p[1].text), approx=approx[i])]))
    if tool_p:
        pre = sys_p[0][1][4] if sys_p else 0
        first = tool_p[0][1][3]
        tn = node("tools", "tools", "tools", pre, len(R), children=[node("tools.preamble", "template", "tools preamble", pre, max(pre, first))])
        for i, p in tool_p:
            tn["children"].append(node(p[1].id, "tool", p[1].label, p[3], p[3], approx=approx[i] or not p[6]))
        tree.append(tn)

    # generation prompt: the tail after the last message unit, if it is at least as long as a turn separator
    last_end = max((p[4] for p in msg_units if p[6]), default=c)
    seps = []
    prev_msg, prev_end = None, None
    for p in msg_units:
        if not p[6]:
            continue
        if prev_msg is not None and p[2] != prev_msg:
            seps.append(p[3] - prev_end)
        prev_msg, prev_end = p[2], p[4]
    tail = len(R) - last_end
    gen = tail > 0 and (not seps or tail >= min(seps))

    m_start = msg_units[0][3] if msg_units else c
    mn = node("messages", "messages", "messages", m_start, len(R))
    if turn_level:
        seen = set()
        for i, p in enumerate(placed):
            if p[0] != "messages" or not p[6] or p[2] in seen:
                continue
            seen.add(p[2])
            role = request.message_role(p[2]) or "?"
            mn["children"].append(node(f"turn[{p[2]}]", "turn", f"{role} turn", p[3], p[3], approx=True))
    else:
        per_msg = {}
        for i, p in enumerate(placed):
            if p[0] != "messages":
                continue
            u, mi = p[1], p[2]
            mnode = per_msg.get(mi)
            if mnode is None:
                role = request.message_role(mi) or "?"
                mnode = node(f"msg[{mi}]", "message", f"{role} #{mi}", p[3], p[3], role=role)
                per_msg[mi] = mnode
                mn["children"].append(mnode)
            kind = "text" if u.whole else u.kind
            leaf = node(u.id, kind, "text" if u.whole else u.label, p[3], p[3], preview=_preview(u.text), approx=approx[i] or not p[6])
            if u.kind == "tool_result":
                leaf["is_error"] = u.is_error
            mnode["children"].append(leaf)
            if leaf["approx"]:
                mnode["approx"] = True
    if msg_units or not tree:
        tree.append(mn)
    if gen:
        tree.append(node("generation_prompt", "template", "generation prompt", last_end, len(R)))
    _make_contiguous(tree, len(R))
    _validate(tree)
    if turn_level:
        return tree, False, f"only {n_found} of {len(msg_units)} message parts found in the rendered prompt; turn-level sections"
    return tree, True, None
