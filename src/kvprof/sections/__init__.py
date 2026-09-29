"""Section trees: which part of the rendered prompt (system text, each tool, each message block) owns which tokens.

A template profile (`chatml_qwen`, `generic`) builds a tree of nodes with character ranges into the rendered
prompt; `assign_tokens` turns those into token ranges whose counts sum exactly to the prompt length.
"""
import bisect
import json



def node(nid, kind, label, cs, ce, **kw):
    d = {"id": nid, "kind": kind, "label": label, "char": [cs, ce], "tok": None, "tokens": 0, "pct": 0.0, "cum": 0,
         "raw_tokens": None, "preview": None, "approx": False, "children": []}
    d.update(kw)
    return d


def _preview(text, n=160):
    if text is None:
        return None
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    text = text.replace("\n", "⏎")
    return text[:n] + ("…" if len(text) > n else "")


def _make_contiguous(nodes, end_char):
    """Each node ends where the next begins; children fill their parent exactly."""
    for k, n in enumerate(nodes):
        n["char"][1] = nodes[k + 1]["char"][0] if k + 1 < len(nodes) else end_char
        if n["char"][1] < n["char"][0]:
            n["char"][1] = n["char"][0]
        if n["children"]:
            n["children"][0]["char"][0] = n["char"][0]
            _make_contiguous(n["children"], n["char"][1])


def _validate(nodes):
    prev = -1
    for n in nodes:
        if n["char"][0] < prev:
            n["approx"] = True
        prev = n["char"][1]
        if n["children"]:
            _validate(n["children"])


def assign_tokens(tree, c2b, bstart, n_tokens):
    """Fill tok ranges: a boundary falls at the first token starting at or after the section's first char,
    so a token straddling two sections belongs to the earlier one. Counts sum exactly to n_tokens."""
    leaves = []

    def walk(nodes, parent_end_tok):
        for k, n in enumerate(nodes):
            cs = n["char"][0]
            ts = bisect.bisect_left(bstart, c2b[cs]) if cs < len(c2b) else n_tokens
            if k + 1 < len(nodes):
                ncs = nodes[k + 1]["char"][0]
                te = bisect.bisect_left(bstart, c2b[ncs]) if ncs < len(c2b) else n_tokens
            else:
                te = parent_end_tok
            ts = min(ts, te)
            n["tok"] = [ts, te]
            n["tokens"] = te - ts
            n["cum"] = ts
            n["pct"] = round(100.0 * (te - ts) / n_tokens, 2) if n_tokens else 0.0
            if n["children"]:
                walk(n["children"], te)
            else:
                leaves.append([ts, te, n["id"]])
    walk(tree, n_tokens)
    if tree:
        tree[0]["tok"][0] = 0
    return leaves


def section_at(leaves, tok):
    lo, hi = 0, len(leaves)
    while lo < hi:
        mid = (lo + hi) // 2
        if leaves[mid][1] <= tok:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(leaves) and leaves[lo][0] <= tok < leaves[lo][1]:
        return leaves[lo][2]
    return leaves[-1][2] if leaves and tok >= leaves[-1][1] else None


def iter_nodes(nodes):
    for n in nodes:
        yield n
        yield from iter_nodes(n["children"])


def find_node(tree, nid):
    for n in iter_nodes(tree):
        if n["id"] == nid:
            return n
    return None


def leaf_source_text(nid, request):
    """The client-side text a leaf came from (None for template overhead)."""
    return request.unit_text(nid)


def add_raw_counts(tree, source_text, raw):
    """raw_tokens on every node: leaves count their source text (source_text(nid) -> str|None) with the
    tokenizer alone, groups sum their children. Rendered minus raw is the template's overhead."""
    def walk(nodes):
        for n in nodes:
            if n["children"]:
                walk(n["children"])
                vals = [c["raw_tokens"] for c in n["children"]]
                n["raw_tokens"] = sum(v for v in vals if v is not None) if any(v is not None for v in vals) else None
            else:
                n["raw_tokens"] = raw.count(source_text(n["id"]))
    walk(tree)


def build_sections(profile, rendered, request, messages, origins, units):
    """Dispatch to a template profile's builder. -> (tree, mapping_ok, reason)"""
    if profile == "chatml_qwen":
        from .chatml_qwen import build_sections_chatml
        return build_sections_chatml(rendered, request, messages, origins, units)
    from .generic import build_sections_generic
    return build_sections_generic(rendered, request, messages, origins, units)
