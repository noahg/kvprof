"""Token-level diff of a prompt against an earlier one, and the request kind that follows from it."""
from .sections import section_at
from .util import lcp_len


def diff_analysis(cur, parent, cur_req, parent_req):
    """cur/parent: analyses with ids, leaves, rendered, bstart. cur_req/parent_req: their Requests (or None)."""
    ids, pids = cur["ids"], parent["ids"]
    lcp = lcp_len(ids, pids)
    d = {"parent_id": parent["id"], "lcp": lcp, "parent_len": len(pids), "cur_len": len(ids),
         "appended": len(ids) - lcp, "parent_dropped": len(pids) - lcp, "first_divergent": None, "changed_text": None,
         "end_extension": lcp >= len(pids) - 2}
    if lcp < len(ids) and lcp < len(pids):
        cs = cur["leaves"] and section_at(cur["leaves"], lcp)
        ps = parent["leaves"] and section_at(parent["leaves"], lcp)
        rb = cur["rendered"].encode("utf-8")
        pb = parent["rendered"].encode("utf-8")
        cb, pbb = cur["bstart"][lcp], parent["bstart"][lcp]
        d["first_divergent"] = {
            "tok": lcp, "cur_section": cs, "parent_section": ps,
            "cur_piece": rb[cb:cur["bstart"][lcp + 1]].decode("utf-8", "replace"),
            "parent_piece": pb[pbb:parent["bstart"][lcp + 1]].decode("utf-8", "replace"),
            "cur_context": rb[max(0, cb - 160):cb + 160].decode("utf-8", "replace"),
            "parent_context": pb[max(0, pbb - 160):pbb + 160].decode("utf-8", "replace"),
            "cur_context_split": len(rb[max(0, cb - 160):cb].decode("utf-8", "replace")),
            "parent_context_split": len(pb[max(0, pbb - 160):pbb].decode("utf-8", "replace")),
        }
        if cs and cs == ps and cur_req is not None and parent_req is not None:
            try:
                a = parent_req.unit_text(cs) or ""
                b = cur_req.unit_text(cs) or ""
                if a != b:
                    p = 0
                    n = min(len(a), len(b))
                    while p < n and a[p] == b[p]:
                        p += 1
                    s = 0
                    while s < n - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
                        s += 1
                    d["changed_text"] = {"section": cs, "removed": a[p:len(a) - s][:400], "added": b[p:len(b) - s][:400],
                                         "at_char": p, "parent_chars": len(a), "cur_chars": len(b)}
            except Exception:
                pass
    elif lcp < len(ids):
        d["first_divergent"] = {"tok": lcp, "cur_section": section_at(cur["leaves"], lcp), "parent_section": None,
                                "cur_piece": None, "parent_piece": None, "cur_context": None, "parent_context": None}
    return d


KINDS = ("cold", "warm", "main_turn", "tool_roundtrip", "side", "rewrite", "compaction", "retry")


def classify(a, request, diff, parent):
    """-> (kind, subkind, hint). kind is one of KINDS; subkind names the divergent section of a rewrite."""
    toolsn = next((n for n in a["sections"] if n["id"] == "tools"), None)
    has_tools = bool(toolsn and toolsn["tokens"] > 0) or (toolsn is None and request.has_tools)
    if diff is None:
        kind = "cold"
    elif diff["lcp"] == diff["parent_len"] == diff["cur_len"]:
        kind = "retry"
    elif diff["end_extension"]:
        kind = "tool_roundtrip" if "tool_result" in request.last_message_kinds else "main_turn"
    elif diff["lcp"] < 0.5 * diff["cur_len"]:
        kind = "cold"
    else:
        fd = diff.get("first_divergent") or {}
        sec = fd.get("cur_section") or ""
        if sec.startswith("msg[") and parent and (parent.get("n_messages") or 0) > request.n_messages and not diff.get("changed_text"):
            kind = "compaction"  # fewer messages and the divergent block is not a small in-place edit
        else:
            kind = "rewrite"
    hint = None
    u = a.get("usage") or {}
    if kind == "cold" and (u.get("cached") or 0) > 0.5 * a["n_tokens"]:
        kind, hint = "warm", "prefix reused from a request the tap did not see"
    if not has_tools and kind in ("cold", "rewrite", "retry"):
        kind = "side"
        hint = request.hint_text().replace("\n", " ")[:60]
    subkind = None
    if kind in ("rewrite", "compaction") and diff and diff.get("first_divergent"):
        subkind = diff["first_divergent"].get("cur_section")
    return kind, subkind, hint
