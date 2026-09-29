"""Small helpers shared by the recorder and the analyzer: addresses, logging, JSON files, byte offsets."""
import bisect
import json
import os
import threading
import time

_print_lock = threading.Lock()


def parse_addr(s, default_port):
    """'http://host:port/', 'host:port', ':port' or 'port' -> (host, port)."""
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("/", 1)[0]
    if s.isdigit():
        return "127.0.0.1", int(s)
    host, _, port = s.rpartition(":")
    if not host and not port.isdigit():
        return (port or "127.0.0.1"), int(default_port)
    return (host or "127.0.0.1"), int(port or default_port)


def now_hms(t=None):
    return time.strftime("%H:%M:%S", time.localtime(t))


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def display_path(path):
    """Relative to the working directory when it is below it, else absolute."""
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return path
    return os.path.abspath(path) if rel.startswith("..") else rel


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path, obj, indent=None):
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ----------------------------------------------------------------------------- bytes / offsets

def piece_bytes(p):
    """A /tokenize piece is a string, or a list of byte values when it is not valid UTF-8 on its own."""
    if isinstance(p, str):
        return p.encode("utf-8")
    if isinstance(p, list):
        return bytes(p)
    return b""


def byte_starts(tokens):
    """tokens: list of (id, piece_bytes) -> cumulative byte offsets, len(tokens)+1 entries."""
    starts = [0]
    b = 0
    for _, piece in tokens:
        b += len(piece)
        starts.append(b)
    return starts


def char_to_byte_table(text):
    """table[i] = byte offset of char i in text.encode('utf-8'); len(text)+1 entries."""
    table = [0] * (len(text) + 1)
    b = 0
    for i, ch in enumerate(text):
        table[i] = b
        o = ord(ch)
        b += 1 if o < 0x80 else 2 if o < 0x800 else 3 if o < 0x10000 else 4
    table[len(text)] = b
    return table


def tok_at(bstart, byte_off):
    """Index of the token containing byte offset (a boundary belongs to the token starting there)."""
    return max(0, bisect.bisect_right(bstart, byte_off) - 1)


def tok_end_at(bstart, byte_off):
    """Exclusive token end for a section ending at byte offset (a token straddling the end belongs to it)."""
    return bisect.bisect_left(bstart, byte_off)


def lcp_len(a, b):
    """Length of the longest common prefix of two sequences."""
    n = min(len(a), len(b))
    step = 4096
    i = 0
    while i < n and a[i:i + step] == b[i:i + step]:
        i += step
    while i < n and a[i] == b[i]:
        i += 1
    return min(i, n)


def fmt_n(n):
    return "?" if n is None else f"{n:,}"
