"""The trace directory: recorded traces (<id>.json), cached analyses (<id>.analysis.json), the calibrated
conversion (.conversion.json) and the raw-text token-count cache (.rawcache.json)."""
import hashlib
import os
import threading

from .util import load_json, save_json_atomic

CONVERSION_FILE = ".conversion.json"
RAWCACHE_FILE = ".rawcache.json"


class TraceStore:
    def __init__(self, trace_dir):
        self.trace_dir = trace_dir

    def ensure(self):
        os.makedirs(self.trace_dir, exist_ok=True)

    def list_ids(self):
        try:
            names = os.listdir(self.trace_dir)
        except OSError:
            return []
        return sorted(n[:-5] for n in names if n.endswith(".json") and not n.endswith(".analysis.json") and not n.startswith("."))

    def trace_path(self, trace_id):
        return os.path.join(self.trace_dir, trace_id + ".json")

    def analysis_path(self, trace_id):
        return os.path.join(self.trace_dir, trace_id + ".analysis.json")

    def load_trace(self, trace_id):
        return load_json(self.trace_path(trace_id))

    def load_analysis(self, trace_id):
        try:
            return load_json(self.analysis_path(trace_id))
        except Exception:
            return None

    def save_analysis(self, a):
        save_json_atomic(self.analysis_path(a["id"]), a)

    def stat(self, trace_id):
        return os.stat(self.trace_path(trace_id))

    def fingerprint(self):
        """Changes when a trace is added or rewritten."""
        ids = self.list_ids()
        mt = 0.0
        for tid in ids:
            try:
                mt = max(mt, os.stat(self.trace_path(tid)).st_mtime)
            except OSError:
                pass
        return f"{len(ids)}-{mt:.3f}"

    def load_conversion(self):
        try:
            return load_json(os.path.join(self.trace_dir, CONVERSION_FILE))
        except Exception:
            return None

    def save_conversion(self, info):
        self.ensure()
        save_json_atomic(os.path.join(self.trace_dir, CONVERSION_FILE), info, indent=1)


class RawCache:
    """Token counts of raw (untemplated) texts, keyed by sha256, persisted in the trace dir."""

    def __init__(self, store, tokenizer):
        self.path = os.path.join(store.trace_dir, RAWCACHE_FILE)
        self.tokenizer = tokenizer
        self.lock = threading.Lock()
        try:
            self.data = load_json(self.path)
        except Exception:
            self.data = {}
        self.dirty = False

    def count(self, text):
        if text is None:
            return None
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self.lock:
            if h in self.data:
                return self.data[h]
        n = len(self.tokenizer.tokenize(text, pieces=False)) if text else 0
        with self.lock:
            self.data[h] = n
            self.dirty = True
        return n

    def save(self):
        with self.lock:
            if self.dirty:
                try:
                    save_json_atomic(self.path, self.data)
                except OSError:
                    pass
                self.dirty = False
