"""llama-server's cache layout: one KV cache per slot (plus an optional host-memory prompt cache)."""
from ..base import CacheTopology


class SlotTopology(CacheTopology):
    def lanes(self, info):
        n = info.n_slots or 1
        return [f"slot {i}" for i in range(n)] + ["no log"]
