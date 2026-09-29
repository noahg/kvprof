"""Backend seams. An inference server plugs in by implementing these (llama.cpp is the only one in v1).

  Renderer         messages + tools -> the exact prompt string the server processes
  Tokenizer        text -> tokens with byte pieces (so sections map to token ranges)
  UsageNormalizer  the response's usage block (either wire shape) -> Usage
  TelemetrySource  server-side per-task timings (a log file) and their pairing with recorded requests
  CacheTopology    how the server's prompt cache is laid out (slots) for the timeline lanes
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import NamedTuple, TypedDict


class Token(NamedTuple):
    id: int
    piece: bytes | None


@dataclass(slots=True)
class Usage:
    """Prompt/output accounting for one response, normalized across wire shapes.

    total = prompt tokens; cached = reused from the server's cache; uncached = prefilled for this request.
    semantics: how the raw prompt count relates to the rendered prompt ("total": the raw count is the whole
    prompt, "uncached": it excludes the cached part, None: no usage)."""
    total: int | None = None
    cached: int | None = None
    uncached: int | None = None
    output_tokens: int | None = None
    output_estimated: bool = False
    prefill_ms: float | None = None
    decode_ms: float | None = None
    decode_rate: float | None = None
    stop_reason: str | None = None
    model: str | None = None
    api: str | None = None
    semantics: str | None = None
    raw_prompt: int | None = None  # the prompt count exactly as the response reported it

    def to_dict(self):
        return asdict(self)


@dataclass(slots=True)
class ServerInfo:
    backend: str
    build: str | None = None
    model_id: str | None = None
    model_alias: str | None = None
    model_path: str | None = None
    n_ctx: int | None = None
    n_slots: int | None = None
    chat_template: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def model_name(self):
        import os
        return os.path.basename(self.model_path) if self.model_path else (self.model_id or self.model_alias or "?")


class LogTask(TypedDict, total=False):
    """One server-side task from the telemetry source (a dict, so analyses can store it as JSON)."""
    task_id: int
    slot: int
    t_launch: float
    t_pp_done: float | None
    t_tg_done: float | None
    t_release: float | None
    select: dict | None          # {"by": "LCP", "f_sim", "f_keep"} | {"by": "LRU", "t_last"}
    pp_samples: list             # [t, n_tokens, progress, t_elapsed, rate]
    pp_ms: float | None
    pp_tokens: int | None
    pp_rate: float | None
    tg_ms: float | None
    tg_tokens: int | None
    tg_rate: float | None
    n_tokens_final: int | None
    truncated: int | None
    cancelled: bool
    t_cancel: float | None
    evictions_mib: list
    prev_slot_n_tokens: int | None
    reused_from_log: int | None
    segment: int
    is_child: int
    line: int | None


class Renderer(ABC):
    @abstractmethod
    def render(self, messages, tools=None) -> str: ...


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text, pieces=True) -> list[Token]: ...

    def count(self, text) -> int:
        return len(self.tokenize(text, pieces=False)) if text else 0


class UsageNormalizer(ABC):
    @abstractmethod
    def normalize(self, raw, n_rendered) -> Usage: ...


class TelemetrySource(ABC):
    """Server-side task timings. `load` -> {"tasks": [LogTask], "last_ts", "segments", ...};
    `anchor` -> (epoch of telemetry time 0, source); `pair` attaches tasks to analyses in place."""

    @abstractmethod
    def load(self): ...

    @abstractmethod
    def anchor(self, logdata, override=None): ...

    @abstractmethod
    def pair(self, analyses, logdata, anchor): ...

    def fingerprint(self):
        return ""


class CacheTopology(ABC):
    @abstractmethod
    def lanes(self, info: ServerInfo) -> list[str]: ...


class Backend(ABC):
    name: str
    renderer: Renderer
    tokenizer: Tokenizer
    usage: UsageNormalizer
    topology: CacheTopology

    @abstractmethod
    def server_info(self, refresh=False) -> ServerInfo: ...

    @abstractmethod
    def telemetry(self, log_path=None, pid_file=None) -> TelemetrySource | None: ...
