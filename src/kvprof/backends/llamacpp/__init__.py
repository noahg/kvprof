"""llama.cpp's llama-server: the one backend that exposes rendering, tokenization with pieces, per-request
cached-token usage, server facts and a per-task log over HTTP and a file."""
import os

from ..base import Backend, TelemetrySource
from .client import LlamaCppClient
from .pairing import pair_with_log
from .serverlog import log_anchor, parse_server_log
from .topology import SlotTopology
from .usage import LlamaCppUsageNormalizer


class LlamaCppLog(TelemetrySource):
    def __init__(self, log_path, pid_file=None):
        self.log_path, self.pid_file = log_path, pid_file

    def load(self):
        return parse_server_log(self.log_path)

    def anchor(self, logdata, override=None):
        return log_anchor(self.pid_file, self.log_path, logdata.get("last_ts"), override)

    def pair(self, analyses, logdata, anchor):
        return pair_with_log(analyses, logdata, anchor)

    def fingerprint(self):
        try:
            return str(os.stat(self.log_path).st_size)
        except (OSError, TypeError):
            return "0"


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(self, upstream, api_key=None, timeout=300):
        self.client = LlamaCppClient(upstream, api_key=api_key, timeout=timeout)
        self.renderer = self.client
        self.tokenizer = self.client
        self.usage = LlamaCppUsageNormalizer()
        self.topology = SlotTopology()
        self._info = None

    def server_info(self, refresh=False):
        if self._info is None or refresh:
            self._info = self.client.server_info()
        return self._info

    def telemetry(self, log_path=None, pid_file=None):
        return LlamaCppLog(log_path, pid_file) if log_path else None

    def default_model(self):
        """Model name for requests the analyzer sends itself: /v1/models id, else /props model_alias."""
        info = self.server_info()
        return info.model_id or info.model_alias
