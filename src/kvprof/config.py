"""Settings: defaults, environment overrides (KVPROF_*), command-line flags. Flags win over env, env over defaults."""
import os
import sys
from dataclasses import dataclass, field, fields

from . import PACKAGE_NAME

ENV_PREFIX = PACKAGE_NAME.upper() + "_"
DEFAULT_UPSTREAM = "http://127.0.0.1:8080"   # llama-server's default port
DEFAULT_LISTEN = "127.0.0.1:8081"
DEFAULT_GUI_HOST = "127.0.0.1"
DEFAULT_GUI_PORT = 8084
DEFAULT_TRACE_DIR = f".{PACKAGE_NAME}"
DEFAULT_RECORD_PATHS = ("/v1/messages", "/v1/chat/completions")


@dataclass
class Settings:
    upstream: str = DEFAULT_UPSTREAM
    listen: str = DEFAULT_LISTEN
    trace_dir: str = DEFAULT_TRACE_DIR
    record_paths: tuple = DEFAULT_RECORD_PATHS
    server_log: str | None = None
    server_pid: str | None = None
    log_anchor: float | None = None
    use_log: bool = True
    model: str | None = None
    api_key: str | None = None
    backend: str = "llamacpp"
    template_profile: str = "auto"
    host: str = DEFAULT_GUI_HOST
    port: int = DEFAULT_GUI_PORT
    quiet: bool = False
    refresh: bool = False
    data_dir: str | None = None   # per-user directory: the default trace root and `start`'s run files
    per_model: bool = False       # trace_dir is a root with one subdirectory per model (see store.model_dir)
    extra: dict = field(default_factory=dict)


# Settings that can come from the environment: KVPROF_UPSTREAM, KVPROF_TRACE_DIR, KVPROF_SERVER_LOG, ...
ENV_KEYS = ("upstream", "listen", "trace_dir", "server_log", "server_pid", "model", "api_key", "template_profile", "data_dir")


def env_name(key):
    return ENV_PREFIX + key.upper()


def default_data_dir(environ=None):
    """Where kvprof keeps a user's traces when no directory was chosen."""
    environ = os.environ if environ is None else environ
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    elif os.name == "nt":
        base = environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
    else:
        base = environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")
    return os.path.join(base, PACKAGE_NAME)


def resolve(args=None, environ=None):
    """Settings from an argparse namespace (None-valued attributes fall through to env, then defaults)."""
    environ = os.environ if environ is None else environ
    s = Settings()
    for key in ENV_KEYS:
        v = environ.get(env_name(key))
        if v:
            setattr(s, key, v)
    if args is not None:
        names = {f.name for f in fields(Settings)}
        for k, v in vars(args).items():
            if k in names and v is not None:
                setattr(s, k, v)
    s.data_dir = s.data_dir or default_data_dir(environ)
    # No trace directory chosen: ./.kvprof when there is one, else the per-user root, split by model. `tap` alone
    # cannot ask which model the server runs, so it keeps recording into ./.kvprof.
    chosen = environ.get(env_name("trace_dir")) or getattr(args, "trace_dir", None)
    if not chosen and getattr(args, "cmd", None) != "tap" and not os.path.isdir(DEFAULT_TRACE_DIR):
        s.trace_dir, s.per_model = os.path.join(s.data_dir, "traces"), True
    if isinstance(s.record_paths, str):
        s.record_paths = tuple(p.strip() for p in s.record_paths.split(",") if p.strip())
    if s.log_anchor is not None:
        s.log_anchor = float(s.log_anchor)
    if not s.server_log:
        s.use_log = False
    return s
