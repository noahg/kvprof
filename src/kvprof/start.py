"""start: the tap and the GUI in one process.

The dashboard comes up at once. The analyzer attaches when llama-server answers and re-attaches when the server
comes back with another model, so the order in which things are started does not matter.
"""
import errno
import json
import os
import signal
import sys
import threading
import urllib.request
import webbrowser

from . import PACKAGE_NAME, __version__
from .analyze import Pipeline, UpstreamUnavailable
from .backends import get_backend
from .serve import Live, make_server, scan_once
from .store import model_dir
from .tap import run_tap
from .util import display_path, load_json, log, parse_addr, save_json_atomic

EXIT_TAP_PORT = 3   # the tap's address is taken; clients are configured with it, so there is no fallback
PROBE_EVERY = 5     # scans between checks of an upstream that answers (one that does not is checked every scan)


def run_file(data_dir, tap_port):
    """One file per running `start`, named by the tap port its clients use."""
    return os.path.join(data_dir, f"run-{tap_port}.json")


def find_running(settings):
    """The status of a `start` already serving this tap address, or None."""
    port = parse_addr(settings.listen, 8081)[1]
    if not settings.data_dir or not port:
        return None
    try:
        run = load_json(run_file(settings.data_dir, port))
        with urllib.request.urlopen(run["gui_url"] + "api/status", timeout=2) as r:
            st = json.load(r)
    except Exception:
        return None
    return st if st.get("package") == PACKAGE_NAME and st.get("pid") == run.get("pid") else None


def _url(host, port):
    return f"http://{'127.0.0.1' if host in ('', '0.0.0.0', '::') else host}:{port}"


def _server_info(settings, timeout=5):
    info = get_backend(settings.backend, upstream=settings.upstream, api_key=settings.api_key, timeout=timeout).server_info()
    if "props_error" in info.extra and not info.model_path:
        raise UpstreamUnavailable(f"cannot reach {settings.upstream}: {info.extra['props_error']}")
    return info


def _locator(live):
    """Per-model trace root: where a new tap connection records."""
    def locate():
        p = live.pipeline
        if p is not None and live.upstream_ok:
            return p.store.trace_dir
        # not attached yet, but the relay has just connected, so the server answers
        return model_dir(live.settings.trace_dir, _server_info(live.settings).model_path)
    return locate


def supervise(live, stop_event, interval=1.0):
    """Attach the analyzer when the upstream answers, re-attach when its model changed, rescan in between."""
    s = live.settings
    state, n = None, 0
    while not stop_event.is_set():
        if state is not True or n % PROBE_EVERY == 0:
            try:
                p = live.pipeline
                if p is None or _server_info(s).model_path != p.model_path:
                    live.pipeline = p = Pipeline(s, verbose=False)
                    log(f"{PACKAGE_NAME}: analyzing {p.info.model_name} at {s.upstream}; traces in {display_path(p.store.trace_dir)}")
                elif state is not True:
                    p.last_fingerprint = None   # analyze again what could not be analyzed while the server was away
                live.upstream_ok, live.upstream_error = True, None
            except Exception as e:
                live.upstream_ok, live.upstream_error = False, str(e) if isinstance(e, UpstreamUnavailable) else repr(e)
                if state is not False:
                    log(f"{PACKAGE_NAME}: waiting for llama-server at {s.upstream} ({live.upstream_error})")
            state = live.upstream_ok
        n += 1
        if state:
            try:
                scan_once(live.pipeline)
            except Exception as e:
                print(f"scan error: {e!r}", file=sys.stderr)
        stop_event.wait(interval)


def run_start(settings, open_browser=True, exit_with_parent=False, stop_event=None, ready_event=None, info=None):
    """Tap and GUI until stop_event is set (or Ctrl-C, SIGTERM, the parent's exit). -> exit code."""
    s = settings
    running = find_running(s)
    if running:
        log(f"{PACKAGE_NAME}: already running: dashboard at {running['gui_url']}, tap at {running['tap_url']}")
        if open_browser:
            webbrowser.open(running["gui_url"])
        return 0
    stop = stop_event or threading.Event()
    info = {} if info is None else info
    live = Live(s)
    tap_ready, tap_failed = threading.Event(), []

    def tap():
        try:
            run_tap(s, ready_event=tap_ready, stop_event=stop, info=info, locate=_locator(live) if s.per_model else None)
        except OSError as e:
            tap_failed.append(e)
            tap_ready.set()

    tap_thread = threading.Thread(target=tap, daemon=True)
    tap_thread.start()
    tap_ready.wait()
    if tap_failed:
        busy = tap_failed[0].errno == errno.EADDRINUSE
        hint = "\n  (another program has the tap's port; pass --listen HOST:PORT and point your client there)" if busy else ""
        print(f"{PACKAGE_NAME}: cannot start the tap on {s.listen}: {tap_failed[0]}{hint}", file=sys.stderr)
        return EXIT_TAP_PORT if busy else 1
    live.tap_url = _url(info["host"], info["port"])
    try:
        httpd = make_server(live, s.host, s.port)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            stop.set()
            raise
        httpd = make_server(live, s.host, 0)   # the dashboard can be anywhere: nothing is configured with its address
    live.gui_url = _url(s.host, httpd.server_address[1]) + "/"
    threads = [tap_thread, threading.Thread(target=httpd.serve_forever, daemon=True),
               threading.Thread(target=supervise, args=(live, stop), daemon=True)]
    for t in threads[1:]:
        t.start()
    run_path = None
    if s.data_dir:
        run_path = run_file(s.data_dir, info["port"])
        try:
            os.makedirs(s.data_dir, exist_ok=True)
            save_json_atomic(run_path, {"pid": os.getpid(), "version": __version__, "gui_url": live.gui_url, "tap_url": live.tap_url,
                                        "upstream": s.upstream, "trace_dir": os.path.abspath(s.trace_dir)}, indent=1)
        except OSError:
            run_path = None
    log(f"{PACKAGE_NAME}: dashboard at {live.gui_url}   (Ctrl-C to stop)")
    info.update(gui_url=live.gui_url, tap_url=live.tap_url, live=live)
    if ready_event:
        ready_event.set()
    if open_browser:
        webbrowser.open(live.gui_url)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, signal.default_int_handler)
    parent = os.getppid()
    try:
        while not stop.wait(0.5):
            if exit_with_parent and os.getppid() != parent:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.shutdown()
        httpd.server_close()
        if run_path:
            try:
                os.unlink(run_path)
            except OSError:
                pass
        for t in threads:
            t.join(2)
        log(f"{PACKAGE_NAME}: stopped")
    return 0
