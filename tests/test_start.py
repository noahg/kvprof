"""start: the tap, the GUI and the analyzer's supervisor in one process."""
import contextlib
import json
import os
import socket
import threading
import time
import urllib.request

import pytest

from conftest import post
from fake_upstream import FakeUpstream
from kvprof import start as start_mod
from kvprof.config import Settings
from kvprof.start import EXIT_TAP_PORT, run_file, run_start
from kvprof.store import model_dir

BODY = {"model": "test-model", "max_tokens": 64, "messages": [{"role": "user", "content": [{"type": "text", "text": "hello there"}]}]}


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def _until(fn, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError("condition not reached")


def _settings(tmp_path, upstream, **kw):
    kw.setdefault("trace_dir", str(tmp_path / "traces"))
    return Settings(upstream=upstream, listen="127.0.0.1:0", port=0, quiet=True, use_log=False, data_dir=str(tmp_path / "data"), **kw)


@contextlib.contextmanager
def _started(settings):
    ready, stop, info, rc = threading.Event(), threading.Event(), {}, []
    th = threading.Thread(target=lambda: rc.append(run_start(settings, open_browser=False, stop_event=stop, ready_event=ready, info=info)), daemon=True)
    th.start()
    assert ready.wait(10), "start did not come up"
    try:
        yield info
    finally:
        stop.set()
        th.join(10)
    assert rc == [0]


@pytest.fixture(autouse=True)
def fast_probe(monkeypatch):
    monkeypatch.setattr(start_mod, "PROBE_EVERY", 1)


def test_records_and_serves(fake_upstream, tmp_path):
    s = _settings(tmp_path, fake_upstream.url)
    with _started(s) as info:
        st = _until(lambda: (lambda d: d if d["ready"] else None)(_get(info["gui_url"] + "api/status")))
        assert st["upstream_ok"] and st["model"] == "test-model.gguf" and st["tap_url"] == info["tap_url"] and st["pid"] == os.getpid()
        assert _get(info["gui_url"] + "api/requests")["requests"] == []
        post(info["tap_url"] + "/v1/messages", BODY)
        d = _until(lambda: (lambda d: d if d["requests"] and d["requests"][0]["outcome"] == "ok" else None)(_get(info["gui_url"] + "api/requests")))
        assert len(d["requests"]) == 1 and d["tap_url"] == info["tap_url"] and d["n_slots"] == 2
        run = run_file(s.data_dir, int(info["tap_url"].rsplit(":", 1)[1]))
        with open(run) as f:
            assert json.load(f)["gui_url"] == info["gui_url"]
    assert not os.path.exists(run)
    with pytest.raises(OSError):
        urllib.request.urlopen(info["gui_url"] + "api/status", timeout=2)


def test_waits_for_upstream(tmp_path):
    port = _free_port()
    s = _settings(tmp_path, f"http://127.0.0.1:{port}")
    with _started(s) as info:
        st = _until(lambda: (lambda d: d if d["upstream_error"] else None)(_get(info["gui_url"] + "api/status")))
        assert not st["ready"] and not st["upstream_ok"] and "cannot reach" in st["upstream_error"]
        d = _get(info["gui_url"] + "api/requests")
        assert d["requests"] == [] and d["ready"] is False and d["upstream"] == s.upstream
        fake = FakeUpstream(str(tmp_path / "server.log"), port=port)
        try:
            st = _until(lambda: (lambda d: d if d["ready"] else None)(_get(info["gui_url"] + "api/status")))
            assert st["upstream_ok"] and st["upstream_error"] is None
            post(info["tap_url"] + "/v1/messages", BODY)
            _until(lambda: _get(info["gui_url"] + "api/requests")["requests"])
        finally:
            fake.close()
        # the server goes away: what was analyzed stays, the status says so
        st = _until(lambda: (lambda d: d if not d["upstream_ok"] else None)(_get(info["gui_url"] + "api/status")))
        assert st["ready"] and len(_get(info["gui_url"] + "api/requests")["requests"]) == 1


def test_per_model_root_follows_the_model(fake_upstream, tmp_path):
    root = str(tmp_path / "root")
    s = _settings(tmp_path, fake_upstream.url, trace_dir=root, per_model=True)
    with _started(s) as info:
        # a request that arrives before the analyzer attached still lands in its model's directory
        post(info["tap_url"] + "/v1/messages", BODY)
        first = model_dir(root, "/models/test-model.gguf")
        d = _until(lambda: (lambda d: d if d["requests"] else None)(_get(info["gui_url"] + "api/requests")))
        assert d["trace_path"] == first and len([n for n in os.listdir(first) if not n.startswith(".") and not n.endswith(".analysis.json")]) == 1
        fake_upstream.model_path = "/models/other.gguf"
        _until(lambda: _get(info["gui_url"] + "api/status")["model"] == "other.gguf")
        assert _get(info["gui_url"] + "api/requests")["requests"] == []
        post(info["tap_url"] + "/v1/messages", BODY)
        d = _until(lambda: (lambda d: d if d["requests"] else None)(_get(info["gui_url"] + "api/requests")))
        assert d["trace_path"] == model_dir(root, "/models/other.gguf") and len(d["requests"]) == 1
    assert not [n for n in os.listdir(root) if n.endswith(".json")]


def test_second_instance_and_busy_port(fake_upstream, tmp_path, capsys):
    port = _free_port()
    s = _settings(tmp_path, fake_upstream.url)
    s.listen = f"127.0.0.1:{port}"
    with _started(s) as info:
        assert run_start(s, open_browser=False) == 0
        assert f"already running: dashboard at {info['gui_url']}" in capsys.readouterr().out
        # same port, but not an instance this data directory knows: the tap's address has no fallback
        other = _settings(tmp_path, fake_upstream.url)
        other.listen, other.data_dir = s.listen, str(tmp_path / "elsewhere")
        assert run_start(other, open_browser=False) == EXIT_TAP_PORT
        assert "cannot start the tap" in capsys.readouterr().err


def test_gui_port_falls_back(fake_upstream, tmp_path):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        s = _settings(tmp_path, fake_upstream.url)
        s.port = busy.getsockname()[1]
        with _started(s) as info:
            assert not info["gui_url"].endswith(f":{s.port}/") and _get(info["gui_url"] + "api/status")["package"] == "kvprof"
