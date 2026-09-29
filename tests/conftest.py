import json
import os
import threading
import urllib.request

import pytest

from fake_upstream import FakeUpstream

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def fake_upstream(tmp_path):
    f = FakeUpstream(str(tmp_path / "server.log"), template="chatml")
    yield f
    f.close()


@pytest.fixture
def fake_upstream_llama3(tmp_path):
    f = FakeUpstream(str(tmp_path / "server-llama3.log"), template="llama3")
    yield f
    f.close()


@pytest.fixture
def tap_proc(fake_upstream, tmp_path):
    """The tap, in-process on a free port, relaying to fake_upstream and recording into tmp_path/traces."""
    from kvprof.config import Settings
    from kvprof.tap import run_tap
    trace_dir = tmp_path / "traces"
    s = Settings(upstream=fake_upstream.url, listen="127.0.0.1:0", trace_dir=str(trace_dir), quiet=True)
    ready, stop = threading.Event(), threading.Event()
    info = {}
    th = threading.Thread(target=run_tap, args=(s,), kwargs={"ready_event": ready, "stop_event": stop, "info": info}, daemon=True)
    th.start()
    assert ready.wait(5), "tap did not start"
    info["trace_dir"] = str(trace_dir)
    info["url"] = f"http://127.0.0.1:{info['port']}"
    yield info
    stop.set()
    th.join(5)


def post(url, body, headers=None, raw=False):
    h = {"content-type": "application/json", "user-agent": "claude-cli/2.1.283 (external, cli)"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    return data if raw else json.loads(data)


def load_fixture_trace(name):
    with open(os.path.join(FIXTURES, "traces", name + ".json"), encoding="utf-8") as f:
        return json.load(f)
