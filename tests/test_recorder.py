import json
import os
import socket
import time

from conftest import post


def traces(d):
    out = []
    for n in sorted(os.listdir(d)):
        if n.endswith(".json") and not n.startswith("."):
            with open(os.path.join(d, n)) as f:
                out.append(json.load(f))
    return out


def wait_for(pred, timeout=5.0):
    t = time.time() + timeout
    while time.time() < t:
        if pred():
            return True
        time.sleep(0.02)
    return False


BODY = {"model": "test-model", "max_tokens": 64, "system": "You are terse.", "messages": [{"role": "user", "content": "héllo €"}]}


def test_anthropic_stream_recorded(tap_proc):
    post(tap_proc["url"] + "/v1/messages", dict(BODY, stream=True), headers={"authorization": "Bearer secret", "x-api-key": "k"}, raw=True)
    assert wait_for(lambda: len(traces(tap_proc["trace_dir"])) == 1 and traces(tap_proc["trace_dir"])[0]["outcome"] == "ok")
    tr = traces(tap_proc["trace_dir"])[0]
    assert tr["schema"] == 2 and tr["api"] == "anthropic" and tr["seq"] == 1 and tr["stream"] is True
    assert "authorization" not in tr["request"]["headers"] and "x-api-key" not in tr["request"]["headers"]
    assert tr["request"]["body"] == dict(BODY, stream=True)
    assert tr["response"]["status"] == 200 and tr["response"]["complete"] and tr["response"]["sse_lines"][0] == "event: message_start"
    assert tr["usage"]["fields"]["input_tokens"] > 0 and tr["usage"]["fields"]["output_tokens"] > 0 and tr["stop_reason"] == "end_turn"
    tm = tr["timing"]
    assert tm["t_request"] <= tm["t_first_event"] <= tm["t_first_delta"] <= tm["t_last_byte"] and tm["t_message_start"] == tm["t_first_event"]
    assert tm["ttft_s"] is not None and tm["total_s"] >= tm["ttft_s"]


def test_openai_nonstream_and_passthrough(tap_proc, fake_upstream):
    body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    r = post(tap_proc["url"] + "/v1/chat/completions", body)
    assert r["choices"][0]["message"]["content"]
    post(tap_proc["url"] + "/tokenize", {"content": "not recorded"})
    assert wait_for(lambda: len(traces(tap_proc["trace_dir"])) == 1 and traces(tap_proc["trace_dir"])[0]["outcome"] == "ok")
    tr = traces(tap_proc["trace_dir"])[0]
    assert tr["api"] == "openai" and tr["response"]["sse_lines"] is None and json.loads(tr["response"]["body_text"])["usage"]
    assert tr["usage"]["fields"]["prompt_tokens"] > 0 and tr["usage"]["timings"]["prompt_n"] > 0 and tr["stop_reason"] == "stop"
    assert any(p == "/tokenize" for p, _, _ in fake_upstream.requests)


def test_bytes_unchanged(tap_proc, fake_upstream):
    direct = post(fake_upstream.url + "/tokenize", {"content": "héllo € world", "with_pieces": True}, raw=True)
    via = post(tap_proc["url"] + "/tokenize", {"content": "héllo € world", "with_pieces": True}, raw=True)
    assert direct == via


def test_http_error(tap_proc):
    big = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "word " * 5000}]}
    try:
        post(tap_proc["url"] + "/v1/messages", big)
    except Exception:
        pass
    assert wait_for(lambda: traces(tap_proc["trace_dir"]) and traces(tap_proc["trace_dir"])[0]["outcome"] == "http_error")
    tr = traces(tap_proc["trace_dir"])[0]
    assert tr["response"]["status"] == 400 and tr["usage"] is None


def test_client_disconnect(tap_proc):
    s = socket.create_connection(("127.0.0.1", tap_proc["port"]))
    data = json.dumps(dict(BODY, stream=True)).encode()
    s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nx-fake-delay: 0.3\r\n"
              + f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    buf = b""
    while b"content_block_delta" not in buf:
        buf += s.recv(4096)
    s.close()
    assert wait_for(lambda: traces(tap_proc["trace_dir"]) and traces(tap_proc["trace_dir"])[0]["outcome"] == "client_disconnect")
    tr = traces(tap_proc["trace_dir"])[0]
    assert tr["response"]["complete"] is False and tr["usage"]["fields"]["input_tokens"] > 0


def test_seq_continues_in_existing_dir(tmp_path):
    from kvprof.tap.recorder import TapContext
    (tmp_path / "20260101-000000.000-0007.json").write_text("{}")
    (tmp_path / "20260101-000000.000-0009.analysis.json").write_text("{}")
    assert TapContext(str(tmp_path)).next_seq() == 8
