from kvprof.backends import get_backend
from kvprof.convert import DEFAULT_VARIANT, anthropic_to_openai, calibrate, load_variant
from kvprof.store import TraceStore

BODY = {"system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
        "tools": [{"name": "Read", "description": "d", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "c1", "name": "Read", "input": {"file_path": "a"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": [{"type": "text", "text": "x"}]},
                                         {"type": "text", "text": "<system-reminder>r</system-reminder>"}]},
        ]}


def test_anthropic_to_openai():
    msgs, tools, origins, flags = anthropic_to_openai(BODY)
    assert msgs[0] == {"role": "system", "content": "A\n\nB"}
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == '{"file_path":"a"}'
    assert msgs[3]["role"] == "tool" and msgs[4]["role"] == "user" and origins[4] == {"msg": 2, "blocks": [1]}
    assert tools[0]["function"]["parameters"] == {"type": "object"}
    assert flags == {"unsupported_blocks": 0, "thinking_blocks": 0}


def test_variants():
    msgs, _, origins, _ = anthropic_to_openai(BODY, {"tool_first": False, "system_join": ""})
    assert msgs[0]["content"] == "AB"
    assert msgs[3]["role"] == "user" and msgs[4]["role"] == "tool"


def test_thinking_and_images():
    body = {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}, {"type": "text", "text": "what"}]},
                         {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "a"}]}]}
    msgs, _, _, flags = anthropic_to_openai(body)
    assert msgs[0]["content"] == "[image]\nwhat" and msgs[1]["reasoning_content"] == "hm"
    assert flags == {"unsupported_blocks": 1, "thinking_blocks": 1}


def test_calibrate_against_fake(fake_upstream, tmp_path):
    backend = get_backend(upstream=fake_upstream.url)
    store = TraceStore(str(tmp_path / "t"))
    v = calibrate(backend, store, "test-model", verbose=False)
    assert v == dict(DEFAULT_VARIANT, tool_result_join="")
    info = store.load_conversion()
    assert info["model_path"] == "/models/test-model.gguf" and info["variant"] == v
    n_before = len(fake_upstream.requests)
    assert load_variant(backend, store, "test-model") == v
    assert len(fake_upstream.requests) == n_before  # cached: no new probes
