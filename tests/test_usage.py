import json

from kvprof.sse import parse_sse_lines
from kvprof.usage import StreamWatcher, extract_raw_usage, upgrade_schema1_usage


def anth_lines(output_tokens=True):
    ev = [("message_start", {"type": "message_start", "message": {"model": "m", "usage": {"input_tokens": 10, "cache_read_input_tokens": 90}}}),
          ("content_block_delta", {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "a"}}),
          ("content_block_delta", {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{"}}),
          ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7} if output_tokens else {}})]
    out = []
    for name, d in ev:
        out += [f"event: {name}", "data: " + json.dumps(d), ""]
    return out


def test_anthropic_stream():
    u = extract_raw_usage("anthropic", parse_sse_lines(anth_lines()))
    assert u["api"] == "anthropic" and u["fields"] == {"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 7}
    assert (u["stop_reason"], u["model"], u["n_deltas"], u["n_text_deltas"], u["output_tokens_estimated"]) == ("end_turn", "m", 2, 1, False)


def test_anthropic_stream_estimated_output():
    u = extract_raw_usage("anthropic", parse_sse_lines(anth_lines(output_tokens=False)))
    assert u["fields"]["output_tokens"] == 2 and u["output_tokens_estimated"]


def test_anthropic_json():
    u = extract_raw_usage("anthropic", body_text=json.dumps({"model": "m", "stop_reason": "max_tokens", "usage": {"input_tokens": 3, "output_tokens": 4}}))
    assert u["fields"] == {"input_tokens": 3, "output_tokens": 4} and u["stop_reason"] == "max_tokens"
    assert extract_raw_usage("anthropic", body_text='{"error": {"message": "x"}}') is None
    assert extract_raw_usage("anthropic", body_text="not json") is None
    assert extract_raw_usage("anthropic", body_text=None) is None


def oai_lines(with_usage=True):
    chunks = [{"model": "m", "choices": [{"delta": {"role": "assistant", "content": None}}]},
              {"choices": [{"delta": {"reasoning_content": "think"}}]},
              {"choices": [{"delta": {"content": "hi"}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "f"}}]}}]},
              dict({"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "timings": {"prompt_n": 5, "cache_n": 95}},
                   **({"usage": {"prompt_tokens": 100, "completion_tokens": 3}} if with_usage else {}))]
    out = []
    for c in chunks:
        out += ["data: " + json.dumps(c), ""]
    return out + ["data: [DONE]", ""]


def test_openai_stream():
    u = extract_raw_usage("openai", parse_sse_lines(oai_lines()))
    assert u["fields"] == {"prompt_tokens": 100, "completion_tokens": 3} and u["timings"] == {"prompt_n": 5, "cache_n": 95}
    assert (u["stop_reason"], u["model"], u["n_deltas"], u["n_text_deltas"], u["output_tokens_estimated"]) == ("tool_calls", "m", 3, 1, False)


def test_openai_stream_without_usage():
    u = extract_raw_usage("openai", parse_sse_lines(oai_lines(with_usage=False)))
    assert u["fields"] == {"completion_tokens": 3} and u["output_tokens_estimated"] and u["timings"]["cache_n"] == 95


def test_openai_json():
    u = extract_raw_usage("openai", body_text=json.dumps({"model": "m", "choices": [{"finish_reason": "stop"}],
                                                          "usage": {"prompt_tokens": 9}, "timings": {"prompt_n": 9}}))
    assert u["fields"] == {"prompt_tokens": 9} and u["stop_reason"] == "stop" and u["timings"] == {"prompt_n": 9}


def test_schema1_upgrade():
    old = {"input_tokens": 5, "cache_read_input_tokens": 95, "cache_creation_input_tokens": None, "output_tokens": 3,
           "n_deltas": 3, "n_text_deltas": 3, "output_tokens_estimated": False}
    u = upgrade_schema1_usage(old, "end_turn")
    assert u["api"] == "anthropic" and u["fields"] == {"input_tokens": 5, "cache_read_input_tokens": 95, "output_tokens": 3}
    assert u["stop_reason"] == "end_turn" and u["n_deltas"] == 3
    assert upgrade_schema1_usage(u) is u and upgrade_schema1_usage(None) is None


def test_stream_watcher():
    w = StreamWatcher("anthropic")
    w.observe("message_start", {}, 1.0)
    w.observe("content_block_delta", {"delta": {"type": "input_json_delta"}}, 2.0)
    w.observe("content_block_delta", {"delta": {"type": "text_delta"}}, 3.0)
    assert w.timing() == {"t_first_event": 1.0, "t_message_start": 1.0, "t_first_delta": 2.0, "t_first_text_delta": 3.0}
    w = StreamWatcher("openai")
    w.observe(None, {"choices": [{"delta": {"role": "assistant", "content": ""}}]}, 1.0)
    w.observe(None, {"choices": [{"delta": {"reasoning_content": "x"}}]}, 2.0)
    w.observe(None, {"choices": [{"delta": {"content": "y"}}]}, 3.0)
    w.observe(None, {"_raw": "[DONE]"}, 4.0)
    assert (w.t_first_event, w.t_first_delta, w.t_first_text_delta) == (1.0, 2.0, 3.0)
