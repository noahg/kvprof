from kvprof.sse import SSEParser, parse_sse_lines


def test_anthropic_events():
    p = SSEParser()
    evs = p.feed_line("event: message_start") + p.feed_line('data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}') + p.feed_line("")
    assert evs and evs[0][0] == "message_start" and evs[0][1]["message"]["usage"]["input_tokens"] == 5


def test_openai_chunks_and_done():
    evs = parse_sse_lines(['data: {"choices":[{"delta":{"content":"hi"}}]}', "", "data: [DONE]", ""])
    assert evs[0] == (None, {"choices": [{"delta": {"content": "hi"}}]})
    assert evs[1] == (None, {"_raw": "[DONE]"})


def test_multiline_data_and_flush():
    evs = parse_sse_lines(["event: x", 'data: {"a":', "data: 1}"])
    assert evs == [("x", {"a": 1})]


def test_type_from_data():
    assert parse_sse_lines(['data: {"type":"ping"}', ""])[0][0] == "ping"
