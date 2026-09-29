from kvprof.tap.httpparser import HttpParser


class Sink:
    def __init__(self, is_response=False):
        self.msgs = []
        self.p = HttpParser(is_response, self.head, self.body, self.end)

    def head(self, h, t):
        self.msgs.append({"head": h, "body": b"", "done": False})

    def body(self, b, t):
        self.msgs[-1]["body"] += b

    def end(self, t):
        self.msgs[-1]["done"] = True


def feed_bytewise(sink, data):
    for i in range(len(data)):
        sink.p.feed(data[i:i + 1], 0.0)


def test_content_length_request():
    s = Sink()
    s.p.feed(b"POST /v1/messages?beta=true HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\nX-A: 1\r\nx-a: 2\r\n\r\nhello", 1.0)
    m = s.msgs[0]
    assert m["done"] and m["body"] == b"hello"
    assert m["head"]["method"] == "POST" and m["head"]["path"] == "/v1/messages" and m["head"]["query"] == "beta=true"
    assert m["head"]["headers"]["x-a"] == "1, 2"


def test_chunked_with_extensions_and_trailers():
    s = Sink(is_response=True)
    data = (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            b"5;ext=1\r\nhello\r\n6\r\n world\r\n0\r\nX-Trailer: t\r\n\r\n")
    feed_bytewise(s, data)
    assert s.msgs[0]["done"] and s.msgs[0]["body"] == b"hello world" and s.msgs[0]["head"]["status"] == 200


def test_pipelining():
    s = Sink()
    s.p.feed(b"GET /props HTTP/1.1\r\n\r\nPOST /a HTTP/1.1\r\nContent-Length: 2\r\n\r\nokGET /b HTTP/1.1\r\n\r\n", 0.0)
    assert [m["head"]["path"] for m in s.msgs] == ["/props", "/a", "/b"]
    assert all(m["done"] for m in s.msgs) and s.msgs[1]["body"] == b"ok"


def test_head_and_204():
    s = Sink(is_response=True)
    s.p.set_no_body(True)
    s.p.feed(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n", 0.0)
    assert s.msgs[0]["done"] and s.msgs[0]["body"] == b""
    s.p.set_no_body(False)
    s.p.feed(b"HTTP/1.1 204 No Content\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nx", 0.0)
    assert len(s.msgs) == 3 and s.msgs[1]["head"]["status"] == 204 and s.msgs[2]["body"] == b"x"


def test_read_until_close():
    s = Sink(is_response=True)
    s.p.feed(b"HTTP/1.1 200 OK\r\n\r\npart1", 0.0)
    s.p.feed(b"part2", 0.0)
    assert not s.msgs[0]["done"]
    s.p.close(1.0)
    assert s.msgs[0]["done"] and s.msgs[0]["body"] == b"part1part2"


def test_split_feeds():
    s = Sink()
    data = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 11\r\n\r\n{\"a\": true}"
    for cut in (3, 30, 50, len(data) - 1):
        s = Sink()
        s.p.feed(data[:cut], 0.0)
        s.p.feed(data[cut:], 0.0)
        assert s.msgs[0]["done"] and s.msgs[0]["body"] == b'{"a": true}'
