"""Passive incremental HTTP/1.1 parser: watches one direction of one relayed connection."""


class HttpParser:
    """Incremental parser for a sequence of HTTP/1.1 messages on one direction of one connection.

    Callbacks: on_head(head_dict, t), on_body(bytes, t) with de-chunked body bytes, on_end(t)."""

    def __init__(self, is_response, on_head, on_body, on_end):
        self.is_response = is_response
        self.on_head, self.on_body, self.on_end = on_head, on_body, on_end
        self.buf = b""
        self.state = "head"
        self.remaining = 0
        self.chunk_state = "size"
        self.head = None
        self.no_body_expected = False

    def set_no_body(self, flag):
        self.no_body_expected = flag

    def feed(self, data, t):
        self.buf += data
        while True:
            if self.state == "head":
                i = self.buf.find(b"\r\n\r\n")
                if i < 0:
                    return
                raw, self.buf = self.buf[:i], self.buf[i + 4:]
                self.head = self._parse_head(raw)
                self.on_head(self.head, t)
                self._pick_body_mode()
                if self.state == "done":
                    self._finish(t)
                    continue
            elif self.state == "length":
                if not self.buf:
                    return
                take = self.buf[:self.remaining]
                self.buf = self.buf[len(take):]
                self.remaining -= len(take)
                if take:
                    self.on_body(take, t)
                if self.remaining == 0:
                    self._finish(t)
                    continue
                return
            elif self.state == "chunked":
                progressed = self._feed_chunked(t)
                if self.state == "done":
                    self._finish(t)
                    continue
                if not progressed:
                    return
            elif self.state == "close":
                if self.buf:
                    self.on_body(self.buf, t)
                    self.buf = b""
                return
            else:
                return

    def _parse_head(self, raw):
        lines = raw.decode("latin-1").split("\r\n")
        first = lines[0]
        headers = {}
        for ln in lines[1:]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                k, v = k.strip().lower(), v.strip()
                headers[k] = headers[k] + ", " + v if k in headers else v
        head = {"headers": headers}
        parts = first.split(" ", 2)
        if self.is_response:
            head["version"] = parts[0]
            head["status"] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        else:
            head["method"] = parts[0]
            target = parts[1] if len(parts) > 1 else "/"
            path, _, query = target.partition("?")
            head["path"], head["query"], head["version"] = path, query, parts[2] if len(parts) > 2 else ""
        return head

    def _pick_body_mode(self):
        h = self.head["headers"]
        if self.is_response:
            st = self.head["status"]
            if self.no_body_expected or st < 200 or st in (204, 304):
                self.state = "done"
                return
        te = h.get("transfer-encoding", "").lower()
        if "chunked" in te:
            self.state, self.chunk_state = "chunked", "size"
        elif "content-length" in h:
            try:
                self.remaining = int(h["content-length"].split(",")[0].strip())
            except ValueError:
                self.remaining = 0
            self.state = "length" if self.remaining > 0 else "done"
        else:
            self.state = "close" if self.is_response else "done"

    def _feed_chunked(self, t):
        progressed = False
        while True:
            if self.chunk_state == "size":
                i = self.buf.find(b"\r\n")
                if i < 0:
                    return progressed
                line = self.buf[:i].decode("latin-1").split(";")[0].strip()
                self.buf = self.buf[i + 2:]
                progressed = True
                try:
                    self.remaining = int(line, 16) if line else 0
                except ValueError:
                    self.remaining = 0
                self.chunk_state = "trailer" if self.remaining == 0 else "data"
            elif self.chunk_state == "data":
                if not self.buf:
                    return progressed
                take = self.buf[:self.remaining]
                self.buf = self.buf[len(take):]
                self.remaining -= len(take)
                progressed = True
                if take:
                    self.on_body(take, t)
                if self.remaining == 0:
                    self.chunk_state = "crlf"
            elif self.chunk_state == "crlf":
                if len(self.buf) < 2:
                    return progressed
                self.buf = self.buf[2:]
                self.chunk_state = "size"
                progressed = True
            elif self.chunk_state == "trailer":
                i = self.buf.find(b"\r\n")
                if i < 0:
                    return progressed
                line, self.buf = self.buf[:i], self.buf[i + 2:]
                progressed = True
                if line == b"":
                    self.state = "done"
                    return True

    def _finish(self, t):
        self.state = "head"
        self.head = None
        self.on_end(t)

    def close(self, t):
        if self.state == "close" or (self.state != "head" and self.buf == b"" and self.state == "length" and self.remaining == 0):
            self._finish(t)

