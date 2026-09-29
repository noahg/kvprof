"""Server-sent events: line-level parser shared by the recorder (live) and the analyzer (from stored lines)."""
import json


class SSEParser:
    """Feed decoded response lines (without line terminators); get (event, data_dict) tuples back.

    The event name is the `event:` field, else the data's "type" (Anthropic), else None (OpenAI chunks)."""

    def __init__(self):
        self.event = None
        self.data = []
        self.events = []

    def feed_line(self, line):
        out = []
        if line == "":
            if self.data or self.event:
                raw = "\n".join(self.data)
                try:
                    data = json.loads(raw) if raw else None
                except ValueError:
                    data = {"_raw": raw}
                ev = (self.event or (data.get("type") if isinstance(data, dict) else None), data)
                self.events.append(ev)
                out.append(ev)
            self.event, self.data = None, []
        elif line.startswith("event:"):
            self.event = line[6:].strip()
        elif line.startswith("data:"):
            self.data.append(line[5:].lstrip(" "))
        return out

    def flush(self):
        return self.feed_line("")


def parse_sse_lines(lines):
    p = SSEParser()
    for ln in lines:
        p.feed_line(ln)
    p.flush()
    return p.events
