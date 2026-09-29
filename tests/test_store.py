import os

from kvprof.store import RawCache, TraceStore


class CountingTokenizer:
    def __init__(self):
        self.calls = 0

    def tokenize(self, text, pieces=True):
        self.calls += 1
        return [(i, None) for i in range(len(text.split()))]


def test_list_and_paths(tmp_path):
    s = TraceStore(str(tmp_path))
    for n in ("20260101-000000.000-0002.json", "20260101-000000.000-0001.json", "20260101-000000.000-0001.analysis.json",
              ".conversion.json", "x.json.123.tmp"):
        (tmp_path / n).write_text("{}")
    assert s.list_ids() == ["20260101-000000.000-0001", "20260101-000000.000-0002"]
    assert s.analysis_path("a").endswith("a.analysis.json") and s.trace_path("a").endswith("a.json")
    assert TraceStore(str(tmp_path / "missing")).list_ids() == []


def test_analysis_roundtrip_and_fingerprint(tmp_path):
    s = TraceStore(str(tmp_path))
    fp0 = s.fingerprint()
    (tmp_path / "t1.json").write_text('{"id": "t1"}')
    assert s.fingerprint() != fp0 and s.load_trace("t1") == {"id": "t1"}
    s.save_analysis({"id": "t1", "kind": "cold"})
    assert s.load_analysis("t1")["kind"] == "cold" and s.load_analysis("nope") is None


def test_conversion(tmp_path):
    s = TraceStore(str(tmp_path / "new"))
    assert s.load_conversion() is None
    s.save_conversion({"variant": {"a": 1}})
    assert s.load_conversion() == {"variant": {"a": 1}}


def test_rawcache(tmp_path):
    s = TraceStore(str(tmp_path))
    tk = CountingTokenizer()
    rc = RawCache(s, tk)
    assert rc.count("a b c") == 3 and rc.count("a b c") == 3 and tk.calls == 1
    assert rc.count(None) is None and rc.count("") == 0
    rc.save()
    assert os.path.exists(tmp_path / ".rawcache.json")
    tk2 = CountingTokenizer()
    assert RawCache(s, tk2).count("a b c") == 3 and tk2.calls == 0
