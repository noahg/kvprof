from kvprof.util import byte_starts, char_to_byte_table, fmt_n, lcp_len, parse_addr, piece_bytes, save_json_atomic, load_json, tok_at, tok_end_at


def test_byte_mapping():
    toks = [(1, b"he"), (2, b"llo"), (3, bytes([0xe2, 0x82])), (4, bytes([0xac])), (5, b"!")]
    bs = byte_starts(toks)
    assert bs == [0, 2, 5, 7, 8, 9]
    c2b = char_to_byte_table("hello€!")
    assert c2b == [0, 1, 2, 3, 4, 5, 8, 9]
    assert tok_at(bs, c2b[5]) == 2 and tok_end_at(bs, c2b[6]) == 4 and tok_at(bs, c2b[6]) == 4


def test_char_to_byte_4byte():
    assert char_to_byte_table("a😀b") == [0, 1, 5, 6]


def test_lcp():
    assert lcp_len([1, 2, 3], [1, 2, 4]) == 2
    assert lcp_len([1] * 10000, [1] * 9000 + [2]) == 9000
    assert lcp_len([], [1]) == 0
    assert lcp_len([1, 2], [1, 2]) == 2


def test_piece_bytes():
    assert piece_bytes("é") == "é".encode()
    assert piece_bytes([0xe2, 0x82]) == b"\xe2\x82"
    assert piece_bytes(None) == b""


def test_parse_addr():
    assert parse_addr("http://127.0.0.1:8082/", 1) == ("127.0.0.1", 8082)
    assert parse_addr("http://localhost:8082/v1", 1) == ("localhost", 8082)
    assert parse_addr("8083", 1) == ("127.0.0.1", 8083)
    assert parse_addr(":9000", 1) == ("127.0.0.1", 9000)
    assert parse_addr("localhost", 8080) == ("localhost", 8080)


def test_fmt_and_json(tmp_path):
    assert fmt_n(None) == "?" and fmt_n(12345) == "12,345"
    p = tmp_path / "x.json"
    save_json_atomic(str(p), {"a": "é"})
    assert load_json(str(p)) == {"a": "é"}
    assert [x.name for x in tmp_path.iterdir()] == ["x.json"]
