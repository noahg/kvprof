import re

from kvprof.serve import index_html


def test_index_is_package_data_and_generic():
    html = index_html().decode("utf-8")
    assert html.startswith("<!doctype html>")
    assert not re.search(r"claudelocal|Qwen|this Mac|two-slot|LOCAL_LLM|tmp/trace|bin/trace\.py|bin/tap\.py|from_claude_code", html)
    for key in ("meta.n_slots", "S.meta.n_ctx", "S.meta.trace_dir", "S.meta.package", "d.profile === \"generic\""):
        assert key in html, key
