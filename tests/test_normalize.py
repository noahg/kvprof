from kvprof.analyze import exactness
from kvprof.backends.llamacpp.usage import LlamaCppUsageNormalizer

N = LlamaCppUsageNormalizer()


def anth(**f):
    return {"api": "anthropic", "fields": f, "timings": None, "stop_reason": "end_turn", "model": "m", "output_tokens_estimated": False}


def oai(fields, timings=None):
    return {"api": "openai", "fields": fields, "timings": timings, "stop_reason": "stop", "model": "m", "output_tokens_estimated": False}


def test_anthropic_uncached_semantics():
    u = N.normalize(anth(input_tokens=10, cache_read_input_tokens=90, output_tokens=5), 100)
    assert (u.total, u.cached, u.uncached, u.output_tokens, u.semantics, u.raw_prompt) == (100, 90, 10, 5, "uncached", 10)
    ex = exactness(u.to_dict(), 100)
    assert ex["usage"] is True and ex["usage_semantics"] == "uncached"


def test_anthropic_total_semantics():
    u = N.normalize(anth(input_tokens=100, cache_read_input_tokens=90), 100)
    assert (u.total, u.cached, u.uncached, u.semantics) == (100, 90, 10, "total")
    assert exactness(u.to_dict(), 100)["usage_semantics"] == "total"


def test_anthropic_mismatch_assumes_uncached():
    u = N.normalize(anth(input_tokens=12, cache_read_input_tokens=90), 100)
    assert (u.total, u.uncached, u.semantics) == (102, 12, "uncached")
    ex = exactness(u.to_dict(), 100)
    assert ex["usage"] is False and ex["delta"] == -88


def test_anthropic_no_usage():
    u = N.normalize(anth(output_tokens=3), 100)
    assert u.total is None and u.cached is None and u.output_tokens == 3
    assert exactness(u.to_dict(), 100)["usage"] is None
    assert N.normalize(None, 100).total is None


def test_openai_usage_and_timings():
    u = N.normalize(oai({"prompt_tokens": 100, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 80}},
                        {"prompt_n": 20, "prompt_ms": 50.0, "predicted_ms": 80.0, "predicted_per_second": 50.0}), 100)
    assert (u.total, u.cached, u.uncached, u.output_tokens, u.prefill_ms, u.decode_rate, u.semantics) == (100, 80, 20, 4, 50.0, 50.0, "total")
    assert exactness(u.to_dict(), 100)["usage"] is True


def test_openai_cache_from_timings():
    u = N.normalize(oai({"prompt_tokens": 100}, {"cache_n": 60}), 100)
    assert (u.cached, u.uncached) == (60, 40)


def test_openai_timings_only():
    # stream without stream_options.include_usage: completion_tokens was estimated from the chunk count
    raw = dict(oai({"completion_tokens": 5}, {"cache_n": 60, "prompt_n": 40, "predicted_n": 3}), output_tokens_estimated=True)
    u = N.normalize(raw, 100)
    assert (u.total, u.cached, u.uncached, u.output_tokens, u.output_estimated) == (100, 60, 40, 3, False)
    assert exactness(u.to_dict(), 100)["usage"] is True


def test_openai_nothing():
    u = N.normalize(oai({"completion_tokens": 3}, None), 100)
    assert u.total is None and u.cached is None and u.output_tokens == 3


def test_openai_uncached_semantics_detected():
    u = N.normalize(oai({"prompt_tokens": 20, "prompt_tokens_details": {"cached_tokens": 80}}), 100)
    assert (u.semantics, u.total, u.uncached) == ("uncached", 100, 20)
