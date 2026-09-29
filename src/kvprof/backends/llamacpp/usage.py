"""llama-server usage semantics, both wire shapes.

Anthropic: input_tokens is the uncached part, cache_read_input_tokens the reused part, their sum the prompt.
OpenAI: prompt_tokens is the whole prompt; the cached part is prompt_tokens_details.cached_tokens or
timings.cache_n; timings.prompt_n is what was prefilled. Streams carry `usage` only when the client sets
stream_options.include_usage, but llama.cpp sends `timings` on the final chunk either way.
"""
from ..base import Usage, UsageNormalizer


def _semantics(raw_prompt, cached, n_rendered):
    """How the reported prompt count relates to the rendered prompt: "total" | "uncached" (default)."""
    if raw_prompt is None:
        return None
    if n_rendered is not None and raw_prompt == n_rendered:
        return "total"
    if n_rendered is not None and raw_prompt + (cached or 0) == n_rendered:
        return "uncached"
    return None


class LlamaCppUsageNormalizer(UsageNormalizer):
    def normalize(self, raw, n_rendered=None):
        if not raw:
            return Usage()
        f = raw.get("fields") or {}
        tm = raw.get("timings") or {}
        u = Usage(api=raw.get("api"), stop_reason=raw.get("stop_reason"), model=raw.get("model"),
                  output_estimated=bool(raw.get("output_tokens_estimated")))
        if tm:
            u.prefill_ms = tm.get("prompt_ms")
            u.decode_ms = tm.get("predicted_ms")
            u.decode_rate = tm.get("predicted_per_second")
        if raw.get("api") == "openai":
            pt = f.get("prompt_tokens")
            if pt is None and tm.get("prompt_n") is not None and tm.get("cache_n") is not None:
                pt = tm["prompt_n"] + tm["cache_n"]  # no usage block (client did not ask for it): llama.cpp timings still add up
            cached = (f.get("prompt_tokens_details") or {}).get("cached_tokens")
            if cached is None:
                cached = tm.get("cache_n")
            u.raw_prompt = pt
            u.total = pt
            u.cached = cached if cached is not None else (0 if pt is not None else None)
            if tm.get("prompt_n") is not None:
                u.uncached = tm["prompt_n"]
            elif pt is not None:
                u.uncached = pt - (u.cached or 0)
            u.output_tokens = f.get("completion_tokens", tm.get("predicted_n"))
            if raw.get("output_tokens_estimated") and tm.get("predicted_n") is not None:
                u.output_tokens, u.output_estimated = tm["predicted_n"], False
            u.semantics = "total" if pt is not None else None
            if pt is not None and n_rendered is not None and pt != n_rendered and pt + (u.cached or 0) == n_rendered:
                u.semantics, u.total, u.uncached = "uncached", n_rendered, pt
            return u
        inp = f.get("input_tokens")
        cached = f.get("cache_read_input_tokens")
        u.raw_prompt = inp
        u.output_tokens = f.get("output_tokens")
        if inp is None:
            return u
        u.cached = cached or 0
        sem = _semantics(inp, cached, n_rendered)
        u.semantics = sem or "uncached"  # unknown: assume llama.cpp's documented meaning; exactness flags a mismatch
        if u.semantics == "total":
            u.total, u.uncached = inp, inp - u.cached
        else:
            u.total, u.uncached = inp + u.cached, inp
        return u
