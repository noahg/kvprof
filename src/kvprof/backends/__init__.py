"""Inference-server backends. v1 ships llama.cpp only; see docs/backends.md for the roadmap."""

BACKENDS = ("llamacpp",)


def get_backend(name="llamacpp", **kw):
    if name == "llamacpp":
        from .llamacpp import LlamaCppBackend
        return LlamaCppBackend(**kw)
    raise ValueError(f"unknown backend {name!r} (available: {', '.join(BACKENDS)})")
