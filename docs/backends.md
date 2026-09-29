# Backends

kvprof's analysis needs five things from an inference server. Each is a small interface in
`src/kvprof/backends/base.py`; llama.cpp (`backends/llamacpp/`) is the only implementation in v1.

| Seam | Interface | What it must provide |
|---|---|---|
| Renderer | `render(messages, tools) -> str` | the exact prompt string the server will process for an OpenAI-shaped message list |
| Tokenizer | `tokenize(text, pieces=True) -> [Token(id, piece_bytes)]` | the server's token ids **with byte pieces**, so tokens map to text |
| UsageNormalizer | `normalize(raw_usage, n_rendered) -> Usage` | prompt / cached / uncached / output counts from the response, either wire shape |
| TelemetrySource | `load()`, `anchor()`, `pair(analyses, ...)` | per-task server timings (prefill, decode, slot choice) and their pairing with recorded requests |
| CacheTopology | `lanes(server_info)` | how the cache is laid out (slots), for the timeline |

Plus `server_info()` (build, model, context size, number of slots, chat template).

## Capability matrix (surveyed 2026-09-29)

| | llama.cpp `llama-server` | Ollama | vLLM / SGLang | mlx-lm `mlx_lm.server` | LM Studio |
|---|---|---|---|---|---|
| Renderer | `POST /apply-template` (exact) | no HTTP endpoint; template in the Modelfile, rendering would have to be reimplemented | chat templates via the tokenizer's Jinja; no documented render-only endpoint in every version | none | none |
| Tokenizer with pieces | `POST /tokenize` `with_pieces` | none | `/tokenize` (ids; pieces vary by version) | none | none |
| Cached tokens in usage | Anthropic `cache_read_input_tokens`; OpenAI `prompt_tokens_details.cached_tokens` + `timings` | `prompt_eval_count` only (no cached split) | `prompt_tokens_details.cached_tokens` when enabled | no | omitted from usage |
| Anthropic Messages endpoint | yes (`/v1/messages`, `count_tokens`) | no | no (OpenAI only) | no | yes |
| Server facts | `/props` (n_ctx, total_slots, build_info, chat_template), `/slots` | `/api/show` | `/v1/models`, metrics | minimal | minimal |
| Per-task telemetry | log: slot selection (LCP/LRU, `f_keep`), prefill/decode per task | log (less detail) | Prometheus metrics (aggregate, prefix-cache hit rate) | none | none |

## Roadmap

- **Ollama (reach).** The most common local server. Missing a render and a tokenize endpoint means sections
  would have to come from a reimplemented template (approximate) and token counts from `prompt_eval_count`.
  Worth doing for the request-level view (prefilled per request, diff by text) even without token-exact sections.
- **vLLM (capability).** Automatic prefix caching with block-level reuse and cached-token usage; a different cache
  model (paged blocks, no slots) that would exercise the CacheTopology seam.
- **mlx-lm, LM Studio.** Not planned until they report cached tokens: on LM Studio 0.4.x with an MLX model, the
  identical 23K-token prompt cost a full prefill every time (no reuse to profile).

## Adding a backend

1. Create `backends/<name>/` with a `Backend` subclass: `renderer`, `tokenizer`, `usage`, `topology`,
   `server_info()`, `telemetry(log_path, pid_file)` (may return `None`).
2. Register it in `backends/__init__.py:get_backend`.
3. Give the fake server in `tests/fake_upstream.py` the endpoints it needs (or add a second fake) and run the
   end-to-end test against it.
4. Record the tested server version in the backend (as llama.cpp's `TESTED_BUILDS`) and teach `doctor` to check it.
