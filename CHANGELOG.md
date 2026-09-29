# Changelog

## 0.1.0 (2026-09-29)

First release, extracted from a private llama-server setup where it was used to trace Claude Code sessions.

- `kvprof tap`: passive recorder for `/v1/messages` (Anthropic) and `/v1/chat/completions` (OpenAI), streamed or
  not; trace schema 2 (schema-1 traces still load).
- `kvprof analyze`: renders every request through the server's template, tokenizes it, maps tokens to sections,
  diffs against the closest earlier prompt, classifies (cold, tool round-trip, main turn, rewrite, compaction,
  side, retry, warm), pairs with the llama-server log, and checks the rendered count against the server's usage.
- `kvprof serve`: live GUI (timeline per slot, requests grouped by conversation, token budget, diff, token view).
- `kvprof doctor`: checks the server's endpoints, build and log format.
- `kvprof calibrate`: derives the undocumented details of the Anthropic → OpenAI conversion from the server.
- Template profiles: `chatml_qwen` (exact per-block sections) and `generic` (approximate, any template).
- Backend: llama.cpp `llama-server`, tested on b10909.
