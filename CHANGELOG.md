# Changelog

## 0.2.0 (2026-10-20)

- `kvprof start`: the tap and the GUI in one process, with the GUI opened in the browser. It starts whether or not
  llama-server is up, attaches the analyzer when the server answers, re-attaches when it comes back with another
  model, and keeps what was analyzed when it goes away. A second `kvprof start` opens the running one's GUI.
- GUI: a first-run panel with the state of llama-server and the client settings for the tap's real address;
  `/api/status`.
- macOS menu-bar app (`kvprof.app`, in a disk image): runs `kvprof start` in the background with its own Python,
  shows whether it is recording or waiting for llama-server, opens the dashboard, copies the client settings, and
  can start at login.
- Self-contained runtime for macOS and Linux and an install script (`curl … | sh`) that needs no Python; a Homebrew
  cask; a release workflow that publishes to PyPI and GitHub.
- Default trace directory: `./.kvprof` when it exists, else a per-user directory with one subdirectory per model
  (`kvprof tap` alone still records into `./.kvprof`). `KVPROF_DATA_DIR` moves the per-user directory.

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
