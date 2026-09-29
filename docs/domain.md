# How llama-server's prompt cache behaves (what kvprof measures)

These are the facts the analyzer relies on. They were established on llama.cpp **b10909** (Qwen3-Coder-Next
UD-Q3_K_XL, 2 slots × 128K, Apple M1 Max) with Claude Code 2.1.283 as the client, by recording every request,
rendering it through the server's own template and pairing it with the server's log. Numbers are from that
setup; the mechanisms are llama.cpp's.

## The prompt is the whole conversation, every time

A coding agent sends the entire conversation on every request: the system prompt, every tool definition,
and all messages so far. The server renders that into one string with the model's chat template, tokenizes
it, and then has to *prefill* (process) every token it does not already have in cache before it can produce
the first output token. Time to first token is almost entirely prefill.

A typical Claude Code request on this setup was 34,430 tokens, of which:

| Section | Tokens | Share |
|---|---:|---:|
| system text | 1,305 | 3.8 % |
| tools (30 definitions, rendered as the template's `<tools>` block) | 27,525 | 79.9 % |
| messages (the prompt, `<system-reminder>`s, a `role: system` message carrying environment, CLAUDE.md, memory) | 5,597 | 16.3 % |
| generation prompt (`<|im_start|>assistant\n`) | 3 | |

One MCP tool alone (`mcp__plugin_posthog_posthog__exec`) was 8,126 tokens. The system + tools prefix is the
part every later request of the session shares; it is what makes follow-up requests cheap and what a cold
start pays for.

## Rendering and counting are exact

- `POST /apply-template` returns exactly the string the server processes for a given message list, and
  `POST /tokenize` with `with_pieces: true, add_special: false` returns the tokens with their byte pieces
  (a piece that is not valid UTF-8 on its own comes back as a list of byte values). Concatenated pieces equal
  the rendered bytes, so every token maps to a byte range and every section to a token range.
- For Anthropic requests the server first converts the body to OpenAI chat messages. The few undocumented
  details of that conversion (how several text blocks are joined; that the parts of a `tool_result` are
  joined with `""` on b10909) are calibrated once per model against `POST /v1/messages/count_tokens`
  (`kvprof calibrate`). After calibration every recorded request rendered to exactly the server's count.

## Usage semantics

| Wire shape | Prompt | Reused | Prefilled |
|---|---|---|---|
| Anthropic `/v1/messages` | `input_tokens + cache_read_input_tokens` | `cache_read_input_tokens` | `input_tokens` |
| OpenAI `/v1/chat/completions` | `prompt_tokens` | `prompt_tokens_details.cached_tokens` (or `timings.cache_n`) | `timings.prompt_n` (or prompt − cached) |

On llama-server the Anthropic `input_tokens` is the *uncached* part; a cold request has `cache_read 0`. The
analyzer checks which reading makes the numbers add up to the rendered prompt and reports `exact` when one does.
OpenAI streams carry `usage` only when the client sets `stream_options.include_usage`; llama.cpp attaches its
`timings` (prompt/predicted counts and milliseconds) to the final chunk either way. A 400 for an oversized
prompt reports `n_prompt_tokens`, which also matched the rendered count.

## The server log

At default verbosity llama-server logs, per task: which slot it chose and why
(`selected slot by LCP similarity, f_sim_best = …, f_keep = …` = a cached prefix matched; `selected slot by LRU` =
nothing matched, least recently used slot, cold), `processing task`, a progress line every 2048 prefilled tokens,
`prompt eval time = … ms / N tokens` (N = tokens prefilled), `eval time = … ms / M tokens` (decode), and
`stop processing: n_tokens = K` on release. **K = prompt + output − 1** (the last sampled token is never fed
back), which pairs a log task with a recorded request exactly by token counts. Timestamps are
`minutes.seconds.ms.us` since server start, so the log needs a wall-clock anchor: the server process's start
time (`--server-pid`) or the log file's mtime; pairing refines it to sub-second.

`f_keep × previous slot size` is how much of the slot the new task kept. A client disconnect mid-request shows
as `cancel task, id_task = N` within a couple of seconds; the tap relays the disconnect so the server cancels
exactly as it would without it.

## Slots and reuse

- Each slot holds one conversation's KV cache. A new task goes to the slot whose cached prompt shares the
  longest prefix with it (LCP), else to the least recently used one. With two slots a small side request
  (a session title, a sub-agent) need not evict the main conversation.
- **Plain-attention models** reuse the cache up to the first token that differs.
- **Hybrid/recurrent models** (Gated DeltaNet or Mamba layers; Qwen3-Next, Qwen3-Coder-Next) cannot roll their
  recurrent state back to an arbitrary position. llama.cpp keeps *context checkpoints* instead: observed near
  the start of each processed prompt (≈ 2,044 tokens, the end of the first 2048-token batch) and near its end
  (≈ prompt − 516, the start of the last 512-token batch). A new prompt that **extends** the previous one reuses
  everything. A prompt that changes in the middle falls back to the nearest checkpoint below the change. One
  measurement on b10909 kept 28,830 tokens (a checkpoint near the end of the tools block) for an edit at token
  29,411 and re-prefilled 7,832 (31 s instead of ~115 s cold); where an edit lands relative to the checkpoints
  decides the cost, and checkpoint positions are not logged at default verbosity.
- The host-memory prompt cache (`--cache-ram`) can restore a slot a conversation lost, but on b10909 it drops
  hybrid checkpoints (llama.cpp issue #25913, open), so a hybrid keeps its prefix only while it holds its slot.
- Two slots still share one GPU: a 764-token request that ran while the other slot prefilled 53K tokens took
  193 s. The analyzer marks overlapping requests (`concurrent with #N`).

## What each kind of request costs

| Kind | What changed | Cost |
|---|---|---|
| cold | nothing cached matched (first turn, new directory, slot taken) | the whole prompt: 34,430 tokens, 106 s at 323 tok/s |
| tool_roundtrip | previous prompt + the tool call + its result, at the end | the size of the result: a 2,188-token README read cost 2,193 prefilled tokens, 9.3 s |
| main_turn | previous prompt + the model's answer + a new user message | tens of tokens, < 1 s |
| rewrite / compaction | something before the end changed | from the change (plain attention) or the nearest checkpoint (hybrid) to the end |
| side | a request without tools (titles, summaries, sub-agents) | its own small prompt; on a one-slot server it evicts the main conversation |

Prefill throughput falls with depth (≈ 850 tok/s over the first 4K tokens to ≈ 210 tok/s by 43K on the
M1 Max), so a large tool result late in a long conversation costs more per token than the same result early on.

## Client-side rewrites (an example)

Claude Code 2.1.269's `-p --continue` appended "Continue from where you left off." to the **first** user message
of the resumed conversation. On a hybrid model that meant a full re-prefill of the whole session. 2.1.283 sends
the previous conversation unchanged plus the new message (pure append, 33 tokens prefilled). This is the kind of
change the analyzer's diff exists to catch: it names the section (`msg[0].block[3]`) and the inserted text.
