# Contributing

## Setup

```sh
uv venv && uv pip install -e '.[dev]'
.venv/bin/pytest -q
```

Runtime code is stdlib-only; keep it that way (tests use pytest). Python 3.11+.

The tests need no llama-server: `tests/fake_upstream.py` is a fake with the endpoints kvprof uses
(`/props`, `/v1/models`, `/slots`, `/tokenize` with a toy tokenizer that splits non-ASCII characters mid-byte,
`/apply-template` with a Qwen3-Coder-style ChatML renderer or a Llama-3-style one, `count_tokens`, streaming and
non-streaming `/v1/messages` and `/v1/chat/completions`), a two-slot prefix cache, and a server log written in
llama.cpp's b10909 format. `tests/test_e2e.py` runs fake → tap → analyzer → GUI API.

Against a real server:

```sh
kvprof doctor --upstream http://127.0.0.1:8080 --server-log server.log
```

## The log-format rule

`src/kvprof/backends/llamacpp/serverlog.py` parses llama-server's log with regular expressions
(`LOG_PATTERNS`). **Keep them in sync with `tests/fixtures/server.log` and `TESTED_BUILDS`**: the fixture lines are
copied from a real b10909 log (values synthetic), so a pattern change that stops matching the real format fails
`tests/test_serverlog.py`. When a new llama.cpp build changes the format:

1. Run `kvprof doctor --server-log <its log>`; it lists slot lines no pattern matched.
2. Add the new line shapes to the fixture (keep the old ones if both formats should parse), adjust the patterns.
3. Add the build to `TESTED_BUILDS`.

## Adding a template profile

A profile builds the section tree for one family of chat templates (`src/kvprof/sections/`).

1. Write `build_sections_<name>(rendered, request, messages, origins, units) -> (tree, mapping_ok, reason)`.
   `units[k]` are the `RenderUnit`s of converted message `k`; `request.tool_units()` the tool definitions. Use the
   same node ids as the other profiles (`system.text`, `tool:<name>`, the unit ids) so diffs and the GUI work.
   Only character ranges are needed; `assign_tokens` maps them to tokens.
2. Detect it in `sections/profiles.py:detect_profile` from the template text (`/props` `chat_template`), add it to
   `PROFILES`, dispatch it in `sections/__init__.py:build_sections`.
3. Add a renderer for it to the fake server and a test like `tests/test_sections_generic.py`.

## Adding a backend

See [docs/backends.md](docs/backends.md).

## Packaging

Three things are built from the wheel, all by scripts that also run in CI:

```sh
uv build                                   # dist/kvprof-<version>-py3-none-any.whl
packaging/build-runtime.sh macos arm64     # the wheel inside a pinned standalone Python: build/runtime/…, dist/….tar.gz
macos/build-app.sh                         # build/app/<arch>/kvprof.app around that runtime, ad-hoc signed
macos/sign-and-notarize.sh                 # dist/kvprof-<version>-macos-<arch>.dmg
packaging/smoke.sh build/runtime/kvprof-*-macos-arm64/bin/kvprof
```

- The runtime is what `packaging/install.sh` downloads and what the app embeds. Its Python is pinned in
  `packaging/python.lock` (python-build-standalone); to update, change the tag, the version and the four digests.
- The app (`macos/Sources/KvprofApp`) only supervises `kvprof start --exit-with-parent` and reads its
  `/api/status`; anything the menu shows should come from there, not from a second implementation in Swift.
- Nothing may write into the app bundle at run time (it is signed): the launcher runs Python with `-B` and the
  bytecode is compiled at build time.
- `SIGN_IDENTITY="Developer ID Application: … (TEAMID)" NOTARY_PROFILE=<profile> macos/sign-and-notarize.sh`
  signs with the hardened runtime, notarizes and staples. Without them the disk image is ad-hoc signed, for local use.
- `swift macos/make-icon.swift` redraws `macos/AppIcon.icns`.

## Releasing

1. Set `__version__` in `src/kvprof/__init__.py`, date the CHANGELOG entry, commit.
2. Rehearse: push a tag like `v0.2.0rc1` (with `__version__ = "0.2.0rc1"`). The release workflow publishes to
   TestPyPI and creates a draft prerelease with every artifact.
3. Push `vX.Y.Z`. The workflow tests, builds the wheel, the Linux and macOS runtimes and the signed, notarized disk
   images, publishes to PyPI (trusted publishing), creates the GitHub release with `SHA256SUMS` and `install.sh`,
   and regenerates the cask in `noahg/homebrew-kvprof` from `packaging/homebrew/kvprof.rb.in`.

Repository secrets the workflow reads: `MACOS_SIGN_IDENTITY`, `MACOS_CERTIFICATE_P12` (base64),
`MACOS_CERTIFICATE_PASSWORD`, `NOTARY_KEY_P8`, `NOTARY_KEY_ID`, `NOTARY_ISSUER_ID`, `HOMEBREW_TAP_TOKEN`. PyPI and
TestPyPI need this repository's `release.yml` (environment `pypi`) registered as a trusted publisher.

## Style

Match the surrounding code: short functions, comments that say why, no new dependencies.
