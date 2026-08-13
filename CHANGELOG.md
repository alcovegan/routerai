# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/) and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-08-13

First public alpha (tag `v0.1.0-alpha.1`; the package version is `0.1.0`).

### Added

- OpenAI-compatible chat completions with sync + async support, streaming (SSE)
  with per-chunk deltas, tools, JSON mode and `service_tier`/`provider` routing.
- Parsed results: content, reasoning, tool calls, alternatives, token usage and
  ruble cost (`Decimal`), generation ids, post-hoc cost lookup.
- Models catalog with client-side search, capabilities grouping (text,
  reasoning, vision, image/video/audio generation, speech, transcription,
  embeddings, rerank, tools) and provider endpoints.
- Multiple API keys via a per-instance `Registry` (contextvar-based).
- Images (typed parameters, strict parsing, SSE previews, safe downloads),
  videos (polling, indexed streaming downloads, webhook verification),
  audio (TTS/STT, byte streaming), embeddings, rerank, completions,
  responses and messages protocols.
- Typed management APIs: account balance, API keys and team management
  (master key).
- Single-decoded response envelope shared by logging, error mapping and
  resource parsers.
- Endpoint-aware retries honouring `Retry-After` with a configurable cap,
  reserved-key policy for `extra`, typed error model (incl. provider errors
  wrapped in HTTP 200 and SSE error envelopes).
- Hermetic test suite (network-deny gate) plus an opt-in live integration
  matrix against the real API.
- `py.typed` marker, strict mypy, ruff lint/format and CI on Python 3.10-3.14.
