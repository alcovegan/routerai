# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/) and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-08-14

### Added

- `client.chat.run_tools(model, prompt, tools=[fn])` runs the loop of model →
  function → model. Schemas are derived from the signature; a tool that raises
  is reported back to the model instead of crashing the caller; `max_turns`
  bounds what a looping model can spend.
- `client.chat.parse(..., response_model=Model)` asks for a JSON schema derived
  from a pydantic model and validates the answer against the same model.
- Cost accounting: `client.usage` keeps running totals by model and by label,
  `client.track("label")` scopes them to a block, `client.on_usage(hook)`
  reports every request. Streamed usage is collected too, and accounting no
  longer depends on the log level.
- Per-call options: `timeout`, `max_retries` and `headers` can be passed to any
  method for that one request.
- `default_headers` and `app_info` on `RouterAI(...)`, and a `User-Agent` of
  `routerai-python/<version>`.
- `response_format` works with `stream()`, which was previously unreachable.
- Async catalog: `asearch`, `agrouped`, `aby_capability`, and
  `models.cheapest()`/`acheapest()` for picking a model by price.
- `stream()`/`astream()` return a wrapper that also works as a context manager,
  so an abandoned async stream releases its connection immediately instead of
  waiting for the loop to shut down its async generators.

### Fixed

- `typing.get_type_hints()` works on the public methods again. Unpack was
  imported only under TYPE_CHECKING, so on Python 3.10 resolving annotations
  raised NameError — which breaks FastAPI, sphinx and anything else that reads
  annotations at runtime. `typing-extensions` is now a dependency below 3.11.

### Changed

- Opening a stream is logged at DEBUG; the INFO line is written when the stream
  finishes, and now carries the tokens and cost from its final chunk.
- The package version lives in `routerai/_version.py`; hatch reads it from
  there, so it is no longer duplicated in `pyproject.toml`.

## [0.2.0] - 2026-08-14

Everything below was found by auditing the SDK against the live API and is
covered by tests over recorded responses in `tests/cassettes/`.

### Breaking

- Typed errors (`AuthenticationError`, `RateLimitError`, …) now inherit
  `APIStatusError`, so `except APIStatusError` catches them and every one of
  them carries `.status_code` and `.body`. Code that relied on
  `APIStatusError` *not* matching a 401 needs to reorder its handlers.
- `status_code` reports the code that explains the failure. When RouterAI
  wraps an upstream error in HTTP 200 or 503, the transport status stays
  available as `.http_status`, with `.provider_code` and `.status_source`
  alongside.
- A closed client raises `RuntimeError` instead of silently opening a new
  connection pool with default settings.
- `KeyInfo.key` is a `SecretStr`; read it with `.get_secret_value()`.
- `models.search(max_price_prompt=...)` no longer returns models billed in
  another unit (images, seconds, search units). They were reported as free
  and therefore ranked first.

### Fixed

- `StreamAccumulator` merges tool-call deltas by index. Streamed tool calling
  produced fragments that no JSON parser accepted.
- `Usage.tokens()` counts `input_tokens`/`output_tokens` too — transcription
  and `GET /generation` use that naming, and reported zero.
- `Usage.cost_rub` survives `model_dump()` → `model_validate()`; caching a
  result used to lose the price silently.
- `ModelPricing` coerces every price to `Decimal`, including units the SDK
  does not know yet; `per_million()` raised `TypeError` on those.
- Embeddings requested with `encoding_format="base64"` decode to numbers
  instead of a list of single characters.
- A provider error inside a stream raises the typed error in image streams
  too; previously it was handed back as an empty chunk.
- An empty SSE `data:` line is treated as a keep-alive rather than parsed.
- Video tasks unwrap the `{"data": ...}` envelope for every field, so a task
  wrapped that way can be polled; unknown statuses count as still running
  instead of as success with no result.
- A failed video generation is reachable again: for a task, `error` is a
  state, not a failure of the HTTP call.
- Path segments are escaped, so an id like `../keys` cannot redirect a
  request — with its Authorization header — to another endpoint.
- Webhook verification compares bytes and signs the timestamp as received;
  a non-ASCII signature header raised `TypeError` instead of failing the check.
- Formatting a cost cannot raise, so an odd `usage.cost` no longer destroys a
  response that was already paid for.
- Transcription reads `text`, `srt` and `vtt` as text instead of failing to
  parse them as JSON.
- `_json()` returns `{}` for a 204, so deleting a key no longer fails after
  the server deleted it.

### Added

- `BadRequestError`, `ConflictError`, `UnprocessableEntityError`,
  `ServerError`, `APIConnectionError`, `APITimeoutError`,
  `ResponseParsingError`, `ErrorInfo`.
- The types the SDK returns are exported from the package: `StreamChunk`,
  `VideoTask`, `ImageResult`, `GeneratedImage`, `ImageStreamChunk`,
  `EmbeddingsResult`, `RerankResult`, `CompletionsResult`, `CompletionChoice`,
  `SpeechResult`, `TranscriptionResult`, `GenerationInfo`, plus
  `verify_video` and `signing_secret`.
- `ModelPricing.price()`, `.priced_units()`, `.is_free()`.
- Cassette fixtures recorded from the live API, and CI jobs that build the
  package, gate coverage and install the declared minimum dependency versions.

## [0.1.1] - 2026-08-14

### Fixed

- Propagated the video polling deadline through HTTP-status retries and
  `Retry-After` backoff; async polling now cancels an in-flight refresh at the
  wall-clock deadline.
- Normalized image/video download transport failures into `RequestError` while
  preserving the original `httpx` exception as `__cause__`.
- Made atomic-download cleanup cancellation-safe and resistant to stale or
  colliding temporary files.
- Unified public HTTPS validation for video inputs, callbacks and image
  downloads; private literal addresses, malformed ports, fragments and
  credential-bearing URLs are rejected before network access.
- Bounded image data-URI decoding before allocation and validated download
  limits before opening a connection.

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
