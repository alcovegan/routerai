# routerai

Python wrapper for the [RouterAI](https://routerai.ru) API — unified access to 450+ AI models
(OpenAI, Anthropic, Google, DeepSeek, Qwen, ...) with ruble pricing.

## Features

- OpenAI-compatible chat completions with sync + async support (one client instance
  supports both; transports are kept separate)
- Parsed responses: content, reasoning, tool calls, alternatives, token usage and cost
  in rubles (`Decimal`)
- Streaming (SSE) with per-chunk deltas; once a successful response stream is opened
  no automatic retries happen (even if 0 chunks arrived), mid-stream failures raise a
  typed `StreamInterruptedError`
- Models catalog: listing, client-side search, grouping by capabilities (text, reasoning,
  vision, image/video/audio generation, speech, transcription, embeddings, rerank, tools)
- Post-hoc cost lookup by generation id (`X-Generation-Id`)
- Multiple API keys via a per-instance `Registry` (contextvar, thread/async safe)
- Images, video (polling), audio (TTS/STT), embeddings, rerank, API-key and team management
- Zero-effort logging via the standard `logging` module (namespace `routerai`, keys masked)

## Install

```bash
pip install routerai
```

## Quickstart

```python
from routerai import RouterAI

client = RouterAI(api_key="sk-...")  # or set ROUTERAI_API_KEY env var

result = client.chat.complete("deepseek/deepseek-v4-pro", "Привет!")
print(result.content)
print(result.cost_rub)  # Decimal, in rubles
```

### Model catalog

```python
client.models.all()                       # full catalog (cached, TTL by default 10 min)
client.models.search("claude", capabilities=["reasoning"], min_context=100_000)
client.models.by_capability("image")      # image generation models
client.models.grouped()                   # dict[Capability, list[Model]]
client.models.get("deepseek/deepseek-v4-pro").pricing.per_million("prompt")
client.models.endpoints("anthropic/claude-sonnet-5")  # providers + prices
```

### Several API keys

```python
from routerai import RouterAI, Registry

registry = Registry(main=RouterAI(api_key=A), personal=RouterAI(api_key=B))
registry["personal"].chat.complete(...)
with registry.using("main"):
    ...
```

### Streaming

```python
for chunk in client.chat.stream("openai/gpt-5.6-sol", "Расскажи сказку"):
    print(chunk.content, end="")
```

### Speech-to-text

```python
client.audio.transcribe("openai/whisper-large-v3", "voice.wav")      # format from suffix
client.audio.transcribe("openai/whisper-large-v3", raw_bytes, format="mp3")
for chunk in client.audio.speech_stream("x-ai/grok-voice-tts-1.0", "текст", voice="eve"):
    ...
```

### Video lifecycle

```python
from routerai import FrameImage, ImageReference

task = client.videos.create(
    "bytedance/seedance-2.0",
    "Персонаж идёт через осенний лес",
    frame_images=[FrameImage(url="https://example.com/first.png", frame_type="first_frame")],
    # or reference-to-video: input_references=[ImageReference(url=...)]
)
task.wait(timeout=600, interval=5)      # absolute deadline, incl. retries
task.save("video.mp4", index=0)         # streaming download, atomic rename
await task.asave("video.mp4")           # async variant, cancellation-safe

# webhooks: verify HMAC over the raw body with your api key
from routerai.webhooks import verify_video
data = verify_video(raw_body, signature, api_key, timestamp, max_age_seconds=300)
```

## Async

```python
result = await client.chat.acomplete("deepseek/deepseek-v4-pro", "Привет!")
async for chunk in client.chat.astream(...):
    ...
await client.aclose()
```

Sync and async transports live in separate slots, so one instance can serve both
modes. Note the lifecycle: `close()` closes the sync connection pool, `await
aclose()` closes the async one. If a single instance was used from both modes,
call both. External transports injected via `http_client`/`async_http_client`
are never closed by the library.

## Configuration

| Option | Description |
| --- | --- |
| `api_key` / `ROUTERAI_API_KEY` | API key (env var used when argument is None) |
| `base_url` / `ROUTERAI_BASE_URL` | base URL; precedence: explicit argument > env var > `https://routerai.ru/api/v1` |
| `timeout` | per-request timeout in seconds (default 60) |
| `max_retries` | retry attempts with exponential backoff + jitter (default 2) |
| `max_retry_after` | upper bound for an upstream `Retry-After` header, seconds (default 60) |
| `retry_unsafe_methods` | retry POST/PATCH/DELETE on 5xx too (default False; RouterAI already
  does provider fallback, a client-side POST retry may start a new billed generation) |
| `http_client` / `async_http_client` | inject external httpx transports (never closed by the library) |

Retries honour the `Retry-After` header. Safe methods (GET/HEAD) are retried on
429/5xx; unsafe methods only on 429 by default.

The `extra` parameter is an escape hatch for provider-specific request fields:
it can never override library-managed keys (`model`, `messages`, `stream`, ...)
— colliding keys raise `ValueError`.

## Errors

| Exception | HTTP |
| --- | --- |
| `AuthenticationError` | 401 |
| `InsufficientFundsError` | 402 |
| `PermissionDeniedError` | 403 |
| `NotFoundError` | 404 |
| `RateLimitError` | 429 |
| `NoProviderError` | 503 with "no provider available" |
| `APIStatusError` | other 4xx/5xx (has `.status_code`, `.body`) |
| `RequestError` | transport failure after retries |
| `DeadlineExceededError` | an absolute polling deadline passed (video `wait()`) |
| `StreamInterruptedError` | SSE broke after the response stream was opened (`.chunks_received` may be 0) |
| `VideoGenerationError` | a video task reached `failed`/`cancelled`/`expired` |
| `WebhookVerificationError` | video webhook failed signature or freshness checks |

## Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("routerai").setLevel(logging.DEBUG)  # API keys are masked
```

## License

MIT
