# routerai

Python wrapper for the [RouterAI](https://routerai.ru) API — unified access to 450+ AI models
(OpenAI, Anthropic, Google, DeepSeek, Qwen, ...) with ruble pricing.

## Features

- OpenAI-compatible chat completions with sync + async support (one client instance
  supports both; transports are kept separate)
- Parsed responses: content, reasoning, tool calls, alternatives, token usage and cost
  in rubles (`Decimal`)
- Streaming (SSE) with per-chunk deltas; typed `StreamInterruptedError` on mid-stream
  failures, no unsafe automatic retries after the first chunk
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
```

## Async

```python
result = await client.chat.acomplete("deepseek/deepseek-v4-pro", "Привет!")
async for chunk in client.chat.astream(...):
    ...
await client.aclose()
```

## Configuration

| Option | Description |
| --- | --- |
| `api_key` / `ROUTERAI_API_KEY` | API key (env var used when argument is None) |
| `base_url` / `ROUTERAI_BASE_URL` | base URL; precedence: explicit argument > env var > `https://routerai.ru/api/v1` |
| `timeout` | per-request timeout in seconds (default 60) |
| `max_retries` | retry attempts with exponential backoff + jitter (default 2) |
| `retry_unsafe_methods` | retry POST/PATCH/DELETE on 5xx too (default False; RouterAI already
  does provider fallback, a client-side POST retry may start a new billed generation) |
| `http_client` / `async_http_client` | inject external httpx transports (never closed by the library) |

Retries honour the `Retry-After` header. Safe methods (GET/HEAD) are retried on
429/5xx; unsafe methods only on 429 by default.

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
| `StreamInterruptedError` | SSE broke after chunks were already delivered |

## Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("routerai").setLevel(logging.DEBUG)  # API keys are masked
```

## License

MIT
