# routerai

[Русская версия](README.ru.md)

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
- Streamed tool calls arrive assembled: `StreamAccumulator` merges the deltas by
  index, so `json.loads(call["function"]["arguments"])` just works
- Typed errors that agree with themselves: every server answer is an
  `APIStatusError` subclass carrying `.status_code`, and a provider failure
  wrapped in HTTP 200 still raises `RateLimitError`
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

# Not every model is billed per token: image models charge per image,
# rerank per search unit, video per second. Ask what a model charges for
# instead of reading a zero token price as "free".
pricing = client.models.get("black-forest-labs/flux.2-pro").pricing
pricing.priced_units()              # {"image_output"}
pricing.price("image_output")       # Decimal per image
pricing.is_free()                   # False

# A price filter therefore skips models billed in another unit rather than
# ranking them first — they are not free, their price is in another currency.
client.models.search(max_price_prompt=1.0)
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
task.wait(timeout=600, interval=5)      # deadline includes sleeps and retries
task.save("video.mp4", index=0)         # streaming download, atomic rename
await task.asave("video.mp4")           # async variant, cancellation-safe

# webhooks: verify HMAC over the raw body with your api key
from routerai.webhooks import verify_video
data = verify_video(raw_body, signature, api_key, timestamp, max_age_seconds=300)
```

## Tools, structured output and cost

The model asks for a function, the SDK runs it and asks again — the schema is
derived from the signature, so what the model is told and what runs cannot
drift apart:

```python
def get_weather(city: str) -> str:
    """Узнать погоду в городе."""
    return f"в городе {city} +17"

answer = client.chat.run_tools(model, "Погода в Москве?", tools=[get_weather])
answer.content          # final reply
answer.runs             # what was executed, with arguments and results
```

A tool that raises is reported back to the model rather than crashing the
caller, and `max_turns` (5 by default) bounds what a looping model can spend.

Structured answers validate against your own model:

```python
class City(BaseModel):
    name: str
    population: int

answer = client.chat.parse(model, "Столица России?", response_model=City)
answer.parsed.population
```

Every request is priced in rubles, and the SDK adds it up:

```python
with client.track("ingest") as spent:
    client.chat.complete(model, prompt)
print(spent.cost_rub, spent.total_tokens)

client.usage.snapshot().by_model        # totals per model
client.on_usage(lambda record: metrics.observe(record))
```

Options can be set per call instead of per client, and the catalog can pick a
model for you:

```python
client.chat.complete(model, prompt, timeout=600, max_retries=0)
client.models.cheapest(capabilities=["tools"], min_context=100_000)
await client.models.asearch(q="claude")     # async twin, no blocked event loop
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

A closed side stays closed: calling into it raises `RuntimeError("client is
closed")` rather than quietly opening a fresh pool with default settings.
`close()` leaves the async side usable, and vice versa.

## Configuration

| Option | Description |
| --- | --- |
| `api_key` / `ROUTERAI_API_KEY` | API key (env var used when argument is None) |
| `base_url` / `ROUTERAI_BASE_URL` | base URL; precedence: explicit argument > env var > `https://routerai.ru/api/v1` |
| `timeout` | per-operation network inactivity timeout in seconds (default 60) |
| `max_retries` | retry attempts with exponential backoff + jitter (default 2) |
| `max_retry_after` | upper bound for an upstream `Retry-After` header, seconds (default 60) |
| `retry_unsafe_methods` | retry POST/PATCH/DELETE on 5xx too (default False; RouterAI already
  does provider fallback, a client-side POST retry may start a new billed generation) |
| `http_client` / `async_http_client` | inject external httpx transports (never closed by the library) |
| `default_headers` | headers added to every request (e.g. `{"X-Title": "my-app"}`) |
| `app_info` | appended to the SDK User-Agent, e.g. `"my-app/1.2"` |

Any call also accepts `timeout`, `max_retries` and `headers` for that one
request; they override the client-wide settings.

Retries honour the `Retry-After` header. Safe methods (GET/HEAD) are retried on
429/5xx; unsafe methods only on 429 by default.

Video polling propagates one deadline through sleeps, attempts and retry
backoff. Async polling actively cancels an in-flight refresh at the deadline.
For sync polling, HTTPX can only interrupt an in-flight socket operation using
its connect/read/write/pool inactivity timeouts; if that operation returns just
after the deadline, the SDK raises `DeadlineExceededError` before processing or
retrying the response.

The `extra` parameter is an escape hatch for provider-specific request fields:
it can never override library-managed keys (`model`, `messages`, `stream`, ...)
— colliding keys raise `ValueError`.

## Errors

Everything the server answers with is an `APIStatusError` or one of its
subclasses, so `except APIStatusError` catches the lot and `.status_code`
is always there:

```
RouterAIError
├─ APIStatusError            .status_code .http_status .provider_code .body
│  ├─ BadRequestError                400
│  ├─ AuthenticationError            401
│  ├─ InsufficientFundsError         402
│  ├─ PermissionDeniedError          403
│  ├─ NotFoundError                  404
│  ├─ ConflictError                  409
│  ├─ UnprocessableEntityError       422
│  ├─ RateLimitError                 429
│  └─ ServerError                    5xx
│     └─ NoProviderError             no provider could serve the model
├─ RequestError              transport failure after retries
│  └─ APIConnectionError     connection never established
│     └─ APITimeoutError     request timed out
├─ ResponseParsingError      the body was not the expected shape
├─ DeadlineExceededError     an absolute polling deadline passed (video `wait()`)
├─ StreamInterruptedError    SSE broke after the stream opened (`.chunks_received` may be 0)
├─ VideoGenerationError      a video task reached a terminal failure state
├─ WebhookVerificationError  webhook signature or freshness check failed
├─ ConfigurationError        the client was configured inconsistently
└─ ModelNotFoundError        no such model in the catalog
```

RouterAI reports upstream failures inside a successful HTTP response, with the
real code wrapped in a JSON string. The SDK unwraps that, so a provider rate
limit raises `RateLimitError` even though the transport said 200:

```python
try:
    client.chat.complete(model, prompt)
except RateLimitError as exc:
    exc.status_code    # 429 — the code that explains the failure
    exc.http_status    # 200 — what the transport actually said
    exc.status_source  # "provider"
```

## Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("routerai").setLevel(logging.DEBUG)  # API keys are masked
```

## License

MIT
