# routerai

Python wrapper for the [RouterAI](https://routerai.ru) API — unified access to 400+ AI models
(OpenAI, Anthropic, Google, DeepSeek, Qwen, ...) with ruble pricing.

## Features

- OpenAI-compatible chat completions with sync + async support
- Parsed responses: content, reasoning, tool calls, token usage and cost in rubles
- Streaming (SSE) with per-chunk deltas
- Models catalog: listing, client-side search, grouping by capabilities (text, reasoning,
  vision, image/video generation, speech, transcription, embeddings, rerank, tools)
- Post-hoc cost lookup by generation id
- Multiple API keys via a named registry
- Images, video, audio (TTS/STT), embeddings, rerank, API-key and team management
- Zero-effort logging via the standard `logging` module (namespace `routerai`)

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
print(result.usage.cost_rub)  # Decimal, in rubles
```

### Model catalog

```python
client.models.search("claude", capabilities=["reasoning"], min_context=100_000)
client.models.by_capability("image")       # image generation models
client.models.grouped()                    # dict[Capability, list[Model]]
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

### Logging

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("routerai").setLevel(logging.DEBUG)  # API keys are masked
```

## Async

```python
result = await client.chat.acomplete("deepseek/deepseek-v4-pro", "Привет!")
async for chunk in client.chat.astream(...):
    ...
```

## License

MIT
