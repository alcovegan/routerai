# routerai

[English version](README.md)

Python-обёртка для API [RouterAI](https://routerai.ru) — единый доступ к более чем 450 моделям ИИ
(OpenAI, Anthropic, Google, DeepSeek, Qwen и другим) с ценами в рублях.

## Возможности

- OpenAI-совместимые чат-комплишены с синхронным и асинхронным режимами (один экземпляр
  клиента поддерживает оба режима; транспорты разделены)
- Разобранные ответы: контент, рассуждения, вызовы инструментов, альтернативы, расход токенов
  и стоимость в рублях (`Decimal`)
- Потоковая передача (SSE) с дельтами в каждом чанке; после открытия успешного потока ответа
  автоматические повторы не выполняются (даже если не был получен ни один чанк), а обрыв
  потока вызывает типизированную ошибку `StreamInterruptedError`
- Вызовы инструментов в потоке приходят собранными: `StreamAccumulator` склеивает
  дельты по индексу, поэтому `json.loads(call["function"]["arguments"])` просто работает
- Типизированные ошибки не спорят сами с собой: любой ответ сервера — наследник
  `APIStatusError` с `.status_code`, а отказ провайдера, завёрнутый в HTTP 200,
  всё равно поднимает `RateLimitError`
- Каталог моделей: получение списка, клиентский поиск и группировка по возможностям (текст,
  рассуждения, зрение, генерация изображений, видео и аудио, синтез и распознавание речи,
  эмбеддинги, реранкинг, инструменты)
- Получение стоимости по идентификатору генерации (`X-Generation-Id`) после выполнения запроса
- Несколько API-ключей через отдельный для каждого экземпляра `Registry` (на основе
  `contextvar`, безопасен для потоков и асинхронного кода)
- Изображения, видео (с опросом статуса), аудио (TTS/STT), эмбеддинги, реранкинг, управление
  API-ключами и командами
- Логирование без дополнительной настройки через стандартный модуль `logging` (пространство
  имён `routerai`, ключи маскируются)

## Установка

```bash
pip install routerai
```

## Быстрый старт

```python
from routerai import RouterAI

client = RouterAI(api_key="sk-...")  # либо задайте переменную окружения ROUTERAI_API_KEY

result = client.chat.complete("deepseek/deepseek-v4-pro", "Привет!")
print(result.content)
print(result.cost_rub)  # Decimal, стоимость в рублях
```

### Каталог моделей

```python
client.models.all()                       # полный каталог (по умолчанию кешируется на 10 минут)
client.models.search("claude", capabilities=["reasoning"], min_context=100_000)
client.models.by_capability("image")      # модели для генерации изображений
client.models.grouped()                   # dict[Capability, list[Model]]
client.models.get("deepseek/deepseek-v4-pro").pricing.per_million("prompt")

# Не всё тарифицируется по токенам: картинки — за изображение, rerank —
# за поисковую единицу, видео — за секунду. Спрашивайте, за что модель берёт
# деньги, вместо того чтобы читать нулевую цену токена как «бесплатно».
pricing = client.models.get("black-forest-labs/flux.2-pro").pricing
pricing.priced_units()              # {"image_output"}
pricing.price("image_output")       # Decimal за изображение
pricing.is_free()                   # False

# Поэтому фильтр по цене пропускает модели с другой единицей тарификации,
# а не ставит их первыми: они не бесплатны, их цена просто в другой валюте.
client.models.search(max_price_prompt=1.0)
client.models.endpoints("anthropic/claude-sonnet-5")  # провайдеры и цены
```

### Несколько API-ключей

```python
from routerai import RouterAI, Registry

registry = Registry(main=RouterAI(api_key=A), personal=RouterAI(api_key=B))
registry["personal"].chat.complete(...)
with registry.using("main"):
    ...
```

### Потоковая передача

```python
for chunk in client.chat.stream("openai/gpt-5.6-sol", "Расскажи сказку"):
    print(chunk.content, end="")
```

### Распознавание речи

```python
client.audio.transcribe("openai/whisper-large-v3", "voice.wav")      # формат из расширения файла
client.audio.transcribe("openai/whisper-large-v3", raw_bytes, format="mp3")
for chunk in client.audio.speech_stream("x-ai/grok-voice-tts-1.0", "текст", voice="eve"):
    ...
```

### Жизненный цикл видео

```python
from routerai import FrameImage, ImageReference

task = client.videos.create(
    "bytedance/seedance-2.0",
    "Персонаж идёт через осенний лес",
    frame_images=[FrameImage(url="https://example.com/first.png", frame_type="first_frame")],
    # либо генерация видео по референсу: input_references=[ImageReference(url=...)]
)
task.wait(timeout=600, interval=5)      # дедлайн включает ожидание и повторные запросы
task.save("video.mp4", index=0)         # потоковая загрузка с атомарным переименованием
await task.asave("video.mp4")           # асинхронный вариант с безопасной отменой

# Вебхуки: проверка HMAC по исходному телу запроса с помощью API-ключа
from routerai.webhooks import verify_video

# Читайте заголовки регистронезависимо: прокси меняют регистр, а штатный
# доступ к заголовкам во фреймворках это уже учитывает.
signature = request.headers["X-RouterAI-Signature"]
timestamp = request.headers["X-RouterAI-Timestamp"]
data = verify_video(raw_body, signature, api_key, timestamp, max_age_seconds=300)
```

Неверная подпись поднимает `WebhookVerificationError` — в том числе когда в
заголовке пришли байты вне ASCII: обработчик отвечает 401, а не 500.

Запуск генерации требует доступного баланса заметно выше стоимости ролика
(RouterAI резервирует сумму на время рендера и списывает фактическую), поэтому
при нехватке средств приходит `InsufficientFundsError` ещё до генерации.

## Инструменты, структурированный вывод и расходы

Модель просит функцию — SDK её вызывает и спрашивает снова. Схема берётся из
сигнатуры, поэтому описание для модели и реально вызываемый код не разъезжаются:

```python
def get_weather(city: str) -> str:
    """Узнать погоду в городе."""
    return f"в городе {city} +17"

answer = client.chat.run_tools(model, "Погода в Москве?", tools=[get_weather])
answer.content          # итоговый ответ
answer.runs             # что было выполнено, с аргументами и результатами
```

Упавший инструмент возвращается модели как результат, а не рушит вызов;
`max_turns` (по умолчанию 5) ограничивает траты, если модель зациклилась.

Структурированные ответы проверяются вашей же моделью:

```python
class City(BaseModel):
    name: str
    population: int

answer = client.chat.parse(model, "Столица России?", response_model=City)
answer.parsed.population
```

Стоимость каждого запроса приходит в рублях, и SDK её суммирует:

```python
with client.track("ingest") as spent:
    client.chat.complete(model, prompt)
print(spent.cost_rub, spent.total_tokens)

client.usage.snapshot().by_model        # итоги по моделям
client.on_usage(lambda record: metrics.observe(record))
```

Опции задаются на вызов, а не только на клиент, а модель можно выбрать
по цене:

```python
client.chat.complete(model, prompt, timeout=600, max_retries=0)
client.models.cheapest(capabilities=["tools"], min_context=100_000)
await client.models.asearch(q="claude")     # асинхронный близнец, цикл не блокируется
```

## Асинхронный режим

```python
result = await client.chat.acomplete("deepseek/deepseek-v4-pro", "Привет!")
async for chunk in client.chat.astream(...):
    ...
await client.aclose()
```

Синхронный и асинхронный транспорты хранятся раздельно, поэтому один экземпляр клиента
можно использовать в обоих режимах. Учитывайте жизненный цикл: `close()` закрывает пул
синхронных соединений, а `await aclose()` — асинхронный. Если один экземпляр использовался
в обоих режимах, вызовите оба метода. Внешние транспорты, переданные через
`http_client`/`async_http_client`, библиотека никогда не закрывает.

Закрытая сторона остаётся закрытой: обращение к ней поднимает
`RuntimeError("client is closed")`, а не открывает молча новый пул с настройками
по умолчанию. При этом `close()` не мешает работе асинхронной стороны, и наоборот.

## Конфигурация

| Параметр | Описание |
| --- | --- |
| `api_key` / `ROUTERAI_API_KEY` | API-ключ (переменная окружения используется, если аргумент равен `None`) |
| `base_url` / `ROUTERAI_BASE_URL` | Базовый URL; приоритет: явный аргумент > переменная окружения > `https://routerai.ru/api/v1` |
| `timeout` | Тайм-аут отсутствия сетевой активности для одной операции в секундах (по умолчанию 60) |
| `max_retries` | Число повторных попыток с экспоненциальной задержкой и джиттером (по умолчанию 2) |
| `max_retry_after` | Верхний предел значения заголовка `Retry-After` в секундах (по умолчанию 60) |
| `retry_unsafe_methods` | Повторять POST/PATCH/DELETE и при ответах 5xx (по умолчанию False; RouterAI уже выполняет переключение между провайдерами, а клиентский повтор POST может запустить новую платную генерацию) |
| `http_client` / `async_http_client` | Внешние транспорты httpx (библиотека никогда их не закрывает) |
| `default_headers` | заголовки, добавляемые к каждому запросу (например, `{"X-Title": "my-app"}`) |
| `app_info` | добавка к User-Agent SDK, например `"my-app/1.2"` |

Любой вызов дополнительно принимает `timeout`, `max_retries` и `headers`
для одного этого запроса — они перекрывают клиентские настройки.

Повторные попытки учитывают заголовок `Retry-After`. Безопасные методы (GET/HEAD)
повторяются при ответах 429/5xx; небезопасные методы по умолчанию — только при 429.

При опросе статуса видео единый дедлайн распространяется на ожидание, попытки и задержки
между повторами. Асинхронный опрос активно отменяет выполняющийся запрос при достижении
дедлайна. В синхронном опросе HTTPX может прервать выполняющуюся операцию с сокетом только
по тайм-аутам отсутствия активности для подключения, чтения, записи или пула. Если операция
завершится сразу после дедлайна, SDK вызовет `DeadlineExceededError` до обработки ответа или
повторной попытки.

Параметр `extra` позволяет передавать специфичные для провайдера поля запроса, но не может
переопределять поля, которыми управляет библиотека (`model`, `messages`, `stream` и другие):
при конфликте ключей вызывается `ValueError`.

## Ошибки

Всё, чем ответил сервер, — это `APIStatusError` или его наследник, поэтому
`except APIStatusError` ловит любую такую ошибку, и `.status_code` есть всегда:

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
│     └─ NoProviderError             ни один провайдер не смог обслужить модель
├─ RequestError              транспортная ошибка после исчерпания повторов
│  └─ APIConnectionError     соединение не было установлено
│     └─ APITimeoutError     истёк таймаут запроса
├─ ResponseParsingError      тело ответа оказалось не той формы
├─ DeadlineExceededError     превышен абсолютный дедлайн опроса (видео `wait()`)
├─ StreamInterruptedError    SSE оборвался после открытия потока (`.chunks_received` может быть 0)
├─ VideoGenerationError      задача видео перешла в терминальное состояние отказа
├─ WebhookVerificationError  не пройдена проверка подписи или свежести вебхука
├─ ConfigurationError        клиент сконфигурирован противоречиво
└─ ModelNotFoundError        такой модели нет в каталоге
```

RouterAI сообщает об отказах провайдера внутри успешного HTTP-ответа, пряча
настоящий код в JSON-строке. SDK её разворачивает, поэтому лимит провайдера
поднимает `RateLimitError`, хотя транспорт сказал 200:

```python
try:
    client.chat.complete(model, prompt)
except RateLimitError as exc:
    exc.status_code    # 429 — код, объясняющий отказ
    exc.http_status    # 200 — что на самом деле сказал транспорт
    exc.status_source  # "provider"
```

## Логирование

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("routerai").setLevel(logging.DEBUG)  # API-ключи маскируются
```

## Лицензия

MIT
