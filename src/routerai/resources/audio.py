from __future__ import annotations

import asyncio
import base64
import os
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from .._extras import merge_extra
from ..schemas import Usage

if TYPE_CHECKING:
    from .._http import HTTPClient

SUPPORTED_AUDIO_FORMATS = {"mp3", "wav", "flac", "m4a", "ogg", "webm", "aac"}
_SUFFIX_FORMATS = {"mpeg": "mp3", "wave": "wav", "oga": "ogg", "mp4": "m4a"}


def _format_from_name(name: object) -> str | None:
    if not isinstance(name, (str, os.PathLike)):
        return None
    suffix = Path(name).suffix.lstrip(".").lower()
    if not suffix:
        return None
    if suffix in SUPPORTED_AUDIO_FORMATS:
        return suffix
    return _SUFFIX_FORMATS.get(suffix)


class Audio:
    """Audio generation (TTS) and transcription (STT).

    STT uses the documented RouterAI JSON contract
    ``{"input_audio": {"data": base64, "format": "..."}}``; the format is
    inferred from the file name and must be passed explicitly for raw bytes.
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    # --- TTS ---

    def speech(
        self,
        model: str,
        input: str,
        voice: str,
        *,
        response_format: str = "mp3",
        speed: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SpeechResult:
        response = self._http.post(
            "audio/speech", json=self._tts_body(model, input, voice, response_format, speed, extra)
        )
        return SpeechResult(response.content, response.headers.get("X-Generation-Id"))

    async def aspeech(
        self,
        model: str,
        input: str,
        voice: str,
        *,
        response_format: str = "mp3",
        speed: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SpeechResult:
        response = await self._http.apost(
            "audio/speech", json=self._tts_body(model, input, voice, response_format, speed, extra)
        )
        return SpeechResult(response.content, response.headers.get("X-Generation-Id"))

    def _tts_body(
        self,
        model: str,
        input: str,
        voice: str,
        response_format: str,
        speed: float | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
        }
        if speed is not None:
            body["speed"] = speed
        merge_extra(extra, reserved=("model", "input", "voice", "response_format"))
        if extra:
            body.update(extra)
        return body

    # --- STT ---

    def transcribe(
        self,
        model: str,
        file: str | Path | bytes | BinaryIO,
        *,
        format: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        timestamp_granularities: list[str] | None = None,
        temperature: float | None = None,
        provider: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> TranscriptionResult:
        payload = self._stt_payload(
            model,
            file,
            format,
            language,
            prompt,
            response_format,
            timestamp_granularities,
            temperature,
            provider,
            extra,
        )
        response = self._http.post("audio/transcriptions", json=payload)
        return TranscriptionResult.from_response(
            self._http._json(response), response.headers.get("X-Generation-Id")
        )

    async def atranscribe(
        self,
        model: str,
        file: str | Path | bytes | BinaryIO,
        *,
        format: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        timestamp_granularities: list[str] | None = None,
        temperature: float | None = None,
        provider: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> TranscriptionResult:
        payload = await asyncio.to_thread(
            self._stt_payload,
            model,
            file,
            format,
            language,
            prompt,
            response_format,
            timestamp_granularities,
            temperature,
            provider,
            extra,
        )
        response = await self._http.apost("audio/transcriptions", json=payload)
        return TranscriptionResult.from_response(
            self._http._json(response), response.headers.get("X-Generation-Id")
        )

    def _stt_payload(
        self,
        model: str,
        file: str | Path | bytes | BinaryIO,
        format: str | None,
        language: str | None,
        prompt: str | None,
        response_format: str,
        timestamp_granularities: list[str] | None,
        temperature: float | None,
        provider: dict[str, Any] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        data, name = _read_audio(file)
        audio_format: str | None
        if format is not None:
            audio_format = format.lower()
        else:
            audio_format = _format_from_name(name)
            if audio_format is None:
                raise ValueError(
                    "cannot infer audio format from the file name; pass format= explicitly "
                    f"(one of {sorted(SUPPORTED_AUDIO_FORMATS)})"
                )
        if audio_format not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"unsupported audio format {audio_format!r}; supported: {sorted(SUPPORTED_AUDIO_FORMATS)}"
            )

        payload: dict[str, Any] = {
            "model": model,
            "input_audio": {
                "data": base64.b64encode(data).decode("ascii"),
                "format": audio_format,
            },
            "response_format": response_format,
        }
        if language:
            payload["language"] = language
        if prompt:
            payload["prompt"] = prompt
        if timestamp_granularities:
            payload["timestamp_granularities"] = timestamp_granularities
        if temperature is not None:
            payload["temperature"] = temperature
        if provider:
            payload["provider"] = provider
        merge_extra(extra, reserved=("model", "input_audio", "response_format"))
        if extra:
            payload.update(extra)
        return payload


def _read_audio(file: str | Path | bytes | BinaryIO) -> tuple[bytes, str | None]:
    if isinstance(file, (str, Path)):
        path = Path(file)
        return path.read_bytes(), path.name
    if isinstance(file, bytes):
        return file, None
    name = getattr(file, "name", None)
    stream = file.read()
    assert isinstance(stream, bytes)
    return stream, name


class SpeechResult:
    def __init__(self, data: bytes, generation_id: str | None = None) -> None:
        self.data = data
        self.generation_id = generation_id

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)
        return path


class TranscriptionResult:
    def __init__(
        self,
        text: str,
        usage: Usage | None,
        generation_id: str | None,
        raw: dict[str, Any],
    ) -> None:
        self.text = text
        self.usage = usage
        self.generation_id = generation_id
        self.raw = raw

    @classmethod
    def from_response(
        cls, payload: dict[str, Any], generation_id: str | None
    ) -> TranscriptionResult:
        text = payload.get("text") or ""
        if isinstance(payload.get("data"), dict):
            text = payload["data"].get("text") or text
        usage = Usage.model_validate(payload["usage"]) if payload.get("usage") else None
        return cls(text, usage, generation_id, payload)

    @property
    def cost_rub(self) -> Decimal | None:
        return self.usage.cost_rub if self.usage else None
