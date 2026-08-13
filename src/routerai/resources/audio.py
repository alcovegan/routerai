from __future__ import annotations

import base64
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from ..schemas import Usage

if TYPE_CHECKING:
    from .._http import HTTPClient


class Audio:
    """Audio generation, transcription and translation."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    # --- TTS ---

    def speech(
        self,
        model: str,
        input: str,
        *,
        voice: str | None = None,
        response_format: str = "mp3",
        speed: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SpeechResult:
        body: dict[str, Any] = {
            "model": model,
            "input": input,
            "response_format": response_format,
        }
        if voice:
            body["voice"] = voice
        if speed is not None:
            body["speed"] = speed
        if extra:
            body.update(extra)
        response = self._http.post("audio/speech", json=body)
        return SpeechResult(response.content, response.headers.get("X-Generation-Id"))

    async def aspeech(
        self,
        model: str,
        input: str,
        *,
        voice: str | None = None,
        response_format: str = "mp3",
        speed: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SpeechResult:
        body: dict[str, Any] = {
            "model": model,
            "input": input,
            "response_format": response_format,
        }
        if voice:
            body["voice"] = voice
        if speed is not None:
            body["speed"] = speed
        if extra:
            body.update(extra)
        response = await self._http.apost("audio/speech", json=body)
        return SpeechResult(response.content, response.headers.get("X-Generation-Id"))

    # --- STT ---

    def transcribe(
        self,
        model: str,
        file: str | Path | bytes | BinaryIO,
        *,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        extra: dict[str, Any] | None = None,
    ) -> TranscriptionResult:
        return self._transcribe(
            "audio/transcriptions", model, file, language, prompt, response_format, extra
        )

    async def atranscribe(
        self,
        model: str,
        file: str | Path | bytes | BinaryIO,
        *,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        extra: dict[str, Any] | None = None,
    ) -> TranscriptionResult:
        return await self._atranscribe(
            "audio/transcriptions", model, file, language, prompt, response_format, extra
        )

    def translate(
        self,
        model: str,
        file: str | Path | bytes | BinaryIO,
        *,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        extra: dict[str, Any] | None = None,
    ) -> TranscriptionResult:
        return self._transcribe(
            "audio/translations", model, file, language, prompt, response_format, extra
        )

    async def atranslate(
        self,
        model: str,
        file: str | Path | bytes | BinaryIO,
        *,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        extra: dict[str, Any] | None = None,
    ) -> TranscriptionResult:
        return await self._atranscribe(
            "audio/translations", model, file, language, prompt, response_format, extra
        )

    def _transcribe(
        self,
        path: str,
        model: str,
        file: str | Path | bytes | BinaryIO,
        language: str | None,
        prompt: str | None,
        response_format: str,
        extra: dict[str, Any] | None,
    ) -> TranscriptionResult:
        payload = self._stt_payload(model, file, language, prompt, response_format, extra)
        response = self._http.post(path, json=payload)
        return TranscriptionResult.from_response(
            response.json(), response.headers.get("X-Generation-Id")
        )

    async def _atranscribe(
        self,
        path: str,
        model: str,
        file: str | Path | bytes | BinaryIO,
        language: str | None,
        prompt: str | None,
        response_format: str,
        extra: dict[str, Any] | None,
    ) -> TranscriptionResult:
        payload = self._stt_payload(model, file, language, prompt, response_format, extra)
        response = await self._http.apost(path, json=payload)
        return TranscriptionResult.from_response(
            response.json(), response.headers.get("X-Generation-Id")
        )

    def _stt_payload(
        self,
        model: str,
        file: str | Path | bytes | BinaryIO,
        language: str | None,
        prompt: str | None,
        response_format: str,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "response_format": response_format}
        if isinstance(file, (str, Path)):
            path = Path(file)
            payload["file"] = {
                "filename": path.name,
                "file_data": base64.b64encode(path.read_bytes()).decode(),
            }
        elif hasattr(file, "read"):
            stream = file.read()
            assert isinstance(stream, bytes)
            payload["file"] = {
                "filename": getattr(file, "name", "audio.bin") or "audio.bin",
                "file_data": base64.b64encode(stream).decode(),
            }
        else:
            assert isinstance(file, bytes)
            payload["file"] = {
                "filename": "audio.bin",
                "file_data": base64.b64encode(file).decode(),
            }
        if language:
            payload["language"] = language
        if prompt:
            payload["prompt"] = prompt
        if extra:
            payload.update(extra)
        return payload


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
