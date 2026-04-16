from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI

from app.config import AppConfig
from app.core.models import Recipe
from app.realtime.session_config import build_session_config

logger = logging.getLogger(__name__)


class RealtimeClient:
    def __init__(
        self,
        config: AppConfig,
        recipe: Recipe,
        on_assistant_text: Callable[[str], Awaitable[None]],
        on_assistant_audio: Callable[[bytes], Awaitable[None]],
        on_function_call: Callable[[str, str, dict[str, Any]], Awaitable[None]],
        on_user_transcript: Callable[[str], Awaitable[None]],
        on_status: Callable[[dict[str, Any]], Awaitable[None]],
        on_error: Callable[[str], Awaitable[None]],
    ) -> None:
        self._config = config
        self._recipe = recipe
        self._on_assistant_text = on_assistant_text
        self._on_assistant_audio = on_assistant_audio
        self._on_function_call = on_function_call
        self._on_user_transcript = on_user_transcript
        self._on_status = on_status
        self._on_error = on_error
        self._client = AsyncOpenAI(api_key=config.openai_api_key)
        self._connection = None
        self._listener_task: asyncio.Task | None = None
        self._connected = False
        self._response_active = False

        # Accumulate transcript deltas for assistant audio output.
        # Keyed by (response_id, item_id, content_index).
        self._assistant_transcript_buffers: dict[tuple[str, str, int], str] = {}

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def response_active(self) -> bool:
        return self._response_active

    async def connect(self) -> None:
        if self._connected:
            return
        async with self._client.realtime.connect(model=self._config.model_name) as connection:
            self._connection = connection
            await connection.session.update(session=build_session_config(self._config, self._recipe))
            self._connected = True
            await self._on_status({"connected": True, "model_name": self._config.model_name})
            self._listener_task = asyncio.create_task(self._listen())
            await self._send_bootstrap_message()
            try:
                await self._listener_task
            finally:
                self._connected = False
                self._response_active = False
                await self._on_status({"connected": False, "model_name": self._config.model_name})
                self._connection = None

    async def _send_bootstrap_message(self) -> None:
        if self._connection is None:
            return
        await self._connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Greet the user briefly and offer the first step."}],
            }
        )
        await self._connection.response.create()
        self._response_active = True

    async def _listen(self) -> None:
        assert self._connection is not None
        async for event in self._connection:
            event_data = event.model_dump() if hasattr(event, "model_dump") else event
            event_type = event_data.get("type")

            if event_type == "error":
                error = event_data.get("error", {})
                await self._on_error(error.get("message", "Unknown Realtime error"))
                continue

            if event_type == "session.created":
                session = event_data.get("session", {})
                await self._on_status({
                    "connected": True,
                    "session_id": session.get("id"),
                    "model_name": session.get("model", self._config.model_name),
                })
                continue

            if event_type == "input_audio_buffer.speech_started":
                await self._on_status({"user_speaking": True})
                continue

            if event_type == "input_audio_buffer.speech_stopped":
                await self._on_status({"user_speaking": False})
                continue

            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event_data.get("transcript", "")
                if transcript:
                    await self._on_user_transcript(transcript)
                continue

            if event_type == "response.created":
                self._response_active = True
                await self._on_status({"assistant_speaking": True})
                continue

            if event_type in {"response.done", "response.completed", "response.cancelled"}:
                self._response_active = False
                await self._on_status({"assistant_speaking": False})
                continue

            if event_type == "response.output_audio.delta":
                delta = event_data.get("delta", "")
                if delta:
                    try:
                        await self._on_assistant_audio(base64.b64decode(delta))
                    except Exception:
                        logger.exception("Failed to decode output audio delta")
                continue

            if event_type == "response.output_audio.done":
                await self._on_status({"assistant_speaking": False})
                continue

            # In audio mode, use the audio transcript stream for text display.
            if event_type == "response.output_audio_transcript.delta":
                key = (
                    event_data.get("response_id", ""),
                    event_data.get("item_id", ""),
                    int(event_data.get("content_index", 0)),
                )
                delta = event_data.get("delta", "")
                if delta:
                    self._assistant_transcript_buffers[key] = self._assistant_transcript_buffers.get(key, "") + delta
                continue

            if event_type == "response.output_audio_transcript.done":
                key = (
                    event_data.get("response_id", ""),
                    event_data.get("item_id", ""),
                    int(event_data.get("content_index", 0)),
                )
                final_text = event_data.get("transcript") or self._assistant_transcript_buffers.get(key, "")
                final_text = (final_text or "").strip()
                if final_text:
                    await self._on_assistant_text(final_text)
                self._assistant_transcript_buffers.pop(key, None)
                continue

            # Fallback for text-mode sessions.
            if event_type == "response.output_text.done":
                text = (event_data.get("text") or "").strip()
                if text:
                    await self._on_assistant_text(text)
                continue

            if event_type == "response.function_call_arguments.done":
                raw_args = event_data.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
                await self._on_function_call(event_data.get("call_id", ""), event_data.get("name", ""), args)
                continue

    async def send_audio_chunk(self, pcm16_bytes: bytes) -> None:
        if not self._connected or self._connection is None:
            return
        await self._connection.input_audio_buffer.append(audio=base64.b64encode(pcm16_bytes).decode("ascii"))

    async def send_text_message(self, text: str) -> None:
        if not self._connected or self._connection is None:
            return
        await self._connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
        )
        await self._connection.response.create()
        self._response_active = True

    async def send_function_result(self, call_id: str, result: dict[str, Any]) -> None:
        if not self._connected or self._connection is None:
            return
        await self._connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            }
        )
        await self._connection.response.create()
        self._response_active = True

    async def send_context_images(self, frames: list[dict[str, Any]], reason: str) -> None:
        if not self._connected or self._connection is None or not frames:
            return

        content = [{"type": "input_text", "text": f"Visual context requested: {reason}. Use these recent images."}]
        for frame in frames:
            b64 = frame.get("jpeg_bytes_base64")
            if b64:
                content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})

        await self._connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": content,
            }
        )
        await self._connection.response.create()
        self._response_active = True

    async def cancel_response(self) -> None:
        if not self._connected or self._connection is None or not self._response_active:
            return
        try:
            await self._connection.response.cancel()
        except Exception:
            logger.exception("Failed to cancel Realtime response")
