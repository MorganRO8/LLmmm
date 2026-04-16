from __future__ import annotations

import io
import json
import logging
import wave
from typing import Any

from openai import AsyncOpenAI

from app.config import AppConfig
from app.core.models import Recipe
from app.realtime.prompts import build_system_prompt
from app.realtime.tools import TOOL_DEFINITIONS

logger = logging.getLogger(__name__)


class ResponsesBackendClient:
    def __init__(self, config: AppConfig, recipe: Recipe, tool_executor) -> None:  # noqa: ANN001
        self._config = config
        self._recipe = recipe
        self._tool_executor = tool_executor
        self._client = AsyncOpenAI(api_key=config.openai_api_key)
        self._previous_response_id: str | None = None
        self._instructions = build_system_prompt(recipe)

    @staticmethod
    def _pcm16_to_wav_bytes(pcm16_bytes: bytes, sample_rate: int, channels: int) -> bytes:
        bio = io.BytesIO()
        with wave.open(bio, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16_bytes)
        return bio.getvalue()

    async def transcribe_pcm16(self, pcm16_bytes: bytes, sample_rate: int, channels: int) -> str:
        wav_bytes = self._pcm16_to_wav_bytes(pcm16_bytes, sample_rate, channels)
        bio = io.BytesIO(wav_bytes)
        bio.name = 'speech.wav'
        result = await self._client.audio.transcriptions.create(
            model=self._config.transcription_model,
            file=bio,
        )
        text = getattr(result, 'text', '') or ''
        return text.strip()

    async def run_turn(
        self,
        user_text: str,
        frames: list[dict[str, Any]] | None = None,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_text}]
        for frame in frames or []:
            b64 = frame.get('jpeg_bytes_base64')
            if b64:
                content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})

        response = await self._client.responses.create(
            model=self._config.model_name,
            instructions=self._instructions,
            previous_response_id=self._previous_response_id,
            tools=TOOL_DEFINITIONS,
            input=[{"role": "user", "content": content}],
        )
        return await self._resolve_response(response)

    async def bootstrap(self) -> str:
        response = await self._client.responses.create(
            model=self._config.model_name,
            instructions=self._instructions,
            tools=TOOL_DEFINITIONS,
            input="Greet the user briefly and offer the first recipe step.",
        )
        return await self._resolve_response(response)

    async def _resolve_response(self, response) -> str:  # noqa: ANN001
        while True:
            data = response.model_dump() if hasattr(response, 'model_dump') else response
            response_id = data.get('id')
            outputs = data.get('output', [])
            assistant_text_parts: list[str] = []
            followup_input: list[dict[str, Any]] = []

            for item in outputs:
                item_type = item.get('type')
                if item_type == 'message':
                    for content in item.get('content', []):
                        if content.get('type') == 'output_text':
                            text = content.get('text', '')
                            if text:
                                assistant_text_parts.append(text)
                elif item_type == 'function_call':
                    name = item.get('name', '')
                    raw_args = item.get('arguments') or '{}'
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                    result = await self._tool_executor.execute(name, args)
                    call_id = item.get('call_id', '')
                    followup_input.append({
                        'type': 'function_call_output',
                        'call_id': call_id,
                        'output': json.dumps(result),
                    })

                    if name == 'capture_context_frames' and result.get('ok'):
                        payload = result.get('result', {})
                        reason = payload.get('reason', 'visual context')
                        frames = payload.get('frames', [])
                        if frames:
                            content = [{
                                'type': 'input_text',
                                'text': f'Visual context for {reason}. Use these images in your answer.'
                            }]
                            for frame in frames:
                                b64 = frame.get('jpeg_bytes_base64')
                                if b64:
                                    content.append({
                                        'type': 'input_image',
                                        'image_url': f'data:image/jpeg;base64,{b64}',
                                    })
                            followup_input.append({'role': 'user', 'content': content})

            if followup_input:
                response = await self._client.responses.create(
                    model=self._config.model_name,
                    instructions=self._instructions,
                    previous_response_id=response_id,
                    tools=TOOL_DEFINITIONS,
                    input=followup_input,
                )
                continue

            self._previous_response_id = response_id
            final_text = '\n'.join(part.strip() for part in assistant_text_parts if part and part.strip()).strip()
            return final_text
