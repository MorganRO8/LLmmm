from __future__ import annotations

import asyncio
import base64
import logging
import math
import queue
import re
import threading
import time
from concurrent.futures import Future
from typing import Any

import cv2
import numpy as np

from app.audio.input_stream import AudioInputStream
from app.audio.playback_controller import PlaybackController
from app.audio.utterance_detector import UtteranceDetector
from app.config import AppConfig
from app.core.bus import EventBus
from app.core.events import TimerExpiredEvent, TimerNotifyEvent, TimerWarningEvent
from app.core.recipe import RecipeLoader
from app.core.state import AppStateStore
from app.gestures.service import GestureService
from app.responses.client import ResponsesBackendClient
from app.realtime.tools import ToolExecutor
from app.timers.manager import TimerManager
from app.tts.kokoro_tts import KokoroTTS
from app.vision.camera_service import CameraService

logger = logging.getLogger(__name__)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class BackendLoopThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="backend-loop", daemon=True)
        self.loop = asyncio.new_event_loop()

    def run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)

    def submit(self, coro) -> Future:  # noqa: ANN001
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


class DesktopController:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.recipe = RecipeLoader.load(self.config.recipe_path)
        self.state_store = AppStateStore(self.recipe)
        self.event_bus = EventBus()

        self.camera_service = CameraService(
            camera_index=self.config.camera_index,
            preview_width=self.config.preview_width,
            preview_height=self.config.preview_height,
            context_width=self.config.context_frame_width,
            context_height=self.config.context_frame_height,
            context_sample_interval_ms=self.config.context_sample_interval_ms,
            context_buffer_maxlen=self.config.context_buffer_maxlen,
        )
        self.timer_manager = TimerManager(self.state_store.state, self.event_bus)
        self.tool_executor = ToolExecutor(self.state_store, self.timer_manager, self.camera_service, self.config)
        self.playback = PlaybackController(
            samplerate=self.config.audio_sample_rate,
            channels=self.config.audio_channels,
            blocksize=self.config.audio_blocksize,
        )
        self.utterance_detector = UtteranceDetector(
            sample_rate=self.config.audio_sample_rate,
            channels=self.config.audio_channels,
            vad_threshold=self.config.local_vad_threshold,
            min_speech_ms=self.config.local_vad_min_speech_ms,
            silence_ms=self.config.local_vad_silence_ms,
            max_utterance_ms=self.config.local_vad_max_utterance_ms,
        )
        self.kokoro = KokoroTTS(
            model_dir=self.config.kokoro_model_dir,
            repo_id=self.config.kokoro_repo_id,
            voice=self.config.kokoro_voice,
            lang_code=self.config.kokoro_lang_code,
            speed=self.config.kokoro_speed,
        )
        self._backend = BackendLoopThread()
        self._ui_events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._audio_input: AudioInputStream | None = None
        self._backend_client: ResponsesBackendClient | None = None
        self._started = False
        self._turn_in_flight = False
        self._last_playback_end_ts = 0.0
        self._mic_suppressed_until = 0.0

        self._gesture_service = GestureService(
            camera_service=self.camera_service,
            confidence_threshold=self.config.gesture_confidence_threshold,
            hold_ms=self.config.gesture_hold_ms,
            cooldown_ms=self.config.gesture_cooldown_ms,
            poll_interval_ms=self.config.gesture_poll_interval_ms,
            on_gesture=self._on_gesture,
        )

        self.event_bus.subscribe(TimerWarningEvent, self._on_timer_warning)
        self.event_bus.subscribe(TimerNotifyEvent, self._on_timer_notify)
        self.event_bus.subscribe(TimerExpiredEvent, self._on_timer_expired)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.camera_service.start()
        self.playback.start()
        self._backend.start()
        self._backend.submit(self.timer_manager.start()).result(timeout=5)
        self._gesture_service.start()
        self.state_store.add_transcript("system", f"Loaded recipe: {self.recipe.name}")

        if self.config.openai_api_key and self.config.auto_connect_backend:
            self._start_backend()
        else:
            self._push_ui_event({"type": "status", "message": "Backend disabled (missing OPENAI_API_KEY or auto-connect off)."})

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._gesture_service.stop()
        self.camera_service.stop()

        if self._audio_input is not None:
            self._audio_input.stop()
            self._audio_input = None

        self.playback.stop()
        try:
            self._backend.submit(self.timer_manager.stop()).result(timeout=5)
        except Exception:
            logger.exception("Failed to stop timer manager cleanly")
        self._backend.stop()

    def _start_backend(self) -> None:
        self._backend_client = ResponsesBackendClient(
            config=self.config,
            recipe=self.recipe,
            tool_executor=self.tool_executor,
        )
        self.state_store.update_realtime_status(connected=True, model_name=self.config.model_name)
        self._push_ui_event({"type": "status", "message": f"Connected to Responses backend ({self.config.model_name})."})

        self._audio_input = AudioInputStream(
            samplerate=self.config.audio_sample_rate,
            channels=self.config.audio_channels,
            blocksize=self.config.audio_blocksize,
            callback=self._on_audio_input,
            queue_maxsize=self.config.audio_input_queue_maxsize,
        )
        self._audio_input.start()

        self._backend.submit(self._bootstrap_assistant())

    def _on_audio_input(self, audio_bytes: bytes) -> None:
        if not self._backend_client or not self.state_store.state.features.speech_enabled:
            return

        now = time.monotonic()
        blocked = (
            self.state_store.state.realtime.assistant_speaking
            or self._turn_in_flight
            or now < self._mic_suppressed_until
        )
        if not blocked and (now - self._last_playback_end_ts) * 1000 < self.config.post_tts_mic_block_ms:
            blocked = True

        utterance = self.utterance_detector.process_chunk(audio_bytes, blocked=blocked)
        if utterance and not self._turn_in_flight:
            self._turn_in_flight = True
            self.state_store.update_realtime_status(user_speaking=False)
            self._backend.submit(self._handle_audio_turn(utterance))
        elif not blocked:
            self.state_store.update_realtime_status(user_speaking=self.utterance_detector._recording)

    async def _bootstrap_assistant(self) -> None:
        try:
            self._push_ui_event({"type": "status", "message": "Preparing local Kokoro TTS assets..."})
            await asyncio.to_thread(self.kokoro.ensure_ready)
            text = await self._backend_client.bootstrap() if self._backend_client else ""
            if text:
                await self._deliver_assistant_text(text)
        except Exception as exc:
            logger.exception("Bootstrap failed")
            await self._async_on_error(str(exc))

    async def _handle_audio_turn(self, pcm16_bytes: bytes) -> None:
        try:
            if not self._backend_client:
                return
            transcript = await self._backend_client.transcribe_pcm16(
                pcm16_bytes,
                sample_rate=self.config.audio_sample_rate,
                channels=self.config.audio_channels,
            )
            transcript = transcript.strip()
            if not transcript:
                return
            await self._async_on_user_transcript(transcript)
            reply = await self._backend_client.run_turn(transcript)
            if reply:
                await self._deliver_assistant_text(reply)
        except Exception as exc:
            logger.exception("Audio turn failed")
            await self._async_on_error(str(exc))
        finally:
            self._turn_in_flight = False
            self.state_store.update_realtime_status(user_speaking=False)

    async def _handle_text_turn(self, text: str, frames: list[dict[str, Any]] | None = None) -> None:
        try:
            if not self._backend_client:
                return
            reply = await self._backend_client.run_turn(text, frames=frames)
            if reply:
                await self._deliver_assistant_text(reply)
        except Exception as exc:
            logger.exception("Text turn failed")
            await self._async_on_error(str(exc))
        finally:
            self._turn_in_flight = False

    async def _deliver_assistant_text(self, text: str) -> None:
        await self._async_on_assistant_text(text)
        if not self.state_store.state.features.tts_enabled:
            return
        self.state_store.update_realtime_status(assistant_speaking=True)
        for chunk in self._split_tts_chunks(text):
            pcm16 = await asyncio.to_thread(self.kokoro.synthesize_pcm16, chunk)
            await self._async_on_assistant_audio(pcm16)

    def _split_tts_chunks(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part and part.strip()]
        return parts or [text]

    def _on_gesture(self, name: str, confidence: float, held_for_ms: int) -> None:
        self.state_store.add_transcript("gesture", name, confidence=confidence, held_for_ms=held_for_ms)
        self._push_ui_event({"type": "gesture", "name": name, "confidence": confidence, "held_for_ms": held_for_ms})
        self.interrupt("gesture")

    def interrupt(self, source: str = "ui") -> None:
        self.playback.interrupt()
        self.state_store.update_realtime_status(assistant_speaking=False)
        self._last_playback_end_ts = time.monotonic()
        self._push_ui_event({"type": "interrupt", "source": source})

    def suppress_mic_for(self, milliseconds: int) -> None:
        until = time.monotonic() + max(0.0, milliseconds / 1000.0)
        self._mic_suppressed_until = max(self._mic_suppressed_until, until)

    async def _async_on_assistant_text(self, text: str) -> None:
        self.state_store.add_transcript("assistant", text)
        self._push_ui_event({"type": "assistant_text", "text": text})

    async def _async_on_assistant_audio(self, audio_bytes: bytes) -> None:
        if audio_bytes:
            self.playback.play_chunk(audio_bytes)
            duration_seconds = len(audio_bytes) / (2 * self.config.audio_channels * self.config.audio_sample_rate)
            self._last_playback_end_ts = time.monotonic() + duration_seconds
            self.state_store.update_realtime_status(assistant_speaking=True)
            self._push_ui_event({"type": "assistant_audio"})

    async def _async_on_user_transcript(self, transcript: str) -> None:
        self.state_store.add_transcript("user", transcript)
        self._push_ui_event({"type": "user_text", "text": transcript})

    async def _async_on_error(self, message: str) -> None:
        self.state_store.add_transcript("system", f"Backend error: {message}")
        self._push_ui_event({"type": "error", "message": message})

    async def _on_timer_warning(self, event: TimerWarningEvent) -> None:
        self._push_ui_event({
            "type": "timer_warning",
            "timer_id": event.timer_id,
            "label": event.label,
            "remaining_seconds": event.remaining_seconds,
        })

    async def _on_timer_notify(self, event: TimerNotifyEvent) -> None:
        self.suppress_mic_for(self.config.timer_alert_mic_block_ms)
        await self._play_timer_alert(event.label, event.remaining_seconds)
        self._push_ui_event({
            "type": "timer_notify",
            "timer_id": event.timer_id,
            "label": event.label,
            "remaining_seconds": event.remaining_seconds,
        })

    async def _on_timer_expired(self, event: TimerExpiredEvent) -> None:
        self._push_ui_event({
            "type": "timer_expired",
            "timer_id": event.timer_id,
            "label": event.label,
        })
        if not self._backend_client:
            return
        await self._run_timer_followup(event.label)

    async def _run_timer_followup(self, label: str) -> None:
        waited = 0.0
        while self._turn_in_flight and waited < 6.0:
            await asyncio.sleep(0.1)
            waited += 0.1
        if self._turn_in_flight:
            return
        prompt = (
            f"The timer for {label} has ended. In one short sentence, tell the user what to do now. "
            "Do not say you will notify them later."
        )
        self.state_store.add_transcript("system", prompt, synthetic=True, reason="timer_expired")
        self._push_ui_event({"type": "system_text", "text": prompt, "reason": "timer_expired"})
        self._turn_in_flight = True
        await self._handle_text_turn(prompt)

    async def _play_timer_alert(self, label: str, remaining_seconds: int) -> None:
        beep = await asyncio.to_thread(self._build_timer_alert_pcm16)
        self.playback.play_chunk(beep)
        duration_seconds = len(beep) / (2 * self.config.audio_channels * self.config.audio_sample_rate)
        self._last_playback_end_ts = time.monotonic() + duration_seconds
        self._push_ui_event({
            "type": "timer_alert_sound",
            "label": label,
            "remaining_seconds": remaining_seconds,
        })

    def _build_timer_alert_pcm16(self) -> bytes:
        sample_rate = self.config.audio_sample_rate
        pattern = [(880.0, 0.20), (0.0, 0.06), (1174.0, 0.22), (0.0, 0.06), (880.0, 0.22)]
        chunks: list[np.ndarray] = []
        for freq, duration in pattern:
            frames = max(1, int(sample_rate * duration))
            if freq <= 0:
                wave = np.zeros(frames, dtype=np.float32)
            else:
                t = np.arange(frames, dtype=np.float32) / sample_rate
                wave = 0.22 * np.sin(2.0 * math.pi * freq * t)
            if self.config.audio_channels > 1:
                wave = np.repeat(wave[:, None], self.config.audio_channels, axis=1).reshape(-1)
            chunks.append(wave.astype(np.float32))
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    def send_text_message(self, text: str) -> None:
        text = text.strip()
        if not text or not self._backend_client or self._turn_in_flight:
            return

        self.state_store.add_transcript("user", text)
        self._push_ui_event({"type": "user_text", "text": text})
        self._turn_in_flight = True
        self._backend.submit(self._handle_text_turn(text))

    def advance_step(self) -> None:
        result = self.state_store.advance_step()
        self._push_ui_event({"type": "manual_step", "result": result})

    def repeat_step(self) -> None:
        result = self.state_store.repeat_step()
        self._push_ui_event({"type": "manual_repeat", "result": result})

    def look_now(self) -> None:
        if not self._backend_client or self._turn_in_flight:
            return
        frames_result = self.camera_service.get_context_frames(count=3, max_age_seconds=8.0)
        frames = [
            {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "width": frame.width,
                "height": frame.height,
                "jpeg_bytes_base64": base64.b64encode(frame.jpeg_bytes).decode("ascii"),
                "motion_score": frame.motion_score,
            }
            for frame in frames_result
        ]
        text = "Please inspect the current cooking scene and tell me what looks most relevant for the current recipe step."
        self.state_store.add_transcript("user", text, synthetic=True)
        self._push_ui_event({"type": "user_text", "text": text})
        self._turn_in_flight = True
        self._backend.submit(self._handle_text_turn(text, frames=frames))

    def set_feature_flag(self, feature_name: str, enabled: bool) -> None:
        if hasattr(self.state_store.state.features, feature_name):
            setattr(self.state_store.state.features, feature_name, enabled)
            if feature_name == "gesture_enabled":
                self._gesture_service.set_enabled(enabled)
            self._push_ui_event({"type": "feature_flag", "feature_name": feature_name, "enabled": enabled})

    def get_latest_preview_rgb(self) -> np.ndarray | None:
        frame = self.camera_service.get_latest_preview_frame()
        if frame is None:
            return None
        if not self.playback.is_playing() and self.state_store.state.realtime.assistant_speaking:
            self.state_store.update_realtime_status(assistant_speaking=False)
            self._last_playback_end_ts = time.monotonic()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_state_snapshot(self) -> dict[str, Any]:
        return self.state_store.snapshot()

    def drain_ui_events(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        while True:
            try:
                items.append(self._ui_events.get_nowait())
            except queue.Empty:
                break
        return items

    def _push_ui_event(self, payload: dict[str, Any]) -> None:
        payload.setdefault("timestamp", time.time())
        self._ui_events.put(payload)
