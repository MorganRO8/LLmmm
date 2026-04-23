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

import json

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
from app.gestures.mediapipe_adapter import GestureTuning
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
        if self.loop.is_running():
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
        self._tts_interrupt_lock = threading.Lock()
        self._tts_interrupt_generation = 0

        self._gesture_service = GestureService(
            camera_service=self.camera_service,
            confidence_threshold=self.config.gesture_confidence_threshold,
            hold_ms=self.config.gesture_hold_ms,
            cooldown_ms=self.config.gesture_cooldown_ms,
            poll_interval_ms=self.config.gesture_poll_interval_ms,
            on_gesture=self._on_gesture,
            hand_model_path=self.config.gesture_hand_model_path,
            face_model_path=self.config.gesture_face_model_path,
            palm_threshold=self.config.gesture_palm_threshold,
            mouth_threshold=self.config.gesture_mouth_threshold,
            thumbs_up_threshold=self.config.gesture_thumbs_up_threshold,
            pinky_threshold=self.config.gesture_pinky_threshold,
            fist_threshold=self.config.gesture_fist_threshold,
            option_threshold=self.config.gesture_option_threshold,
            tuning=GestureTuning(
                mouth_box_expand=self.config.gesture_mouth_box_expand,
                mouth_distance_factor=self.config.gesture_mouth_distance_factor,
                mouth_min_overlap_ratio=self.config.gesture_mouth_min_overlap_ratio,
                pinky_min_extension=self.config.gesture_pinky_min_extension,
                pinky_min_separation=self.config.gesture_pinky_min_separation,
                fist_max_avg_tip_distance_ratio=self.config.gesture_fist_max_avg_tip_distance_ratio,
                fist_max_bbox_height_ratio=self.config.gesture_fist_max_bbox_height_ratio,
                debug_logging=self.config.gesture_debug_logging,
            ),
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
        if self._backend.is_alive():
            self._backend.join(timeout=2.0)

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
        tts_generation = self._get_tts_interrupt_generation()
        self.state_store.update_realtime_status(assistant_speaking=True)
        for chunk in self._split_tts_chunks(text):
            if self._is_tts_interrupted(tts_generation):
                break
            pcm16 = await asyncio.to_thread(self.kokoro.synthesize_pcm16, chunk)
            if self._is_tts_interrupted(tts_generation):
                break
            await self._async_on_assistant_audio(pcm16)
        if self._is_tts_interrupted(tts_generation):
            self.state_store.update_realtime_status(assistant_speaking=False)

    def _split_tts_chunks(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part and part.strip()]
        return parts or [text]

    def _on_gesture(self, name: str, confidence: float, held_for_ms: int) -> None:
        self.state_store.add_transcript("gesture", name, confidence=confidence, held_for_ms=held_for_ms)
        self._push_ui_event({"type": "gesture", "name": name, "confidence": confidence, "held_for_ms": held_for_ms})

        if name == "mouth_cover_toggle_speech":
            self._toggle_feature_flag("speech_enabled", source_gesture=name)
            return

        self.interrupt(f"gesture:{name}")

        if name == "raised_palm_interrupt":
            return

        if name == "thumbs_up_next_step":
            self._advance_recipe_from_gesture()
            return

        if name == "pinky_up_previous_step":
            self._go_back_recipe_from_gesture()
            return

        if name == "fist_repeat_step":
            self._repeat_recipe_from_gesture()
            return

        if name.startswith("option_choice_"):
            try:
                option_number = int(name.rsplit("_", 1)[-1])
            except ValueError:
                return
            self._send_gesture_choice(option_number)

    def interrupt(self, source: str = "ui") -> None:
        self._bump_tts_interrupt_generation()
        self.playback.interrupt()
        self.state_store.update_realtime_status(assistant_speaking=False)
        self._last_playback_end_ts = time.monotonic()
        self._push_ui_event({"type": "interrupt", "source": source})

    def _bump_tts_interrupt_generation(self) -> int:
        with self._tts_interrupt_lock:
            self._tts_interrupt_generation += 1
            return self._tts_interrupt_generation

    def _get_tts_interrupt_generation(self) -> int:
        with self._tts_interrupt_lock:
            return self._tts_interrupt_generation

    def _is_tts_interrupted(self, generation: int) -> bool:
        with self._tts_interrupt_lock:
            return generation != self._tts_interrupt_generation

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

    def _toggle_feature_flag(self, feature_name: str, source_gesture: str | None = None) -> None:
        if not hasattr(self.state_store.state.features, feature_name):
            return
        current_value = bool(getattr(self.state_store.state.features, feature_name))
        enabled = not current_value
        setattr(self.state_store.state.features, feature_name, enabled)
        if feature_name == "gesture_enabled":
            self._gesture_service.set_enabled(enabled)
        self._push_ui_event({
            "type": "feature_flag",
            "feature_name": feature_name,
            "enabled": enabled,
            "source_gesture": source_gesture,
        })
        state_word = "enabled" if enabled else "disabled"
        self._push_ui_event({
            "type": "status",
            "message": f"{feature_name.replace('_', ' ').title()} {state_word} via gesture.",
        })

    def _advance_recipe_from_gesture(self) -> None:
        if not self._backend_client or self._turn_in_flight:
            return
        result = self.state_store.advance_step()
        current_step = result.get("current_step") or {}
        next_step = result.get("next_step") or {}
        title = current_step.get("title") or "the next step"
        text = (
            "The user gave a thumbs-up to confirm the current step is complete. "
            f"The recipe state has already advanced. Guide them through the new current step: {title}. "
            "Briefly acknowledge the completion and say what to do now."
        )
        if next_step:
            text += f" After that, the following step will be {next_step.get('title', 'next')}."
        self.state_store.add_transcript("user", text, synthetic=True, gesture_navigation="next_step")
        self._push_ui_event({"type": "user_text", "text": "[gesture] next step"})
        self._push_ui_event({"type": "manual_step", "result": result, "source": "gesture"})
        self._turn_in_flight = True
        self._backend.submit(self._handle_text_turn(text))

    def _go_back_recipe_from_gesture(self) -> None:
        if not self._backend_client or self._turn_in_flight:
            return
        result = self.state_store.previous_step()
        current_step = result.get("current_step") or {}
        title = current_step.get("title") or "the previous step"
        if result.get("moved_back"):
            text = (
                "The user held up a pinky to go back one step. "
                f"The recipe state has already moved back. Restate the step now in progress: {title}. "
                "Keep it short and practical."
            )
        else:
            text = (
                "The user held up a pinky to go back, but the recipe is already at the first step. "
                f"Repeat the current step: {title}."
            )
        self.state_store.add_transcript("user", text, synthetic=True, gesture_navigation="previous_step")
        self._push_ui_event({"type": "user_text", "text": "[gesture] previous step"})
        self._push_ui_event({"type": "manual_step", "result": result, "source": "gesture"})
        self._turn_in_flight = True
        self._backend.submit(self._handle_text_turn(text))

    def _repeat_recipe_from_gesture(self) -> None:
        if not self._backend_client or self._turn_in_flight:
            return
        result = self.state_store.repeat_step()
        current_step = result.get("current_step") or {}
        title = current_step.get("title") or "the current step"
        text = (
            "The user made a fist to hear the current step again without changing recipe state. "
            f"Repeat the current step, which is {title}, in a concise way."
        )
        self.state_store.add_transcript("user", text, synthetic=True, gesture_navigation="repeat_step")
        self._push_ui_event({"type": "user_text", "text": "[gesture] repeat step"})
        self._push_ui_event({"type": "manual_repeat", "result": result, "source": "gesture"})
        self._turn_in_flight = True
        self._backend.submit(self._handle_text_turn(text))

    def _send_gesture_choice(self, option_number: int) -> None:
        if option_number < 1 or option_number > 3 or not self._backend_client or self._turn_in_flight:
            return

        frames = self._build_gesture_confirmation_frames()
        payload = {
            "gesture_choice": option_number,
            "requires_visual_confirmation": bool(frames),
        }
        text = (
            f"The user is signaling option {option_number} with their fingers. "
            f"Confirm from the attached image whether the hand signal looks like choice {option_number}. "
            "If the gesture looks ambiguous, say so briefly and ask the user to repeat it. "
            "If it looks correct, continue using that numbered choice.\n"
            f"gesture_payload={json.dumps(payload)}"
        )
        self.state_store.add_transcript("user", text, synthetic=True, gesture_choice=option_number, visual_confirmation_requested=bool(frames))
        self._push_ui_event({"type": "user_text", "text": f"[gesture] option {option_number}"})
        self._turn_in_flight = True
        self._backend.submit(self._handle_text_turn(text, frames=frames))

    def _build_gesture_confirmation_frames(self) -> list[dict[str, Any]]:
        frame = self.camera_service.get_latest_preview_frame()
        if frame is None:
            return []
        resized = cv2.resize(frame, (self.config.context_frame_width, self.config.context_frame_height), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return []
        return [{
            "frame_id": f"gesture-preview-{int(time.time() * 1000)}",
            "timestamp": time.time(),
            "width": resized.shape[1],
            "height": resized.shape[0],
            "jpeg_bytes_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "motion_score": None,
        }]

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
