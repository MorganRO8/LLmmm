import asyncio
import sys
import types
from types import SimpleNamespace

sd_stub = types.SimpleNamespace(OutputStream=object, InputStream=object)
sys.modules.setdefault("sounddevice", sd_stub)
openai_stub = types.SimpleNamespace(AsyncOpenAI=object)
sys.modules.setdefault("openai", openai_stub)

from app.controller import DesktopController


class _PlaybackStub:
    def __init__(self) -> None:
        self.interrupt_calls = 0
        self.played_chunks: list[bytes] = []

    def interrupt(self) -> None:
        self.interrupt_calls += 1

    def play_chunk(self, audio_bytes: bytes) -> None:
        self.played_chunks.append(audio_bytes)


class _StateStoreStub:
    def __init__(self) -> None:
        self.state = SimpleNamespace(
            features=SimpleNamespace(tts_enabled=True, speech_enabled=True),
            realtime=SimpleNamespace(assistant_speaking=False),
        )
        self.transcripts: list[tuple[str, str, dict]] = []
        self.status_updates: list[dict] = []

    def add_transcript(self, speaker: str, text: str, **metadata):
        self.transcripts.append((speaker, text, metadata))

    def update_realtime_status(self, **kwargs) -> None:
        self.status_updates.append(kwargs)
        for key, value in kwargs.items():
            setattr(self.state.realtime, key, value)


class _KokoroStub:
    def __init__(self, controller: DesktopController | None = None, interrupt_after_first: bool = False) -> None:
        self.controller = controller
        self.interrupt_after_first = interrupt_after_first
        self.calls: list[str] = []

    def synthesize_pcm16(self, text: str) -> bytes:
        self.calls.append(text)
        if self.interrupt_after_first and len(self.calls) == 1 and self.controller is not None:
            self.controller.interrupt("test")
        return text.encode("utf-8")


def _make_controller() -> DesktopController:
    controller = DesktopController.__new__(DesktopController)
    controller.playback = _PlaybackStub()
    controller.state_store = _StateStoreStub()
    controller.config = SimpleNamespace(audio_channels=1, audio_sample_rate=24000)
    controller._last_playback_end_ts = 0.0
    controller._ui_events = []
    controller._tts_interrupt_generation = 0
    controller._tts_interrupt_lock = __import__("threading").Lock()
    controller._turn_in_flight = False
    controller._backend_client = object()
    controller._push_ui_event = controller._ui_events.append
    controller._advance_recipe_from_gesture = lambda: controller._ui_events.append({"type": "next"})
    controller._go_back_recipe_from_gesture = lambda: controller._ui_events.append({"type": "back"})
    controller._repeat_recipe_from_gesture = lambda: controller._ui_events.append({"type": "repeat"})
    controller._send_gesture_choice = lambda option: controller._ui_events.append({"type": "option", "option": option})
    return controller


async def _assistant_text(controller: DesktopController, text: str) -> None:
    await DesktopController._deliver_assistant_text(controller, text)


def test_interrupt_invalidates_remaining_tts_chunks() -> None:
    controller = _make_controller()
    controller.kokoro = _KokoroStub(controller=controller, interrupt_after_first=True)

    asyncio.run(_assistant_text(controller, "First sentence. Second sentence."))

    assert controller.kokoro.calls == ["First sentence."]
    assert controller.playback.played_chunks == []
    assert controller.state_store.state.realtime.assistant_speaking is False


def test_non_mouth_gestures_interrupt_before_action() -> None:
    controller = _make_controller()
    controller.kokoro = _KokoroStub()

    DesktopController._on_gesture(controller, "thumbs_up_next_step", 0.93, 700)

    assert controller.playback.interrupt_calls == 1
    assert {event["type"] for event in controller._ui_events if isinstance(event, dict)} >= {"interrupt", "next"}


def test_mouth_cover_toggle_does_not_interrupt() -> None:
    controller = _make_controller()
    controller.kokoro = _KokoroStub()

    DesktopController._on_gesture(controller, "mouth_cover_toggle_speech", 0.97, 700)

    assert controller.playback.interrupt_calls == 0
    assert controller.state_store.state.features.speech_enabled is False
