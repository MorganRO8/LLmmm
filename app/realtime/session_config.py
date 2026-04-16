from __future__ import annotations

from app.config import AppConfig
from app.core.models import Recipe
from app.realtime.prompts import build_system_prompt
from app.realtime.tools import TOOL_DEFINITIONS


def _audio_format(name: str) -> dict:
    if name == "pcm16":
        return {"type": "audio/pcm", "rate": 24000}
    if name == "pcmu":
        return {"type": "audio/pcmu"}
    if name == "pcma":
        return {"type": "audio/pcma"}
    return {"type": "audio/pcm", "rate": 24000}


def build_session_config(config: AppConfig, recipe: Recipe) -> dict:
    return {
        "type": "realtime",
        "instructions": build_system_prompt(recipe),
        # Current Realtime API supports ["audio"] or ["text"], not both together.
        # Audio responses still emit output-audio transcript events.
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": _audio_format(config.input_audio_format),
                "noise_reduction": {"type": config.noise_reduction_mode},
                "transcription": {
                    "model": "gpt-4o-mini-transcribe",
                    "language": "en",
                },
                "turn_detection": {
                    "type": config.vad_mode,
                    "create_response": True,
                    # Prevent the server from auto-canceling the assistant just
                    # because speaker leakage is detected as speech.
                    "interrupt_response": False,
                    "prefix_padding_ms": config.vad_prefix_padding_ms,
                    "silence_duration_ms": config.vad_silence_duration_ms,
                    "threshold": config.vad_threshold,
                },
            },
            "output": {
                "format": _audio_format(config.output_audio_format),
                "voice": config.output_voice,
            },
        },
        "tools": TOOL_DEFINITIONS,
        "tool_choice": "auto",
    }
