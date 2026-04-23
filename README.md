# Cooking Assistant Demo (Windows)

## Recommended Python version
Use **Python 3.11** on Windows.

## Setup
```powershell
cd path	o\cooking_assistant
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your OpenAI API key to `.env`:
```env
OPENAI_API_KEY=your_key_here
```

## Smoke tests
```powershell
python -m app.main
pytest -q
```

## Run the desktop app
```powershell
python -m app.main_ui
```

## Notes
- If you do not set `OPENAI_API_KEY`, the desktop app will still open, but Realtime will stay disabled.
- Gesture detection supports raised palm (interrupt), cover mouth (toggle speech recognition), thumbs up (next step), pinky up (previous step), fist (repeat step), and 1/2/3 finger choice gestures.
- The camera preview is decoupled from model calls; image context is pulled from a rolling low-resolution buffer.
- Audio input/output assumes the default Windows microphone and speaker devices.


## New turn-based backend

This version uses:
- OpenAI Audio Transcriptions for speech-to-text
- OpenAI Responses API for text+image reasoning and tool use
- Local Kokoro TTS for speech output

### Additional Windows steps for Kokoro
1. Install `espeak-ng` using the Windows MSI from the Kokoro/PyPI instructions.
2. Run the app once while connected to the internet. The code will download `hexgrad/Kokoro-82M` into `./assets/models/Kokoro-82M`.

### Notes
- Spoken interruption while the assistant is talking is intentionally disabled for stability over speakers.
- Use the raised palm gesture or the Interrupt button to stop speech.
- Default model is now `gpt-5.4-nano` for lower cost.


Gesture tuning
- The mouth-cover gesture needs a compatible MediaPipe face detector model, such as `assets/models/face_detector.tflite` or one of the official BlazeFace `.tflite` downloads.
- You can tune each gesture independently with environment variables such as `COOKING_ASSISTANT_GESTURE_MOUTH_THRESHOLD`, `COOKING_ASSISTANT_GESTURE_THUMBS_UP_THRESHOLD`, `COOKING_ASSISTANT_GESTURE_PINKY_THRESHOLD`, and `COOKING_ASSISTANT_GESTURE_FIST_THRESHOLD`.
- For debugging classifier behavior, set `COOKING_ASSISTANT_GESTURE_DEBUG_LOGGING=true`.
