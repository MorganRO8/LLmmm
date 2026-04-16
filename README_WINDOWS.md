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

## Enable hand recognition
Gesture controls rely on mediapipe, and for that a specific model must be downloaded and placed into ./assets/models (it should be named hand_landmarker.task)
Download here:
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models

## Enable TTS
For local TTS we use Kokoro-82M, to run this on windows you must download and install espeak-ng manually.

To install espeak-ng on Windows:

Go to espeak-ng releases
Click on Latest release
Download the appropriate *.msi file (e.g. espeak-ng-20191129-b702b03-x64.msi)
Run the downloaded installer

## Run the desktop app
```powershell
python -m app.main_ui
```

## Notes
- If you do not set `OPENAI_API_KEY`, the desktop app will still open, but Realtime will stay disabled.
- Gesture detection currently implements a single high-threshold raised-palm interrupt gesture.
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
