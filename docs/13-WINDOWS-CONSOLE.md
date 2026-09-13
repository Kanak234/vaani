# Windows integrated console

The Windows desktop workflow is now exposed as one GUI entry point:

```powershell
python -m vaani.ui
```

or, after editable installation:

```powershell
vaani-windows
```

## Meeting takeover

The GUI owns the complete Windows path:

`physical microphone -> Vaani session -> VB-CABLE playback -> meeting microphone`

`meeting speaker loopback -> Whisper -> takeover policy -> local Ollama response -> configured voice -> VB-CABLE playback`

The meeting client must use the VB-CABLE recording endpoint as its microphone. Vaani writes to the matching VB-CABLE playback endpoint.

The STOP / EMERGENCY STOP control calls the existing session stop path and is intentionally separate from response generation.

## Screen recording analysis

The Screen Recording Analysis tab samples frames from a local recording with OpenCV and sends those sampled images to a local Ollama vision model. The default model is `qwen3-vl:4b` because it is listed by Ollama as a 3.3 GB text+image model. No recording is uploaded by Vaani.

The workflow is:

1. Choose a screen recording.
2. Analyze sampled frames.
3. Review the visual context report.
4. Generate a concise response using the local text model.
5. If personal voice is selected and consent is active, synthesize that response with the enrolled XTTS voice and route it through VB-CABLE.

Personal voice use remains gated by the existing enrollment and consent ledger.

## Installation

The existing Windows environment should be refreshed with:

```powershell
python -m pip install -e ".[stt,translate,voice,windows,screen]"
```

The Ollama vision model is separate from the Python package. Install a local vision model only when screen analysis is needed, for example:

```powershell
ollama pull qwen3-vl:4b
```

## Current hardware/runtime boundary

Vaani does not assume that Ollama CUDA is healthy. The Windows desktop currently has a verified CPU Ollama path; the existing GPU crash must be treated as a separate runtime issue. The GUI therefore does not claim GPU acceleration merely because an RTX GPU is present.
