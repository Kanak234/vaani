# Windows audio support

Vaani's original audio path uses PulseAudio/PipeWire and `pactl`, which is Linux-specific. The Windows implementation now uses the native Windows WASAPI stack through SoundCard.

## Architecture

### Translation

`Windows microphone -> Whisper -> translation -> XTTS -> selected playback endpoint`

### Meeting takeover

`Windows microphone -> translation -> virtual-cable playback endpoint -> meeting microphone`

`meeting speaker loopback -> Whisper -> takeover policy -> qwen3:8b -> XTTS -> virtual-cable playback endpoint`

The meeting application must select the virtual cable's **recording** endpoint as its microphone. Vaani writes to the matching **playback** endpoint.

## Why a virtual cable is still required

A Python application cannot create a Windows kernel audio driver merely by opening a WASAPI stream. Vaani therefore does not pretend that a software-only virtual microphone exists. The Windows backend treats the virtual cable as an explicit external audio endpoint.

This is intentionally different from Linux, where the existing PipeWire implementation creates `VaaniSink` and `VaaniVirtualMic` itself.

## Device discovery

Run:

```powershell
python -m vaani.devices.windows_manager
```

Use the exact device names printed by that command with `--input`, `--remote-input`, and `--output-device`.

## Meeting takeover

Example shape:

```powershell
python -m vaani.session.meeting_takeover_app --input "YOUR MICROPHONE" --remote-input "YOUR SPEAKER LOOPBACK" --output-device "YOUR VIRTUAL CABLE PLAYBACK"
```

`--remote-input` must be a WASAPI loopback microphone exposed by SoundCard. Do not point it at Vaani's own output endpoint, or the assistant can hear its own generated speech.

## Model/runtime note

The Windows audio port does not assume that Ollama CUDA is healthy. The target desktop currently has a verified CPU Ollama path; GPU inference must be treated separately until the Windows Ollama `0xc0000005` crash is resolved.

## Verification boundary

The repository changes establish the Windows audio architecture and integration. Actual end-to-end microphone, loopback, virtual-cable, Whisper, XTTS, and meeting-client verification must be performed on the target Windows desktop because the GitHub integration cannot access that machine's audio devices.
