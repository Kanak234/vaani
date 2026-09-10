# Phase 0 — Requirements Discovery & Environment Baseline

**Status:** Complete. All figures below were *measured on the target machine*, not estimated.
**Date measured:** 2026-09-09

## 0.1 Target machine

| Property | Value | Consequence for design |
|---|---|---|
| OS | Ubuntu 26.04.1 LTS (kernel 7.0.0-31) | Linux is the v1 target platform |
| CPU | AMD Ryzen 5 5600H, 6C/12T | Enough for CPU-side VAD + audio DSP |
| RAM | 14 GiB total, **~2.9 GiB free at probe time** | Hard ceiling. Cannot hold multiple large models resident |
| GPU | NVIDIA RTX 3050 Mobile, **4096 MiB VRAM**, driver 595.84 | Fits *one* mid-size model at a time. Not two |
| Audio server | **PipeWire 1.6.2** with PulseAudio compat (protocol 35) | Virtual-source route is available natively |
| Default format | float32le, 2ch, 48000 Hz | Resample to 16 kHz mono for STT |
| Python | 3.14.4 (system) | Very new; wheel availability had to be verified, not assumed |
| Node | v22.22.1 | Available if an Electron/web UI is chosen |
| Rust | cargo 1.98.1 | Available for a native audio path if needed |
| Disk | 214 GiB free | Model storage is not a constraint |

## 0.2 Verified capabilities (tested, not assumed)

### Virtual microphone — **VERIFIED WORKING**
```
pactl load-module module-null-sink media.class=Audio/Source/Virtual \
      sink_name=VT_TestMic channel_map=mono
```
Result: module id `536870916`; the node appeared in `pactl list short sources` as
`VT_TestMic  PipeWire  float32le 1ch 48000Hz` and unloaded cleanly.

This is the single highest-risk platform dependency and it is confirmed on this
machine, **without sudo**. Any application using the PulseAudio or PipeWire client
API (Zoom, Google Meet in a browser, Teams, Firefox, Chrome) will enumerate this
node as a selectable microphone.

### Audio capture — **VERIFIED WORKING**
`libportaudio2` is **not installed** and installing it requires sudo, so
`sounddevice` fails with `OSError: PortAudio library not found`.

However `libpulse-simple.so.0` is present. A ctypes binding to it was written and
tested end-to-end:

```
record stream OPEN, requested fragsize=640B (20ms)
read 25 frames in 524.4ms (expected ~500ms)
server-reported capture latency: 22.33 ms
peak amplitude: 781   (live signal, mic not muted)
```

**Measured capture latency: 22.33 ms** at a 20 ms fragment size. This becomes the
first entry in the latency budget and it is a real number.

### ML runtime — **VERIFIED**
Wheels install cleanly on CPython 3.14:
`faster-whisper 1.2.1`, `ctranslate2 4.8.2`, `onnxruntime 1.29.0`, `numpy 2.5.3`,
`soundfile 0.14.0`.

- `ctranslate2.get_cuda_device_count()` -> **1**
- CUDA compute types: `float16`, `int8_float16`, `bfloat16`, `int8`, `float32`
- CPU compute types: `int8`, `int8_float32`, `float32`
- **`onnxruntime` providers: `['AzureExecutionProvider', 'CPUExecutionProvider']` — no CUDA EP.**
  ONNX models (Silero VAD) therefore run on CPU. This is acceptable: Silero VAD is
  ~1.8 MB and costs well under a millisecond per 32 ms frame.

## 0.3 Decisions forced by the baseline

1. **The audio layer binds `libpulse-simple` directly via ctypes**, not PortAudio.
   Rationale: no sudo needed, one fewer abstraction layer between the app and
   PipeWire, and explicit control over `fragsize`/`tlength` which is exactly the knob
   that governs latency. PortAudio would sit on top of the same server anyway.
2. **4 GiB VRAM means one GPU-resident model at a time.** A design that keeps STT and
   a voice-cloning TTS both hot on the GPU will OOM. The engine must either
   size models to co-reside, or schedule them.
3. **~2.9 GiB free RAM at probe time is the binding constraint, more than VRAM.**
   Model choice must be validated against real memory headroom, and the app must
   degrade rather than crash when memory is short.
4. **Python 3.14 is new enough that every dependency must be verified before it is
   written into a design.** Two candidate libraries were already eliminated this way.

## 0.4 Open questions carried into later phases

| # | Question | Resolved in |
|---|---|---|
| Q1 | Whisper accuracy on *this user's* Hindi/Hinglish speech | Requires user audio — Phase 10 |
| Q2 | Is local translation quality on code-mixed Hinglish acceptable? | Phase 7 |
| Q3 | Can a voice-cloning TTS meet the latency target on 4 GiB VRAM? | Phase 7 |
| Q4 | Does Zoom/Meet accept the virtual source without extra config? | Phase 10 manual test |
