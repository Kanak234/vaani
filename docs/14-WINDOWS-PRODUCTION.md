# 14. Windows Production Guide & Architecture

## Overview

Vaani is designed for production execution on Windows 10/11 x64 systems without requiring cloud dependencies or manual code patching. It features runtime hardware detection, automatic capability profiling, GPU inference probing with safe CPU fallback, WASAPI audio integration, and emergency stop protection.

---

## Target Hardware Configurations

### 1. High Performance (Desktop)
- **CPU**: Intel Core i9 KF-series (24 cores)
- **GPU**: NVIDIA GeForce RTX 3050 6GB VRAM
- **RAM**: 32GB DDR5
- **Profile**: `HIGH`
- **STT**: `faster-whisper` (medium/small, CUDA compute type float16/int8)
- **LLM**: Ollama (`qwen3:14b` or `qwen3:8b`, with runtime GPU health check)
- **TTS**: `XTTS-v2` (CUDA accelerated) or formant fallback

### 2. Balanced (Laptop)
- **CPU**: AMD Ryzen 5 5600H (12 threads)
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU 4GB VRAM
- **RAM**: 16GB DDR4
- **Profile**: `BALANCED` / `LOW`
- **STT**: `faster-whisper` (small/base, int8)
- **LLM**: Ollama (`qwen3:8b` or `phi4-mini:latest`)
- **TTS**: CPU fallback or lightweight voice cloning

---

## Critical Windows Architecture Components

### 1. Hardware Detection & Profiling (`vaani.system.hardware`, `vaani.system.profile`)
- Inspects CPU models, logical cores, and AVX2/AVX512 instruction flags.
- Probes GPU model, VRAM capacity, and driver version using `nvidia-smi`.
- Queries installed Ollama models dynamically via `/api/tags`.
- Detects physical microphones, speakers, and virtual audio devices via WASAPI (`soundcard`).
- Assigns a capability profile (`LOW`, `BALANCED`, `HIGH`) and per-component decisions.

### 2. Runtime GPU Probing & CPU Ollama Fallback (`vaani.system.ollama_manager`)
- **Observed Issue**: On certain Windows desktop configurations, Ollama CUDA backend crashes with status `0xc0000005`.
- **Mitigation**: Vaani sends a probe prompt (`{"model": ..., "prompt": "test", "stream": false}`) with a 15s timeout before routing meeting or translation queries.
- **CPU Fallback**: If GPU inference fails, Vaani launches or connects to a dedicated fallback server:
  ```cmd
  set OLLAMA_LLM_LIBRARY=cpu_avx2
  set OLLAMA_HOST=127.0.0.1:11435
  ollama serve
  ```
- Gracefully continues execution on CPU without killing unrelated processes.

### 3. Windows Audio & WASAPI Topology (`vaani.audio.backend.windows_backend`, `vaani.devices.windows_manager`)
- **WASAPI Mono/Stereo Handling**: Supports native 1-channel and 2-channel endpoints with automatic channel duplication on playback.
- **Sample Rate Conversion**: Bridges XTTS-v2 24kHz outputs to 16kHz WASAPI playback buffers cleanly.
- **Virtual Audio Cable**: Automatically identifies VB-CABLE (`CABLE Input`, `CABLE Output`) and routes synthesized speech directly into virtual meeting inputs.
- **Loopback Capture**: Captures incoming remote meeting audio for question and turn-taking detection without feedback loops.

### 4. Meeting Takeover Engine (`vaani.session.meeting_takeover_app`, `vaani.session.takeover`)
- **Explicit Trigger**: "Vaani, take over" detected exclusively from local microphone audio.
- **Automatic Handoff**: Detects remote questions, user hesitation ("um", "uh"), and silence timeouts (>4.0s or extended >8.0s silence).
- **Emergency Stop**: Global hotkey (`Esc` or `Ctrl+Shift+S`) and red UI button immediately halt all audio output and release device handles.

---

## Installation & Launch on Windows

### Option A: Standalone Release Installer
1. Download and run `Vaani-1.0.0-Setup.exe`.
2. Follow the setup wizard to install to `C:\Program Files\Vaani`.
3. Launch Vaani from the Start Menu or Desktop shortcut.

### Option B: Batch / Source Launch
1. Ensure Python 3.12+ and Git are installed.
2. Run `vaani.bat` from the repository root:
   ```cmd
   .\vaani.bat gui
   ```
   *This automatically creates `.venv`, installs dependencies, and boots the console GUI.*

---

## Prerequisites for Windows

1. **VB-Audio Virtual Cable**:
   - Download free from [VB-Audio](https://vb-audio.com/Cable/).
   - Set meeting app (Zoom/Teams/Meet) microphone to `CABLE Output (VB-Audio Virtual Cable)`.
2. **Ollama**:
   - Install from [Ollama.com](https://ollama.com).
   - Recommended models:
     ```cmd
     ollama pull qwen3:8b
     ollama pull deepseek-r1:8b
     ollama pull qwen2.5-vl:3b  # For screen analysis
     ```

---

## Troubleshooting

| Symptom | Cause | Remediation |
|---|---|---|
| `Ollama GPU crash (0xc0000005)` | CUDA runtime incompatibility in llama-server | Vaani auto-detects and redirects to port 11435 with `OLLAMA_LLM_LIBRARY=cpu_avx2`. |
| `No virtual cable detected` | VB-CABLE driver not installed | Install VB-Audio Virtual Cable; restart Vaani. |
| `Audio played too fast/slow` | Sample rate mismatch | `_ensure_sample_rate` resamples 24kHz to 16kHz automatically. |
| `Screen analysis unavailable` | Vision model missing | Run `ollama pull qwen2.5-vl:3b` or `ollama pull qwen3-vl:4b`. |
