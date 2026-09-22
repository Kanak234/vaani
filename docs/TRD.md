# Vaani v1.0 — Technical Requirements (source of truth)

**Status:** Phase 0 draft, 2026-09-22. It supersedes `03-TRD.md` for v1.0. The
architecture in `06-ARCHITECTURE.md` and ADR-001…004 still holds.

## 1. Stack (fixed; don't change without asking Kanak)

| Stage | Choice | Module |
|---|---|---|
| Capture / output | libpulse-simple via ctypes (PipeWire pulse compat) | `audio/backend/pulse_*` |
| VAD | Silero VAD (ONNX, CPU) with energy VAD fallback | `ai/vad/` |
| STT | faster-whisper (CTranslate2), tiny…large-v3 | `ai/stt/whisper.py` |
| Translation | Router: NLLB-200 600M (CT2) for Devanagari/English, local Ollama LLM for romanised Hinglish | `ai/translate/` (ADR-004) |
| TTS | Coqui XTTS-v2 with enrolled reference; fallback voice with a flag | `ai/tts/` |
| Virtual mic | null-sink + remap-source, two modules (ADR-001) | `devices/manager.py` |
| UI | tkinter overlay + app (no new GUI toolkit) | `ui/` |
| IPC | Unix socket, one command per connection | `ipc/control.py` |
| Storage | SQLite (consent, profiles; transcripts opt-in) | `storage/` |

Python ≥ 3.12. The venv is built by `install.sh` with **Python 3.12 via uv**: 3.14 has
no PyTorch wheels on the desktop setup, and XTTS needs torch. (Proposed. The
laptop venv was 3.14 without `[voice]`, see "Open items".)

## 2. Hardware targets

| Machine | GPU | CUDA state | Notes |
|---|---|---|---|
| Laptop HP Victus (Ryzen 5 5600H, 14–16 GB) | RTX 3050 Laptop 4 GB | proprietary 595 **or** Nouveau/NVK → no CUDA | GNOME/Wayland |
| Desktop i9-14900KF, 32 GB | RTX 3050 6 GB | driver 595 / CUDA 13.x | KDE/Wayland (Kubuntu); crashes under sustained load → checkpoint long jobs |

CUDA is "available" only if `ctranslate2.get_cuda_device_count() > 0` **and** a
2-second test op succeeds. Nouveau/NVK has no CUDA, so it lands on CPU automatically.
`VAANI_FORCE_CPU=1` forces CPU (used to verify A4 on a CUDA machine).

## 3. New / changed components for v1

| Component | File (new = ★) | Responsibility |
|---|---|---|
| Input gate | ★ `session/input_mode.py` | always-on / ptt-toggle / ptt-hold / bypass state; the capture loop asks it per frame |
| Hold-to-talk reader | ★ `system/evdev_ptt.py` | optional; reads `/dev/input/event*` if readable, otherwise disabled with a reason |
| Bypass path | `session/engine.py` | copies mic frames to the virtual-mic writer; flushes the TTS queue first |
| Model selector | `system/profile.py`, ★ `system/model_select.py` | ranks candidates per stage from detection + benchmark; returns choice + reason |
| Benchmark cache | ★ `system/bench_cache.py` | JSON under the data dir, keyed by (hostname, GPU, CUDA ok, sorted Ollama models) |
| Stage supervisor | ★ `core/supervisor.py` | wraps each stage call: timeout, exception/OOM → mark failed → next candidate; emits `fallback` events |
| Takeover A | ★ `session/continuation.py` | silence + incomplete-sentence detection, prompt building, generation, cancel token |
| Takeover B | existing `session/takeover*.py`, `meeting_takeover_app.py` | wired into the Linux meeting session behind `--listen-remote` |
| Script mode | ★ `session/script_mode.py` | file decode (ffmpeg if present, else soundfile) → VAD segments → STT → translate → text/wav; chunk checkpoints |
| CLI | `cli.py` | new commands: `ptt`, `bypass`, `takeover`, `cancel`, `script`, `models`; the `--llm-model` default becomes `auto` |
| Config | `config/__init__.py` | JSON at `~/.config/vaani/config.json`; new keys listed in §5 |
| Installer | `install.sh` (repo root, ★) | one step; see §6 |

### Data flow (meeting)
```
mic ─► capture loop ─► InputGate ─┬─(bypass)──────────────────────────────► vmic writer
                                  └─(open)─► VAD/segmenter ─► worker:
         STT ─► router(NLLB|LLM) ─► gate ─► TTS ─► output queue ─► vmic writer
         (every stage call goes through Supervisor → fallback chain)
continuation (Takeover A) ◄─ context engine (turns) ; output ─► same output queue,
         tagged source=AI  ─► overlay "AI SPEAKING"; `cancel` flushes the tagged items
```

## 4. Stability requirements (Phase 1)

- Bounded queues everywhere (capture, worker, output). When full, drop the oldest
  audio and count it. Never block the capture thread.
- No per-utterance growth: context turns are capped (existing), latency samples use a
  ring buffer, and torch/CT2 tensors are released. Measured by the soak test: RSS every
  60 s, pass if RSS at 30 min ≤ RSS at 5 min × 1.10.
- Each worker thread is supervised: an uncaught exception is logged, the stage is
  reloaded, and the session keeps going. Three crashes of one stage within 60 s →
  fallback candidate.
- Log file: `~/.local/state/vaani/vaani.log`, rotating 5 × 2 MB. No transcript
  text in logs unless `persist_transcript` is on.

## 5. Config keys (additions; defaults in parentheses)

`input_mode` ("always_on" | "ptt_toggle" | "ptt_hold"), `takeover.auto_armed`
(false), `takeover.pause_s` (4.0), `takeover.max_sentences` (3),
`takeover.listen_remote` (false), `llm_model` ("auto"), `llm_preference`
(["phi4-mini", "llama3.1:8b", "qwen2.5", "qwen3", "deepseek-r1"]; used only to
break ties among models that passed the benchmark; reasoning models like deepseek-r1 are
ranked last because `<think>` output adds latency), `stage_timeout_factor` (3.0),
`bench_max_s` (20), `script.whisper_model` ("auto" → largest that fits).

## 6. Installer (`./install.sh`)

1. Check: Linux, `pactl`, `libpulse-simple.so.0`, `uv` (installed to `~/.local/bin`
   if missing, no root), `ffmpeg` (optional; warn only).
2. `uv venv --python 3.12 .venv` and `uv pip install -e ".[all]"`. The CUDA wheels
   are installed only if `nvidia-smi` works.
3. Model downloads into `~/.local/share/vaani/models` (resumable): Silero VAD,
   Whisper (the auto-chosen size + `small` fallback), NLLB-600M-CT2, XTTS-v2 (the
   Coqui licence prompt is answered only with the user's explicit `--agree-cpml`
   or an interactive yes).
4. Ollama: detect only. If it's missing, print how to install it. **It doesn't
   install system software or pull LLMs without asking.**
5. The virtual mic isn't persistent config: it's created at runtime by `vaani`
   (V-5), so install doesn't touch PipeWire config. `install.sh` does a
   create→tone→destroy self-test.
6. Launcher `~/.local/bin/vaani`, desktop entry, and desktop shortcuts (GNOME
   gsettings / KDE kglobalaccel) for ptt, bypass, takeover, cancel, panic, set only
   if the keys are free. Otherwise it prints them for manual binding.
7. `--hold-to-talk`: explains the `input` group trade-off and asks before running `sudo
   usermod -aG input $USER`.

Idempotent: re-running it upgrades in place.

## 7. Testing strategy

- Unit tests (fast, no models) for every new module, with fakes for the audio backend,
  models and Ollama HTTP.
- Integration tests use the existing fake providers (`tests/integration`).
- Lint: `ruff check src tests` must be clean for new/changed files.
- Real-hardware checks: `scripts/verify_phaseN.sh` prints PASS/FAIL/UNVERIFIED per
  item. Kanak runs them and the output is pasted into TASK_PLAN.md.

## 8. Security / privacy

Loopback-only Ollama (it refuses a non-loopback host in local-only mode). Consent
ledger unchanged. Takeover is disabled without an enrolled, consented voice. The
evdev reader opens keyboards read-only, looks only for the configured key code and
never logs keystrokes.

## 9. Open items

- O1 Venv Python 3.12 vs 3.14: 3.12 proposed (torch/XTTS wheels).
- O2 Default hotkeys: proposed Ctrl+Alt+Space (ptt), Ctrl+Alt+B (bypass),
  Ctrl+Alt+T (takeover), Ctrl+Alt+C (cancel), Ctrl+Alt+Esc (panic).
- O3 The committed `release/*.deb` binaries (Windows/Debian packaging) stay in the repo
  but aren't v1 deliverables; `install.sh` is the v1 install path.
