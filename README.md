# Vaani

Real-time personal voice translation and meeting assistant.

Speak Hindi, English or Hinglish; have it come out as English, in your own voice,
through a virtual microphone that Zoom, Google Meet and Teams see as an ordinary
input device.

> **Status: MVP, end to end.** Capture → VAD → recognition → translation →
> confidence gate → synthesis → virtual microphone all run, and every performance
> figure below was measured on real hardware. Read
> [What actually works](#what-actually-works) for the honest limits — notably that
> **romanised Hinglish is suppressed rather than translated**.

---

## What actually works

Everything in this table was run and measured on the development machine
(Ubuntu 26.04, PipeWire 1.6.2, Ryzen 5 5600H, RTX 3050 Mobile).

| Component | Status | Evidence |
|---|---|---|
| Audio capture | **Working** | 22.33 ms measured server-side latency |
| Virtual microphone | **Working** | Tone round trip 0.500 → 0.500; speech cross-correlation 0.930 |
| Device management | **Working** | Create, orphan reuse without stacking, feedback guard, clean teardown |
| VAD + segmentation | **Working** | 13 unit tests |
| Speech recognition | **Working** | Whisper medium on GPU, **578 ms** p50 per utterance, correct transcripts |
| Confidence gate | **Working** | Suppresses untranslated output; 21 tests |
| Confidence gate | **Working** | 16 tests; suppression is the default on low confidence |
| Pipeline + state machine | **Working** | 30 tests; audio provably unreachable from a failed stage |
| Diagnostics | **Working** | 8/8 checks pass with measured values |
| End-to-end latency | **Measured** | **2411 ms** with every stage real, against a 4000 ms budget |
| Translation | **Working** | Routing NMT+LLM: **14/14 (100%)** on all registers, 424 ms p50 |
| Voice enrollment + consent | **Working** | Append-only consent ledger, quality gating, 22 tests |
| Voice cloning (XTTS-v2) | **Working** | **1188 ms p50**, 2133 MiB peak VRAM, measured on this GPU |
| Desktop GUI | **Working** | 5 screens, tkinter, design system from `docs/04` |
| SQLite persistence | **Working** | Transcript text never written unless opted in |
| Latency budget warnings | **Working** | Warns on sustained breach, names the slow stage |
| Emergency stop | **Measured** | **0 ms** of audio heard past the stop (budget 200 ms) |
| Model memory scheduling | **Working** | Places models against *measured* free VRAM, not nominal |

### Full latency budget — every stage measured, none estimated

```
capture       22 ms
VAD hangover 500 ms   (tunable: 350 low-latency / 500 balanced / 800 quality)
STT          578 ms   Whisper medium, CUDA
translation  424 ms   routing: NLLB, or local LLM for romanised Hinglish
TTS         1188 ms   XTTS-v2, CUDA, your enrolled voice
------------------
total       2712 ms   vs a 4000 ms balanced budget
```

### Translation accuracy, measured

Against `tests/fixtures/hinglish_cases.py` — 14 cases, four registers:

| Register | NLLB alone | **Routing (default)** |
|---|---|---|
| Devanagari Hindi | 4/4 | **4/4** |
| Code-mixed Hinglish | 6/6 | **6/6** |
| English | 2/2 | **2/2** |
| Romanised Hindi | 0/2 | **2/2** |
| **Overall** | 12/14 (86%) | **14/14 (100%)** |
| Latency p50 | 87 ms | **424 ms** |

Routing sends Devanagari and English to NLLB (fast) and only romanised Hinglish
to a local LLM via Ollama — see [ADR-004](docs/decisions/ADR-004-translation-routing.md).
Ollama runs on loopback, so **this does not weaken the local-only guarantee**.

### Known limits, stated plainly
- **All three models do not fit on the GPU together.** STT (~900 MiB) + TTS
  (2133 MiB measured) is ~3.0 GiB. Free VRAM was **3770 MiB idle** but only
  **1610 MiB with a browser open** — i.e. under the conditions the product actually
  runs in, TTS gets demoted to CPU and will be far slower than 1188 ms. The
  scheduler makes that call against live VRAM rather than the card's nominal size.
- **Voice similarity is unverified.** All cloning numbers used a public-domain
  reference clip, not your voice. That measures mechanics and latency, not whether
  it sounds like you. **This is the single biggest untested area.**
- **Nothing has been tested on your actual speech.** Every STT figure comes from
  one English clip. Accuracy on your accent and your Hinglish is unknown.
- **Running the LLM costs GPU headroom.** Ollama holds ~2.7 GB, which pushes
  Whisper and XTTS to CPU. The scheduler handles it, but it is a real trade.

## Requirements

- Linux with PipeWire or PulseAudio (`libpulse-simple.so.0`, `pactl`)
- Python ≥ 3.12
- **No root required**, including for the virtual microphone

## Install

```bash
uv venv --python 3.14 .venv
```

```bash
uv pip install --python .venv/bin/python -e ".[stt,translate,cuda]"
```

Voice cloning is a separate, much larger install (~3 GB, pulls torch):

```bash
uv pip install --python .venv/bin/python -e ".[voice]"
```

**Two dependency traps, both pinned in `pyproject.toml`:** `coqui-tts` declares
`transformers>=4.57` with no upper bound, but transformers 5.x removed a function
it imports — so the newest version installs cleanly and then fails at import.
And PyTorch ≥ 2.9 needs `torchcodec` for audio IO, which is why the `[codec]`
extra is required.

## Use

Check that everything on this machine works, with real measurements:

```bash
PYTHONPATH=src .venv/bin/python -m vaani.cli doctor
```

List audio devices:

```bash
PYTHONPATH=src .venv/bin/python -m vaani.cli devices
```

Enroll your voice (records consent first, then ~90 s of speech):

```bash
PYTHONPATH=src .venv/bin/python -m vaani.cli consent grant
```

```bash
PYTHONPATH=src .venv/bin/python -m vaani.cli enroll record
```

Launch the desktop app:

```bash
PYTHONPATH=src .venv/bin/python -m vaani.cli gui
```

Or run a session in the terminal:

```bash
PYTHONPATH=src .venv/bin/python -m vaani.cli run --mode balanced
```

Check translation quality against the Hinglish fixtures:

```bash
.venv/bin/python scripts/eval_translation.py --engine nllb
```

Then select **"Vaani Virtual Microphone"** as your microphone in Zoom/Meet/Teams.

Prove that audio genuinely reaches the virtual mic:

```bash
PYTHONPATH=src .venv/bin/python tests/manual/verify_virtual_mic_audio.py
```

Measure end-to-end latency through the real pipeline:

```bash
.venv/bin/python scripts/bench_e2e.py your_speech_16k.wav --mode balanced
```

For GPU acceleration (10x faster STT, no root needed):

```bash
uv pip install --python .venv/bin/python nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

## Documentation

| Document | Contents |
|---|---|
| [00-ENVIRONMENT-BASELINE.md](docs/00-ENVIRONMENT-BASELINE.md) | What the target machine can actually do, measured |
| [01-BRD.md](docs/01-BRD.md) | Vision, personas, requirements, risks |
| [02-PRD.md](docs/02-PRD.md) | Features, user stories, measurable acceptance criteria |
| [03-TRD.md](docs/03-TRD.md) | Technology choices with rejected alternatives |
| [04-UIUX-BRIEF.md](docs/04-UIUX-BRIEF.md) | Design system, defined before implementation |
| [05-DATA-MODEL.md](docs/05-DATA-MODEL.md) | Schema, constraints, retention |
| [06-ARCHITECTURE.md](docs/06-ARCHITECTURE.md) | Graphs, state machine, data flow |
| [07-PROVIDER-EVALUATION.md](docs/07-PROVIDER-EVALUATION.md) | Local vs hybrid vs cloud, with measurements |
| [ADR-001](docs/decisions/ADR-001-virtual-microphone-topology.md) | Why the virtual mic needs two modules |
| [ADR-002](docs/decisions/ADR-002-silero-vad-context-window.md) | Why Silero VAD needs 576 samples, not 512 |
| [ADR-003](docs/decisions/ADR-003-untranslated-passthrough-detection.md) | Why the gate must detect untranslated output, not just wrong script |
| [ADR-004](docs/decisions/ADR-004-translation-routing.md) | Why translation routes per-utterance instead of picking one engine |
| [09-ACCEPTANCE-STATUS.md](docs/09-ACCEPTANCE-STATUS.md) | Every acceptance criterion: MET / UNMET, with evidence |

## Design principles

**Silence beats a confident mistranslation.** The gate suppresses by default when
confidence is low, and records why. A wrong sentence in your own voice, to your
colleagues, is not recoverable; a repeated sentence is.

**Never substitute a voice silently.** Any synthesiser that cannot use your enrolled
voice must set `is_fallback_voice=True`, and the UI must say so.

**Measure, never estimate.** Every performance number in these documents was
produced by running something. One benchmark in this project produced numbers that
were wrong by 40× and would have forced the wrong architecture; it is documented in
[07-PROVIDER-EVALUATION.md §2](docs/07-PROVIDER-EVALUATION.md) rather than deleted.

**Audio is the most sensitive thing here.** It is never persisted, and in local-only
mode it never leaves the machine.

## Not yet built

Keyring integration (no provider currently needs a secret — everything is local) ·
Windows/macOS audio backends · streaming partial STT and TTS · bidirectional
translation (understanding other participants).

**The most valuable next step is not a feature.** It is running this on the actual
user's voice and speech, which nothing here has done yet.
