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

One step:

```bash
./install.sh
```

That creates the virtualenv, installs the dependencies, **downloads the models
up front**, checks the virtual microphone can be created, and installs the
`vaani` launcher and desktop entry. Re-running it is safe and is the normal way
to repair a half-finished install — it reuses the venv and skips models it
already has (a second run takes about 11 seconds).

Why models are fetched during install rather than on first run: without it,
`vaani meeting` prints one line and then sits silent for several minutes while
~1.5 GB downloads, with no progress reaching the terminal. That is
indistinguishable from a hang — it was mistaken for one three times during
development. A long wait during an install is expected; a long silence on first
launch is a bug.

Options:

```bash
./install.sh --cpu          # no NVIDIA GPU, or Nouveau/NVK instead of the proprietary driver
./install.sh --no-voice     # skip XTTS (~1.8 GB); the fallback voice still works, but it is not YOUR voice
./install.sh --no-models    # code only; models download on first run instead
./install.sh --python PATH  # pin an interpreter
```

If `vaani` is not found afterwards, `~/.local/bin` is not on your PATH:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && exec bash
```

<details>
<summary>Manual install, if you would rather not run the script</summary>

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e ".[stt,translate,cuda]"
uv pip install --python .venv/bin/python -e ".[voice]"   # voice cloning, ~3 GB
.venv/bin/python -m vaani fetch-models --voice
```

</details>

**Two dependency traps, both pinned in `pyproject.toml`:** `coqui-tts` declares
`transformers>=4.57` with no upper bound, but transformers 5.x removed a function
it imports — so the newest version installs cleanly and then fails at import.
And PyTorch ≥ 2.9 needs `torchcodec` for audio IO, which is why the `[codec]`
extra is required.

## Use

After `./install.sh` the `vaani` command is on your PATH. (Without installing,
every command below also works as
`PYTHONPATH=src .venv/bin/python -m vaani ...`.)

**First time:**

```bash
vaani doctor            # check audio, models and the virtual mic, with real measurements
vaani models            # what Vaani will use for each stage, and WHY
vaani consent grant     # required before enrollment
vaani enroll record     # ~90 s of your speech, so it can be YOUR voice
```

**In a meeting:**

```bash
vaani meeting           # start; the virtual mic is created automatically
```

Then select **"Vaani Virtual Microphone"** as the microphone *inside* your
meeting app — browsers keep their own device list.

Everything below works from any terminal while a meeting is running, and is
meant to be bound to desktop shortcuts. None of it needs special permissions.

| Command | What it does |
|---|---|
| `vaani status` | is it live, muted, or passing your real voice through? |
| `vaani talk` | push-to-talk: toggle whether you are live |
| `vaani bypass` | put YOUR OWN voice on the mic, untranslated |
| `vaani continue` | finish your own point, in your voice |
| `vaani cancel` | **stop Vaani speaking for you, immediately** |
| `vaani mute` / `unmute` | stop capturing entirely |
| `vaani panic` | emergency stop and exit |
| `vaani toggle` | show/hide the overlay |

Input modes: `vaani meeting --input-mode always_on` (default), `ptt_toggle`, or
`ptt_hold`. Switch mid-meeting with `vaani mode-ptt-toggle` and friends.

**Presentation prep, offline:**

```bash
vaani script recording.wav --out script.txt          # Hindi/Hinglish in, English script out
vaani script --record 120 --out script.txt           # record from the mic instead
vaani script recording.wav --audio script.wav        # also synthesise it in your voice
```

Long recordings are checkpointed after every chunk, so an interrupted run
resumes where it stopped rather than starting over.

**Checking and measuring:**

```bash
vaani devices                                        # list audio devices
vaani fetch-models --voice                           # download everything up front
vaani fix-audio                                      # undo routing after a kill -9
.venv/bin/python scripts/eval_translation.py --engine nllb
.venv/bin/python scripts/bench_e2e.py your_speech_16k.wav --mode balanced
.venv/bin/python scripts/soak_test.py --minutes 30   # memory over a long session
PYTHONPATH=src .venv/bin/python tests/manual/verify_virtual_mic_audio.py
```

## Troubleshooting

Every entry here is a failure that actually happened during development, not a
guess at what might go wrong.

**`vaani meeting` prints one line and then nothing for minutes.**
It is downloading a model, not hanging. Run `./install.sh` (or
`vaani fetch-models`) to get that over with once. `vaani models` shows which
models are cached and which are not.

**The meeting hears my untranslated voice.**
The meeting app is still on your real microphone. Select **Vaani Virtual
Microphone** inside the app — browsers keep their own device list, separate from
the system default. `vaani status` says `meeting app NOT CONNECTED` when this is
the problem.

**Nothing reaches the meeting at all.**
Check `vaani status`:
- `input MUTED` — you are in a push-to-talk mode and not live. Press `vaani talk`.
- `input BYPASS` — the room is hearing your real, untranslated voice. `vaani bypass-off`.

**Vaani drops everything I say / every utterance is suppressed.**
Check which translation model is in use with `vaani models`. A custom Ollama
build can carry a system prompt that defeats translation — one on this machine
says *"Reply in the same language he uses (English or Hindi)"*, which produces
Devanagari that the output gate then correctly suppresses. Vaani now prefers the
plain base model over a derived build for exactly this reason, but a pinned
`--llm-model` overrides that.

**The microphone stops working after Vaani is killed.**
`vaani fix-audio` undoes the routing. This only happens after `kill -9`; a
normal exit or Ctrl-C restores applications automatically.

**`vaani meeting --input-mode ptt_hold` says push-to-talk is unavailable.**
True hold-to-talk needs raw keyboard access, which a fresh install does not
have:

```bash
.venv/bin/python -m pip install evdev
sudo usermod -aG input $USER     # then log out and back in
```

Vaani falls back to `ptt_toggle` and says so. `vaani talk` needs no special
permissions.

**The script from `vaani script` says "Thanks for watching!"**
Whisper hallucinates YouTube boilerplate when fed audio it cannot understand —
silence, noise, or a bad recording. Text it is unsure about is marked `[?]` in
the script, so check those. If everything is marked, the recording is the
problem.

**It is using the CPU when I have a GPU.**
`vaani models` prints the reason. Usually the proprietary NVIDIA driver is not
installed (Nouveau/NVK cannot run CUDA). Vaani still works, just slower, and it
steps Whisper down from `medium` to `small` because `medium` on CPU would blow
every latency budget. `VAANI_FORCE_CPU=1` forces this path deliberately.

**Something else.**
`vaani doctor` checks audio, models and the virtual mic with real measurements.
Logs are at `~/.local/share/vaani/logs/vaani.log` (structured JSON, rotated at
5 MB x 3); `vaani --log-level DEBUG meeting` makes them verbose.

## Cross-Platform Quickstart

### Windows (10/11 x64)
1. **Prerequisite**: Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) and [Ollama](https://ollama.com).
2. **Launch Application**:
   - Double-click `vaani.bat` or run:
     ```cmd
     .\vaani.bat gui
     ```
   - Or install using the generated installer in `release/windows/`.
3. Select **"CABLE Output (VB-Audio Virtual Cable)"** as microphone in Zoom/Meet/Teams.

### Linux / Debian
1. **Install Package**:
   ```bash
   sudo dpkg -i release/debian/vaani_1.0.0_amd64.deb
   ```
   Or launch directly via virtual environment:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e ".[stt,translate]"
   .venv/bin/python -m vaani.cli gui
   ```
2. Select **"Vaani Virtual Microphone"** as microphone in Zoom/Meet/Teams.

## Testing & Verification

```bash
# Run release test suite
bash scripts/test_release.sh

# Run quick hardware and pipeline smoke test
bash scripts/smoke_test.sh
```

## Documentation

| Document | Contents |
|---|---|
| [14-WINDOWS-PRODUCTION.md](docs/14-WINDOWS-PRODUCTION.md) | Complete Windows production guide, WASAPI topology, and GPU probing |
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
produced by running something.

**Audio is the most sensitive thing here.** It is never persisted, and in local-only
mode it never leaves the machine.
