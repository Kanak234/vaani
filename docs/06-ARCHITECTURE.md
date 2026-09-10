# Architecture — Vaani

**Phase:** 6

---

## A. High-level system graph

```
┌──────────┐   analog    ┌───────────────┐
│  USER    │────────────▶│ Physical mic  │
└──────────┘   speech    └───────┬───────┘
     ▲                           │ PCM s16le 16 kHz mono, 20 ms frames
     │ hears self                ▼
     │ (monitor,          ┌─────────────────────────────────────┐
     │  optional)         │           AUDIO ENGINE              │
     │                    │  capture → resample → gain → ring   │
     │                    └───────────────┬─────────────────────┘
     │                                    ▼
     │                    ┌─────────────────────────────────────┐
     │                    │  VAD (Silero, ONNX, CPU)            │
     │                    │  emits utterance boundaries         │
     │                    └───────────────┬─────────────────────┘
     │                                    ▼  utterance PCM
     │                    ┌─────────────────────────────────────┐
     │                    │  STT  (faster-whisper / CTranslate2) │
     │                    │  → text + per-language probabilities │
     │                    └───────────────┬─────────────────────┘
     │                                    ▼
     │                    ┌─────────────────────────────────────┐
     │                    │  LANGUAGE ID (script + model prior) │
     │                    └───────────────┬─────────────────────┘
     │                                    ▼
     │                    ┌─────────────────────────────────────┐
     │                    │  CONTEXT ENGINE (last N turns)      │
     │                    └───────────────┬─────────────────────┘
     │                                    ▼
     │                    ┌─────────────────────────────────────┐
     │                    │  TRANSLATION  (local NMT | cloud LLM)│
     │                    └───────────────┬─────────────────────┘
     │                                    ▼
     │                    ┌─────────────────────────────────────┐
     │                    │  CONFIDENCE GATE ── fail ──▶ SUPPRESS│
     │                    └───────────────┬─────────────────────┘
     │                                    ▼ pass
     │                    ┌─────────────────────────────────────┐
     │                    │  NORMALIZE → TTS (personal voice)   │
     │                    └───────────────┬─────────────────────┘
     │                                    ▼
     │                    ┌─────────────────────────────────────┐
     └────────────────────│  OUTPUT BUFFER → jitter smoothing   │
                          └───────────────┬─────────────────────┘
                                          ▼
                          ┌─────────────────────────────────────┐
                          │ PipeWire virtual source             │
                          │ "Vaani Virtual Microphone"          │
                          └───────────────┬─────────────────────┘
                                          ▼
                          ┌─────────────────────────────────────┐
                          │ Zoom · Google Meet · Teams · any     │
                          │ PulseAudio/PipeWire client           │
                          └─────────────────────────────────────┘
```

**Trust boundary.** Everything above the virtual source runs in-process on the user's
machine. The only arrows that may leave the machine are inside the STT and
TRANSLATION boxes, and only when a cloud provider is selected — which is why the
privacy indicator is bound to exactly those two stages.

## B. Component tree

```
vaani/
├── core/                  # orchestration; owns nothing platform-specific
│   ├── pipeline           # stage sequencing, per-stage timing
│   ├── state_machine      # authoritative session state
│   ├── events             # in-process pub/sub (engine → UI)
│   └── errors             # typed error taxonomy
├── audio/
│   ├── backend/           # PLATFORM-SPECIFIC, behind AudioBackend
│   │   └── pulse/         #   libpulse-simple via ctypes (Linux v1)
│   ├── capture            # frame pump, ring buffer
│   ├── output             # playback to virtual sink + monitor
│   ├── resample           # 48k↔16k
│   ├── dsp                # gain, DC block, level metering
│   └── virtual_mic        # PipeWire module lifecycle + orphan reclaim
├── ai/
│   ├── vad/               # VadProvider  → SileroVad
│   ├── stt/               # SpeechRecognizer → FasterWhisper | cloud
│   ├── langid/            # code-mix language identification
│   ├── translate/         # TranslationEngine → local NMT | LLM | passthrough
│   └── tts/               # VoiceSynthesizer → cloning | fallback
├── voice/                 # enrollment, quality checks, embedding storage
├── context/               # bounded conversation window, reset, isolation
├── session/               # session lifecycle, persistence, crash recovery
├── devices/               # enumeration, selection, hotplug watch
├── security/              # keyring, encryption at rest, redaction
├── storage/               # SQLite schema, migrations, retention job
├── diagnostics/           # the checks from PRD §9
├── config/                # typed settings, validation, defaults
├── ipc/                   # engine ↔ UI transport
└── ui/                    # presentation only; no engine logic
```

**Rule that keeps this honest:** `ai/` may not import `audio/`, and neither may
import `ui/`. The pipeline in `core/` is the only module that knows about both. This
is what makes providers swappable (FR-13) rather than nominally abstract.

## C. Processing chain with latency budget

Budgets below are **measured on the target machine** in balanced mode unless
stated. Figures marked *to measure* are stages not yet implemented.

| # | Stage | Budget | **Measured** |
|---|---|---|---|
| 1 | Capture (frame available) | 30 ms | **22.3 ms** ✅ |
| 2 | Resample + DSP | 5 ms | in-server, included above |
| 3 | VAD decision (Silero, CPU) | 10 ms | **0.17 ms** mean, 2.5 ms max ✅ |
| 4 | End-of-speech hangover | 500 ms | **500 ms** (fixed, tunable per mode) |
| 5 | STT (medium, CUDA, beam=5) | 600 ms | **578.4 ms** p50 ✅ |
| 6 | Language ID | 5 ms | **< 0.1 ms** (script ratio + model prior) ✅ |
| 7 | Context assembly | 5 ms | **< 0.1 ms** ✅ |
| 8 | Translation | 400 ms | *pass-through only — real engine not built* |
| 9 | Confidence gate + normalize | 10 ms | **< 0.1 ms** ✅ |
| 10 | TTS | 800 ms | **2.4 ms** (fallback synth; *cloning not built*) |
| 11 | Post-process + buffer | 100 ms | *to measure* |
| 12 | Virtual mic write | 30 ms | **58–65 ms** (server-reported) |
| | **Measured user-perceived p50** | **≤ 4000 ms** | **1103 ms** ✅ |

Measured end-to-end through the real pipeline: p50 **580.9 ms**, p95 588.3 ms,
plus 22.3 ms capture and the 500 ms hangover.

**GPU contention is real and measurable.** With NLLB-200 also resident on the card,
the same STT stage rose from **578 ms to 838 ms p50** — a 45 % increase from
sharing a 4 GB laptop GPU, with no change to the model or the audio. User-perceived
p50 went 1103 ms → 1363 ms. This is why `core/model_budget.py` decides placement
against live free VRAM: the cost of co-residency is not just the risk of OOM, it is
a measurable slowdown in the stage that matters most.

Note also that `translate` reads 0.0 ms in that run: the input was English, so the
pipeline correctly bypassed NLLB via the pass-through engine. NLLB's own latency was
measured separately at **123 ms p50** across the Hinglish fixtures.

**Headroom: roughly 2.9 s remains** for a real translation engine and a
voice-cloning TTS before the balanced budget is threatened. That is the number that
matters when choosing those two components.

Two observations that shape the design:

- **Stage 4 is a pure tradeoff, not an inefficiency.** The hangover is how long we
  wait after speech stops before deciding the utterance ended. Shorter means faster
  output but more mid-sentence cuts. It is the single most impactful tuning knob and
  is exposed per performance mode.
- **Stage 5 dominates today, and stage 4 is second.** Together they are 93 % of
  measured latency (578 + 500 of 1103 ms). Every other implemented stage is under
  a millisecond, so optimising them is wasted effort — a fact worth knowing before
  spending a day on the DSP path.
- **Stage 10 will dominate once cloning is added.** Neural TTS is typically the
  most expensive stage in a pipeline like this, and the 2.4 ms measured here is the
  fallback synthesiser, not a real voice model.

## D. State machine

```
                    ┌──────────────────────────────────┐
                    ▼                                  │
   ┌──────┐    ┌──────────────┐    ┌───────────┐       │
   │ IDLE │───▶│ INITIALIZING │───▶│ LISTENING │◀──────┤
   └──────┘    └──────┬───────┘    └─────┬─────┘       │
       ▲              │ fail             │ speech      │
       │              ▼                  ▼             │
       │        ┌───────────┐      ┌──────────────┐    │
       │        │  FAILED   │      │ TRANSCRIBING │    │
       │        └─────┬─────┘      └──────┬───────┘    │
       │              │                   ▼            │
       │              │            ┌─────────────┐     │
       │              │            │ TRANSLATING │     │
       │              │            └──────┬──────┘     │
       │              │                   ▼            │
       │              │            ┌──────────────┐    │
       │              │            │  GATING      │    │
       │              │            └───┬──────┬───┘    │
       │              │        pass    │      │ fail   │
       │              │                ▼      ▼        │
       │              │       ┌────────────┐ ┌────────┐│
       │              │       │SYNTHESIZING│ │SUPPRESS││
       │              │       └─────┬──────┘ └───┬────┘│
       │              │             ▼            │     │
       │              │       ┌───────────┐      │     │
       │              │       │ OUTPUTTING│──────┴─────┘
       │              │       └───────────┘
       │              │
       │              ▼
       │        ┌──────────────┐     ┌───────────┐
       └────────│  RECOVERING  │◀────│ DEGRADED  │
                └──────────────┘     └───────────┘

  Cross-cutting: PAUSED  ← from LISTENING/any active state, models stay warm
                 MUTED   ← capture stopped at source
                 STOPPING → teardown → IDLE
  EMERGENCY_STOP: reachable from ANY state → flush buffer, silence, → IDLE
```

**Error and recovery states.**
- `DEGRADED` — a provider failed over to a lower-quality one; the session continues
  and the UI says so. Not an error state; a *disclosed* state.
- `RECOVERING` — device lost or provider unhealthy; retrying with backoff. Capture is
  paused; no audio is emitted while here.
- `FAILED` — initialization could not complete. Terminal; requires user action.
- `SUPPRESSED` is a *normal* terminal state for an utterance, not an error. This is
  the architectural expression of FR-15: refusing to speak is a success path.

Invariant, enforced in code and asserted by tests: **audio can only be written to the
virtual microphone from `OUTPUTTING`.** Every other state writes silence. There is no
code path from a failed stage to the speaker.

## E. Data flow — what is generated, moved, stored, deleted

| Data | Created | Lives | Leaves machine? | Stored? | Deleted |
|---|---|---|---|---|---|
| Raw mic frames | Capture | Ring buffer, RAM | **Never** | **Never** | Overwritten continuously |
| Utterance PCM | VAD | RAM | Only if cloud STT | **Never** | At utterance end |
| Transcript text | STT | RAM | Only if cloud translation | Opt-in only | Retention job |
| Language distribution | LangID | RAM → DB | No | Yes (metadata) | 90 d |
| Context window | Context | RAM | With cloud translation | **Never** | Session end / reset |
| Translated text | Translation | RAM | No | Opt-in only | Retention job |
| Synthesised audio | TTS | RAM → buffer | **Never** | **Never** | After playout |
| Voice embedding | Enrollment | Encrypted file | **Never** | Yes | User deletion |
| Enrollment audio | Enrollment | Encrypted dir | **Never** | Yes | User deletion |
| Consent record | Enrollment | DB | **Never** | Yes, permanent | Never |
| Latency metrics | Pipeline | DB | **Never** | Yes (no content) | 90 d |
| API keys | User | **OS keyring** | Only as auth header | Not in DB/repo | User action |

**In fully local mode, the "leaves machine" column is empty for every row.** That is
the claim the local-only mode makes, and Phase 10 must verify it with the network
physically disabled before it may be stated as fact anywhere in the UI.
