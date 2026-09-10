# Repository Structure

**Phase:** 8 · Every directory has one responsibility. Derived from the architecture
in `06-ARCHITECTURE.md`, not from a generic template.

```
vaani/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 00-ENVIRONMENT-BASELINE.md    measured capability of the target machine
│   ├── 01-BRD.md  02-PRD.md  03-TRD.md
│   ├── 04-UIUX-BRIEF.md  05-DATA-MODEL.md
│   ├── 06-ARCHITECTURE.md  07-PROVIDER-EVALUATION.md
│   ├── 08-REPOSITORY-STRUCTURE.md
│   └── decisions/                    ADRs: decisions with a measured rationale
├── src/vaani/
│   ├── cli.py                        entry point
│   ├── core/                         orchestration; nothing platform-specific
│   │   ├── types.py                  value types crossing module boundaries
│   │   ├── errors.py                 closed error taxonomy
│   │   ├── gate.py                   THE safety component: speak or stay silent
│   │   ├── state_machine.py          enforces "only OUTPUTTING emits audio"
│   │   └── pipeline.py               stage sequencing + per-stage timing
│   ├── audio/
│   │   ├── backend/                  PLATFORM-SPECIFIC, isolated here
│   │   │   ├── pulse_bindings.py     ctypes -> libpulse-simple
│   │   │   └── pulse_backend.py      capture/playback streams
│   │   └── segmenter.py              utterance boundaries; owns the latency knob
│   ├── ai/
│   │   ├── base.py                   provider interfaces (Protocols + ABCs)
│   │   ├── vad/                      silero.py, energy.py
│   │   ├── stt/                      whisper.py
│   │   ├── translate/                passthrough.py  (+ future engines)
│   │   └── tts/                      fallback.py     (+ future cloning backend)
│   ├── context/engine.py             bounded, session-isolated conversation memory
│   ├── devices/manager.py            enumeration + virtual mic lifecycle
│   ├── session/engine.py             threads, device lifecycle, audio path
│   ├── diagnostics/checks.py         measured health checks
│   ├── voice/    security/    storage/    config/    ipc/    ui/    (scaffolded)
├── tests/
│   ├── unit/                         segmenter, gate, state machine
│   ├── integration/                  full pipeline with stub providers
│   └── manual/                       verify_virtual_mic_audio.py  (needs real audio)
└── scripts/bench_stt.py              real-speech STT benchmark
```

## Rules that keep the structure honest

1. **`ai/` may not import `audio/`; neither may import `ui/`.** Only `core/pipeline`
   knows about both. This is what makes providers swappable in practice.
2. **All OS-specific code lives in `audio/backend/` and `devices/`.** A second
   platform is an addition there, not a rewrite elsewhere.
3. **`core/` has no I/O.** It can be tested without a sound card, which is why the
   pipeline tests run in 0.2 s.
4. **`tests/manual/` is for tests needing real audio hardware.** They are not
   optional — ADR-001's bug was invisible to every test that did not capture from
   the virtual device.
