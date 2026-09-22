# Vaani v1.0 — Product Requirements (source of truth)

**Status:** Phase 0 draft, 2026-09-22. It supersedes `02-PRD.md` for v1.0 scope.
`02-PRD.md` stays in the repo as the historical record, and its F-xx acceptance
criteria that are already MET still apply unless something below changes them.

## 1. Overview

Vaani is a private, local-only voice translator for one user (Kanak). He speaks
Hindi, English or Hinglish. Vaani says it in English **in his own cloned voice**
through a PipeWire virtual microphone, which Zoom, Google Meet and Teams show as
an ordinary input device.

## 2. Problem

Kanak thinks and speaks most fluently in Hindi/Hinglish. In English meetings and
presentations, the language barrier stops him expressing ideas he fully
understands. Existing tools either use the cloud, use a stranger's voice, or can't
act as a microphone for a meeting.

## 3. Goals / non-goals

**Goals (v1.0)**
- G1 Real-time meeting translation in his own voice, usable every day.
- G2 Runs fully offline on both of his machines, on GPU **or** CPU.
- G3 He is always in control: bypass, push-to-talk, takeover cancel, and a
  panic stop, all one keypress away.
- G4 An offline script mode for preparing presentations.
- G5 One-step install and one launch command.

**Non-goals (v1.0)**
- Cloud APIs of any kind. Windows or macOS as *supported* targets (the Windows
  code on this branch stays but isn't maintained or verified for v1).
- Translating *other* participants for Kanak to hear (bidirectional).
- Cloning anyone's voice except the consenting enrolled user's.

## 4. Users

A single user: Kanak, a BCA student and developer, comfortable in a terminal. He
uses the laptop (Ubuntu, GNOME/Wayland) and the desktop (Kubuntu, KDE/Wayland).

## 5. Features

| ID | Feature | Phase |
|---|---|---|
| V-1 | Hardened core pipeline (VAD→STT→translate→TTS→virtual mic) | 1 |
| V-2 | Meeting mode: always-on and push-to-talk input modes | 2 |
| V-3 | Bypass: pass the real voice through untranslated | 2 |
| V-4 | Live latency display | 2 |
| V-5 | Virtual mic auto-created on start, removed on exit (crash-safe) | 2 |
| V-6 | Hardware + model auto-detection, quick benchmark, choice explained | 3 |
| V-7 | Mid-session automatic model fallback | 3 |
| V-8 | Takeover A: continue *my* point when I stall or press a hotkey | 4 |
| V-9 | Takeover B: answer the other participant's question (armed only) | 4 |
| V-10 | Takeover indicator + instant cancel | 4 |
| V-11 | Offline script mode: audio file/recording → English script (+ optional audio) | 5 |
| V-12 | Minimal control window, config file, install.sh, README | 6 |

## 6. Exact behaviour

### V-2 Input modes
- **always-on** (default): every utterance the VAD detects is translated.
- **push-to-talk, toggle**: `vaani ptt` (bound to a desktop shortcut) opens the
  gate and a second press closes it. The utterance in progress when it closes is
  finished, not cut off. It needs no root.
- **push-to-talk, hold**: optional. It's enabled only if the user is in the
  `input` group (a one-time sudo step that `install.sh --hold-to-talk` explains and
  runs **only after asking**). Speech is captured only while the key is held.
- The overlay also has an on-screen hold-to-talk button.
- A mode switch takes effect at the next utterance boundary.

### V-3 Bypass
- `vaani bypass` toggles it. While bypass is on, the raw microphone signal is
  routed straight to the virtual mic, with no STT/translation/TTS. Any queued
  translated audio is flushed first, so the two never overlap.
- The overlay shows a distinct BYPASS state (see UI_UX_BRIEF).

### V-4 Latency
- The overlay shows the last utterance's end-to-end latency and the rolling p50,
  updated per utterance. The existing budget warning (AC-13.3) stays.

### V-5 Virtual mic lifecycle
- It's created (or an orphan is reused) at meeting start and unloaded on normal
  exit, on `panic`, on SIGINT/SIGTERM, and on the next start after a crash.
  `vaani fix-audio` stays as the manual escape hatch.

### V-6 / V-7 Auto model detection
- At startup, Vaani detects: NVIDIA GPU present, a working CUDA (proprietary driver)
  vs Nouveau/NVK (treated as **no CUDA**), free VRAM, free RAM, whether Ollama is
  running, and its installed models.
- Quick benchmark (≤20 s total, results cached per machine plus model list, redone
  when either changes): Whisper size/device, and each candidate Ollama model on
  the fixed Hinglish fixtures (correctness first, latency second).
- **No model name is hardcoded as the only choice.** The config key
  `llm_model = "auto"` means "pick from what's installed"; the preference order
  lives in config.
- Every choice is shown with a one-line reason, e.g. `LLM: phi4-mini — 3/3 fixtures,
  410 ms (qwen3-coder: 2/3)`.
- Mid-session: if a stage raises, times out (> 3× its p50 and > 5 s), or runs out of
  memory, the utterance is suppressed (never spoken wrong), the next candidate
  is loaded, and the overlay shows `fallback: <old> → <new> (<reason>)`. The
  chain for each stage: GPU→CPU, then the next-ranked model. If the chain runs out,
  translation drops to NLLB-only, and TTS drops to the fallback voice **with the
  mandatory fallback-voice warning**.

### V-8 Takeover A: continue my point
- Triggers: (a) Kanak goes silent for `takeover.pause_s` (default 4.0 s)
  **mid-thought**, meaning his last utterance didn't end as a complete sentence
  per the LLM/heuristic check; (b) `vaani takeover` hotkey, which fires at once.
- Automatic triggering is **armed per session** (off by default). The hotkey
  always works.
- The LLM gets the conversation so far (his translated English turns, plus the
  remote participant's turns if Takeover B capture is on) and writes 1–3 English
  sentences continuing his point, in first person, without inventing facts,
  numbers or commitments he didn't state.
- The text goes through XTTS in his voice to the virtual mic.
- If he starts speaking again, takeover stops at the next sentence boundary and
  control returns to him.

### V-9 Takeover B: answer the other person
- It needs a second capture source (meeting speaker monitor) and must be
  explicitly armed. Behaviour is as in `docs/11-MEETING-TAKEOVER.md`: pending
  question + hesitation + silence window, or the phrase/hotkey.

### V-10 Indicator + cancel
- While AI speech is playing, the overlay shows a red **"AI SPEAKING"** banner
  that can't be missed, and the tray/window title changes.
- `vaani cancel` (hotkey) stops playback within 200 ms, flushes queued audio and
  returns to Kanak's normal mode. `panic` still ends everything.

### V-11 Offline script mode
- `vaani script <file.wav|mp3|m4a|…>` or `vaani script --record` (press Enter to
  stop).
- Output, written next to the input (or `--out DIR`): `<name>.en.txt` (English
  script, one paragraph per speech segment), `<name>.transcript.txt` (the original
  Hindi/Hinglish transcript), and with `--audio`, `<name>.en.wav` in his voice.
- Long files are processed in checkpointed chunks, and a crash resumes from the
  last finished chunk (for the i9 desktop's instability).
- In offline mode quality comes before latency: the larger Whisper model and the
  LLM path are allowed.

## 7. Functional requirements (short form)

FR-1 Nothing leaves the machine (local-only mode stays a hard error for
networked providers). FR-2 Raw audio is never written to disk, except for
offline script mode's own input/output files. FR-3 No output is spoken from a
failed or low-confidence stage (existing gate). FR-4 Every hotkey action is also
a CLI command (`vaani <cmd>`) over the existing IPC socket.

## 8. Edge cases

No mic, or the mic unplugged mid-session → pause with a banner, resume when it
comes back. Ollama not running → NLLB-only, with a banner. No voice enrolled →
fallback voice plus warning, and takeover is disabled (it would not be "his
voice"). Another Vaani already running → the second instance refuses and
tells you. The meeting app grabbed the real mic → `vaani route` / the auto-route
fix. VRAM taken by a browser → CPU placement (existing scheduler).

## 9. Acceptance criteria (v1.0)

| # | Criterion | How verified |
|---|---|---|
| A1 | Fresh clone → `./install.sh` → `vaani` works with no manual steps | verify_phase6.sh on the laptop |
| A2 | Virtual mic visible and usable in a browser Meet call | manual, on the laptop |
| A3 | Meeting mode 30+ min, no crash, RSS growth < 10 % after warm-up | `tests/manual/soak_test.py` on the laptop |
| A4 | GPU used when CUDA works; CPU fallback when it doesn't | verify_phase3.sh with `VAANI_FORCE_CPU=1` plus a normal run |
| A5 | Auto-detection chooses and explains; mid-session fallback works | unit tests with fault injection + a real run |
| A6 | Takeover triggers, speaks in his voice, cancels ≤200 ms | unit tests + manual |
| A7 | Offline script mode gives a correct English script | Hinglish fixtures + his own recording |
| A8 | Code heavily commented (rule 4); docs match implementation | review in Phase 7 |

Anything not run on Kanak's real hardware is reported as **UNVERIFIED**, never
MET.

## 10. Confirmed decisions (2026-09-22)

The user answered "all" to each Phase 0 question. It's read as:
- D1 Base: `v1-complete` branches from `feature/meeting-takeover`, which contains
  `main` plus all later work. Nothing is deleted. Windows code is kept but out of v1
  scope.
- D2 Takeover: **both** A (continue my point) and B (answer the other person).
- D3 PTT: **both** toggle (default, no root) and hold (optional, `input` group).
- D4 Verification: **both**. Per-phase `scripts/verify_phaseN.sh` for Kanak to run,
  and TASK_PLAN.md kept so a local Claude Code session can resume.
