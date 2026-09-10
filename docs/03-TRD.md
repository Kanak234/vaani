# Technical Requirements Document — Vaani

**Phase:** 3 · Every technology below is justified, and alternatives are recorded
with the reason they were rejected. Popularity is not a reason.

---

## 1. System architecture

Single-process desktop application, three long-lived threads (capture / worker /
output), a bounded queue between capture and worker, and a SQLite store. See
`06-ARCHITECTURE.md` for the graphs and the state machine.

The one structural rule: `ai/` may not import `audio/`, and neither may import
`ui/`. Only `core/pipeline` knows about both. That is what makes providers
genuinely swappable instead of nominally abstract.

## 2. Language: Python 3.14

**Why:** the entire local speech stack (CTranslate2, ONNX Runtime, faster-whisper)
is Python-first; numpy handles the DSP at C speed; ctypes gives direct access to
the system audio libraries without a build step.

**Verified, not assumed:** Python 3.14 is new enough that wheel availability was
tested before adopting anything. `faster-whisper 1.2.1`, `ctranslate2 4.8.2`,
`onnxruntime 1.29.0`, `numpy 2.5.3`, `soundfile 0.14.0` all install cleanly.
`sounddevice` installs but fails at import — see §4.

**Alternatives rejected:** *Rust* — best latency and a real cpal/PipeWire story, but
the ML inference bindings are immature and it would triple implementation time for
a single-user tool. *C++* — same, more so. *Electron/TypeScript front + Python back*
— adds an IPC hop and ~150 MB of runtime for a UI that must show six numbers and
five buttons; reconsider only if a rich GUI becomes the priority.

**Risk accepted:** the GIL. Mitigated because the three threads are each dominated
by blocking I/O or C-extension calls that release it.

## 3. Audio server: PipeWire via the PulseAudio compatibility API

**Why:** PipeWire 1.6.2 is what the target machine runs, and its Pulse layer is the
common denominator every meeting application already speaks.

**Alternatives rejected:** *native libpipewire* — lower latency and finer routing
control, but a far larger C surface to bind through ctypes and it forfeits
compatibility with apps that use the Pulse API. *ALSA directly* — bypasses the
session mixer; cannot create a virtual device other apps can see. *JACK* — not
present on a normal desktop.

## 4. Audio I/O: ctypes binding to `libpulse-simple`

**Why:**
1. `libportaudio2` is **not installed** on the target machine and installing it
   needs root. `sounddevice` fails with `OSError: PortAudio library not found`.
   `libpulse-simple.so.0` is already present. No-sudo is a hard constraint (BRD C4).
2. PortAudio would sit on top of this same server anyway — one more layer whose
   buffering we do not control.
3. Latency is a first-class requirement and `fragsize`/`tlength` in
   `pa_buffer_attr` are precisely the knobs that set it.

**Measured:** capture latency **22.33 ms** at a 20 ms fragment; server-reported.
Playback into the virtual sink measured 58–65 ms.

**Alternatives rejected:** *sounddevice/PortAudio* — see above. *`pw-cat` subprocess
pipes* — works, but process boundaries and pipe buffering add uncontrolled latency
and make error handling coarse. *GStreamer* — a large dependency for a fixed,
simple graph.

## 5. Virtual microphone: two PipeWire modules

`module-null-sink` (write target) + `module-remap-source` (what apps read).

**This is documented in full as ADR-001, including the measured failure of the
obvious one-module alternative, which produces a device that every app lists and
that carries pure silence while the user's speech plays out of the laptop
speakers.** Do not "simplify" it back.

**Verified:** tone round trip 0.500 → 0.500; speech round trip cross-correlation
0.930, envelope correlation 0.981.

## 6. Voice activity detection: Silero VAD (ONNX), energy VAD as fallback

**Why Silero:** ~1.8 MB, materially better than energy or WebRTC on keyboard noise,
music and breath; permissive licence; runs on CPU in well under a millisecond per
32 ms frame.

**Why ONNX rather than the torch build:** the torch distribution is ~2.5 GB for a
1.8 MB model. On a machine that had **~2.9 GiB free RAM** at baseline, that is not
a trade worth making.

**CPU by necessity and by preference:** ONNX Runtime on this machine reports only
`['AzureExecutionProvider', 'CPUExecutionProvider']` — there is no CUDA EP. At this
model size CPU is the right answer regardless.

**Alternatives rejected:** *WebRTC VAD* — faster but noticeably worse on
non-stationary noise. *Energy-only* — kept as a zero-dependency fallback and for
isolating model-layer faults in diagnostics, but not the default.

## 7. Speech recognition: faster-whisper (CTranslate2)

**Why Whisper:** it is trained on genuinely multilingual audio and — decisively for
this product — treats language as a decoder token rather than a separate model, so
it transcribes code-mixed Hinglish without being forced into one language. That is
the property this product needs most (AC-04.2).

**Why the faster-whisper/CTranslate2 build:** int8 and float16 quantisation is what
makes a useful model fit the available memory; it does not pull torch; and it is
verified installable on CPython 3.14.

**Measured on this machine — real speech, 11.0 s sample:**

| Model | Device | Compute | Beam | Median | RTF |
|---|---|---|---|---|---|
| small | CPU (6C/12T) | int8 | 1 | 2883.6 ms | 0.262 |
| small | CPU | int8 | 5 | 2914.8 ms | 0.265 |
| small | **CUDA** | int8_float16 | 1 | 278.8 ms | **0.025** |
| small | CUDA | int8_float16 | 5 | 318.1 ms | 0.029 |
| medium | **CUDA** | int8_float16 | 1 | 667.9 ms | **0.061** |
| medium | CUDA | int8_float16 | 5 | 718.1 ms | 0.065 |

Transcripts correct in every configuration; language detected `en` at p≈0.94.

**Three findings that changed the design, none of them guessable:**

1. **`medium` on GPU (RTF 0.061) is four times faster than `small` on CPU (0.262).**
   When CUDA works, the accuracy-for-latency trade largely disappears and the
   better model is simply the better choice. Balanced mode therefore defaults to
   `medium` on GPU and steps down to `small` on CPU.
2. **Beam search is nearly free.** beam=5 costs ~1 % on CPU and ~16 % on GPU. The
   conventional advice to drop to greedy decoding for latency does not apply here,
   so beam=5 is the default outside low-latency mode.
3. **Short utterances pay a fixed cost.** Whisper pads every input to a 30 s
   window, so a 4 s utterance takes 586 ms while an 11 s one takes 713 ms — RTF
   0.146 vs 0.065. Latency is therefore roughly *constant*, not proportional, in
   the range this product cares about. Budgeting per-second of speech would have
   been wrong.

> **A correction worth recording.** An earlier run of this benchmark used a
> synthetic formant stack instead of speech and reported RTF 5.8–22.7. Those
> numbers were meaningless: on non-speech input Whisper hallucinates and decodes to
> its token limit, so the benchmark measured degenerate decoding, not
> transcription. Real speech was 40× faster. **Never benchmark an ASR model on
> synthetic audio.**

**GPU status — RESOLVED.** Initially inference failed with
`Library libcublas.so.12 is not found or cannot be loaded`, despite CTranslate2
reporting a CUDA device. No system CUDA toolkit is installed. Fixed by installing
`nvidia-cublas-cu12` and `nvidia-cudnn-cu12` from PyPI and preloading them with
`ctypes.CDLL(..., RTLD_GLOBAL)` before CTranslate2 initialises — see
`ai/stt/cuda_setup.py`. No root and no system CUDA toolkit required.

The failure mode is worth recording because it is silent in three separate ways:
`get_cuda_device_count()` returns 1, `WhisperModel(device="cuda")` constructs
successfully, and only the **first encode** fails — i.e. on the user's first
utterance, mid-meeting. Any diagnostic that checks CUDA by loading a model rather
than running inference will report a healthy GPU that does not work.

**Model sizing:** `medium` on GPU for balanced and quality, `small` for
low-latency, stepping down to `small` automatically when CUDA is unusable.
End-to-end measured through the provider: a 4 s utterance costs **586 ms** of STT
with `medium` on GPU — inside the 600 ms budget.

**Alternatives rejected:** *openai-whisper* — pulls torch, no int8, slower.
*transformers Whisper* — same. *Vosk* — streaming and light, but weak on Hindi and
poor on code-mixing. *Cloud STT* — better accuracy, but sends the user's raw audio
off the machine on every utterance and adds network RTT to the critical path;
retained as an optional provider, never the default.

## 8. Translation: NLLB-200 distilled 600M (CTranslate2)

**Why NLLB-200 600M:** it runs on the CTranslate2 runtime already used for STT, so
it adds no new inference stack and inherits the same int8 quantisation; ~600 MB at
int8 fits the memory this machine actually has free (the 1.3B and 3.3B variants do
not); and it is genuinely offline once downloaded, which is what makes local-only
mode real rather than aspirational.

**Alternatives rejected:** *IndicTrans2* — better on clean Indic→English but ships a
bespoke tokenizer and preprocessing chain and is larger; worth revisiting if quality
proves to be the blocker. *Cloud LLM* — materially better on code-mixed input and
the only option that genuinely uses conversational context; kept as a future
provider, not the default, because it puts transcript text on the network.

**Three limitations, stated because they are structural rather than bugs:**

1. **Code-mixing is out of distribution.** NLLB trained on monolingual sentence
   pairs. Hinglish is a register it never saw. It is handled by script-aware source
   tagging — if the text contains Devanagari we tag it Hindi regardless of the
   recogniser's dominant-language guess, because mis-tagging mixed text as English
   makes NLLB pass it through untranslated, which the gate then suppresses and the
   utterance is lost entirely.
2. **The model cannot use conversational context.** NLLB is sentence-level with no
   mechanism to condition on prior turns, so `used_context_turns` is honestly
   reported as 0. **This means the PRD's own motivating example (AC-06.2) —
   resolving "उसमें" against an earlier turn — is a case this engine cannot
   satisfy.** A context-capable engine is required for that, and the acceptance
   criterion stands unmet rather than being quietly reinterpreted.
3. **Romanised Hindi is the weak spot.** "mera point ye hai" is Latin script but
   Hindi language. The gate catches Devanagari leakage but not this.

**Runaway generation** is bounded with `max_decoding_length=256`,
`repetition_penalty=1.1` and `no_repeat_ngram_size=4`, because NLLB's failure mode
on out-of-distribution input is looping.

## 9. Voice synthesis: XTTS-v2, with an always-flagged fallback

**Why XTTS-v2:** zero-shot cloning from a short reference recording, so enrollment
is a two-minute task rather than a training job; supports both English and Hindi,
so one profile serves a user whose reference audio is code-mixed; and it runs
locally, keeping the voice profile — the most sensitive artefact this product
handles — on the machine.

**Alternatives rejected:** *Piper* — far faster and tiny, but cannot clone, which
is the entire promise. *OpenVoice v2* — tone conversion layered on a base TTS; more
moving parts and an extra resident model for a comparable result. *ElevenLabs and
similar* — better quality, but the voice leaves the machine.

**Consent is enforced in the synthesiser, not only in the UI.** `synthesize()`
re-checks the consent ledger on every call and refuses if consent has been revoked,
so a revocation takes effect immediately even if a profile directory survived. A
UI-only check would be bypassed by any future caller, and this is the one
capability where that matters.

**The fallback synthesiser remains** and always sets `is_fallback_voice=True`.
Silently substituting a different voice would break the product's core promise
without telling anyone.

## 9a. Memory budgeting

Three models now compete for a 4 GB laptop GPU: Whisper medium (~0.9 GB),
XTTS-v2 (~2.3 GB) and NLLB-600M (~0.8 GB). Measured free VRAM was **1610 MiB with
a browser open** and **3770 MiB idle** — so a design sized against the card's
nominal 4096 MiB would load fine on an idle machine and OOM the moment the user
opens the meeting client, which is precisely when the product runs.

`core/model_budget.py` therefore decides placement against live `nvidia-smi` free
memory, holds back a 400 MiB reserve for the CUDA context and compositor, and
demotes by priority: **STT keeps the GPU first** (largest latency contributor,
10x speedup), then TTS, then translation. It refuses to over-commit rather than
trying to be clever — a slower stage is a far better outcome than a mid-meeting OOM.

Measured plans, showing the policy adapting to live conditions:

| Free VRAM | Plan | Why |
|---|---|---|
| 3770 MiB (idle) | `stt=cuda, tts=cuda, translation=cpu` | 170 MiB left after the first two |
| 2682 MiB (NLLB loaded) | `stt=cuda, translation=cuda, tts=cpu` | 1382 MiB left; XTTS's 2300 MiB does not fit |
| 1610 MiB (browser open) | `stt=cuda, rest=cpu` | 310 MiB left after STT |

**Co-residency has a measured cost beyond OOM risk.** With NLLB sharing the GPU,
STT p50 rose from **578 ms to 838 ms** — 45 % slower with no change to the model.
End-to-end went 1103 ms → 1363 ms. Both are inside budget, but it means "does it
fit" is the wrong question; "what does it cost the stage that matters most" is the
right one.

## 10. Context: in-process bounded window

Last N turns (default 6), hard character cap, oldest-first eviction, per-session.
No vector store, no embedding model: the window that matters for pronoun resolution
is a handful of recent turns, and a retrieval system would add a model, latency and
a place for conversation data to accumulate — all to solve a problem this does not
have.

## 11. Storage: SQLite (WAL)

Single-user, single-process, zero-configuration, transactional. Schema in
`05-DATA-MODEL.md`. **No audio is ever stored in the database.** Transcript text is
opt-in and structurally absent when persistence is off, rather than deleted later.

**Alternatives rejected:** *Postgres* — a server process for one user. *JSON files*
— no transactions, no indexes, no retention queries.

## 12. Security

API keys in the OS keyring, referenced from the DB by name only, never written to
the repo or logs. Voice embeddings and enrollment audio encrypted at rest. Consent
is an append-only ledger — a revocation is a new row, never an update or a delete.
Local-only mode is enforced by the provider registry refusing to hand out a
network-requiring provider, not by each provider remembering to check.

## 13. Deployment

Linux/PipeWire, x86-64, Python ≥ 3.12. Installed into a venv; **no root required**
for any part of normal operation, including the virtual microphone. Windows and
macOS are out of scope for v1; the OS-specific code is confined to
`audio/backend/` and `devices/` so that a second backend is an addition, not a
rewrite. Virtual audio routing is genuinely different on each OS and this document
does not pretend otherwise.

## 14. Open technical risks

| # | Risk | Status |
|---|---|---|
| T-1 | Voice cloning may not fit in available memory | **RESOLVED.** XTTS-v2 measured at 1188 ms p50 / 2133 MiB peak on CUDA. Full pipeline 2411 ms vs 4000 ms budget. Fits when the GPU is idle; `model_budget` demotes TTS to CPU when it is not |
| T-2 | GPU unusable — `libcublas.so.12` missing | **RESOLVED.** pip CUDA libs + RTLD_GLOBAL preload; medium/GPU at RTF 0.061 |
| T-3 | Hinglish translation quality | **Partly addressed.** NLLB-200 implemented with script-aware tagging; evaluated against `tests/fixtures/hinglish_cases.py`. Context resolution (AC-06.2) remains unmet — see §8 |
| T-4 | Whisper accuracy on this user's accent | **Open.** Requires real user speech |
| T-5 | End-to-end latency budget | **MEASURED, every stage.** 22 + 500 + 578 + 123 + 1188 = **2411 ms** against a 4000 ms balanced budget |
