# Provider & Architecture Evaluation — Local vs Hybrid vs Cloud

**Phase:** 7 · Grounded in measurements from this machine. Where a number is not
yet measured, it says so rather than guessing.

---

## 1. What the hardware actually allows

| Resource | Nominal | **Actually available** | Consequence |
|---|---|---|---|
| VRAM | 4096 MiB | **1610 MiB free** (2161 MiB already in use) | One small model, or none |
| RAM | 14 GiB | **2.9 GiB free at baseline** (7.3 GiB later) | Varies with what else is open |
| CPU | 6C/12T Ryzen 5 5600H | Fully available | Carries the whole pipeline today |
| GPU compute | RTX 3050 | **Usable after installing pip CUDA libs** | 10x speedup; see §2 |

The gap between nominal and available is the single most important fact in this
document. A design sized against "4 GB VRAM, 14 GB RAM" would fail on this machine
in a real session, because a browser and a meeting client are also running — which
is, definitionally, when this product is used.

## 2. Measured: speech recognition

Real speech, 11.0 s sample, `faster-whisper` int8 / int8_float16:

| Model | Device | Beam | Median | RTF |
|---|---|---|---|---|
| small | CPU | 1 | 2883.6 ms | 0.262 |
| small | CPU | 5 | 2914.8 ms | 0.265 |
| small | CUDA | 1 | 278.8 ms | **0.025** |
| small | CUDA | 5 | 318.1 ms | 0.029 |
| medium | CUDA | 1 | 667.9 ms | **0.061** |
| medium | CUDA | 5 | 718.1 ms | 0.065 |

Measured through the provider on realistic utterance lengths (medium, CUDA, beam=5):
**4 s utterance → 586 ms**, 11 s → 713 ms.

Transcripts correct throughout. Three findings that changed the design:

- **`medium` on GPU beats `small` on CPU by 4x.** The accuracy/latency trade
  largely disappears when the GPU works, so balanced mode uses the bigger model.
- **Beam search is nearly free** (~1 % CPU, ~16 % GPU). Greedy decoding for latency
  is not worth it here.
- **Latency is roughly constant, not proportional**, because Whisper pads to a 30 s
  window. Budgeting per-second-of-speech would have been wrong.

**The GPU was initially unusable** — `libcublas.so.12` missing, no system CUDA
toolkit. Resolved with pip-installed `nvidia-cublas-cu12` / `nvidia-cudnn-cu12`
preloaded via `RTLD_GLOBAL`. Worth noting: `get_cuda_device_count()` returned 1 and
`WhisperModel(device="cuda")` constructed fine in the broken state — only the first
*encode* failed. A health check that loads a model rather than running inference
would have reported a working GPU.

**Beam size is nearly free here** (0.262 → 0.265 RTF, ~1 %). The usual
latency-for-accuracy trade does not apply on this CPU, so beam=5 should be the
default rather than beam=1. That is a real, counter-intuitive finding and it came
only from measuring.

### The invalid benchmark, and why it is recorded here

The first version of this benchmark fed Whisper a synthetic formant stack and
reported RTF 5.8–22.7 — which would have made local STT completely non-viable and
forced a cloud-only architecture.

Those numbers were wrong. On non-speech input Whisper hallucinates and decodes to
its token limit, so the benchmark measured degenerate decoding rather than
transcription. Real speech was **40× faster**.

Recorded because the wrong conclusion was one step away: *never benchmark an ASR
model on synthetic audio.*

## 3. Architecture comparison

### A. Fully local
| | |
|---|---|
| **Latency** | STT ~1.05 s (measured, CPU). Translation and TTS unmeasured |
| **Cost** | Zero marginal |
| **Quality** | STT good. **Hinglish translation is the weak point** — local NMT is trained on monolingual pairs and code-mixing is out of distribution |
| **Hardware** | Feasible on CPU today; memory is tight with a browser open |
| **Privacy** | **Nothing leaves the machine.** The strongest property available |
| **Complexity** | Model downloads, versioning, memory scheduling |
| **Reliability** | No network dependency, no rate limits, no vendor outage |

### B. Hybrid — local STT + local VAD, cloud translation
| | |
|---|---|
| **Latency** | STT ~1.05 s local + translation network RTT (200–800 ms typical) |
| **Cost** | Per-utterance, text only — small |
| **Quality** | **Best available for Hinglish.** An LLM handles code-mixing and context far better than local NMT |
| **Hardware** | Same as local; no extra memory |
| **Privacy** | **Audio never leaves. Transcript text does.** A meaningful distinction |
| **Complexity** | Key management, failover, offline degradation |
| **Reliability** | Degrades to local on failure |

### C. Fully cloud
| | |
|---|---|
| **Latency** | Audio upload + STT + translation + TTS + audio download. Worst |
| **Cost** | Highest — audio in and audio out on every utterance |
| **Quality** | Best STT and best cloning, if the vendor supports Hindi |
| **Privacy** | **Raw meeting audio leaves the machine continuously.** Disqualifying for the stated use case |
| **Reliability** | Entirely dependent on network and vendor |

## 4. Measured: translation (NLLB-200 600M, CTranslate2, CPU)

Evaluated against `tests/fixtures/hinglish_cases.py` — 14 cases across four
registers, with `must_contain` / `must_not_contain` assertions on the parts of the
meaning that cannot be lost.

| Register | Pass | Notes |
|---|---|---|
| Devanagari (pure Hindi) | **4/4 (100%)** | |
| **Hinglish (code-mixed)** | **6/6 (100%)** | **Far better than predicted** |
| English (pass-through) | 2/2 (100%) | |
| Romanised Hindi | **0/2 (0%)** | Returned verbatim; **both suppressed by the gate** |
| **Overall** | **12/14 (86%)** | p50 latency **123 ms**, max 277 ms |

The motivating example from the brief works:

> `हाँ, मेरा कहना ये है कि हम इस project को अगले हफ्ते तक complete कर सकते हैं।`
> → *"Yes, I am saying that we can complete this project by next week."*

`अगले हफ्ते` → "next week" — the commitment survives intact.

**This overturns the prior assumption.** §8 of the TRD expected code-mixing to be
NLLB's weak point because it was trained on monolingual pairs. On these cases it
scored 100 %. Script-aware source tagging (tag as Hindi whenever Devanagari is
present, regardless of the recogniser's dominant-language guess) appears to be
enough for mixed input where the Hindi is in Devanagari.

**Romanised Hindi is the genuine failure**, exactly as predicted. NLLB returns
`"haan bilkul, main kal call kar lunga"` unchanged, at confidence **0.86**.

### Two defects this evaluation found

Both were invisible until real output was inspected, and both are the kind that
fail silently:

1. **The gate could not see untranslated romanised Hindi.** It only checked for
   Devanagari, so all-Latin Hindi passed through at high confidence and would have
   been *spoken to an English audience*. Fixed by adding a verbatim-echo check and
   a romanised-Hindi marker check.
2. **The first fix suppressed all correct English.** Requiring output ≠ input is
   wrong when source and target are both English — identical output is the correct
   result there. As first written it would have silenced every English utterance.
   Fixed by applying the verbatim check only across a language change.

A third issue was in the *test fixture itself*: the romanised case originally
asserted `must_contain=("call",)`, which matched the untranslated Hindi word
`call` and reported a **false PASS on completely untranslated output**. Overall
accuracy read 93 % instead of 86 %. A fixture that can pass on untranslated output
is worse than no fixture.

## 5. Recommendation
## 6. Measured: voice cloning (XTTS-v2)

**It works on this machine.** This was the largest open risk in the project and it
is now closed with measurements rather than estimates.

| Metric | Measured |
|---|---|
| VRAM for the model | **+1916 MiB** on load |
| Peak VRAM during synthesis | **2133 MiB** |
| Synthesis latency (p50) | **1188 ms** |
| Synthesis range | 670 ms – 2523 ms |
| RTF | 0.30 – 0.67 |
| Device | CUDA (RTX 3050 Mobile) |

Per-utterance detail:

| Sentence | Audio | Synth | RTF |
|---|---|---|---|
| "Yes, I am saying that we can complete this project by next week." | 3.76 s | 2523 ms | 0.67 |
| "The authentication part is still pending." | 2.83 s | 860 ms | 0.30 |
| "I need three days for this task." | 2.04 s | 670 ms | 0.33 |
| "Schedule a meeting with the team on Thursday afternoon." | 4.54 s | 1515 ms | 0.33 |

### Full latency budget, every stage measured

```
capture      22 ms   (measured)
hangover    500 ms   (fixed, tunable per mode)
STT         578 ms   (Whisper medium, CUDA)
translate   123 ms   (NLLB-200)
TTS        1188 ms   (XTTS-v2, CUDA)
--------------------
total      2411 ms   against a 4000 ms balanced budget  -> WITHIN BUDGET
```

**The product's core promise is deliverable on this hardware.** Nothing in the chain
is estimated.

### The constraint that remains

Peak VRAM of 2133 MiB for TTS plus ~900 MiB for STT is ~3.0 GiB. Free VRAM was
measured at **3770 MiB idle** but only **1610 MiB with a browser open**. So:

- Idle: STT + TTS both fit on the GPU; translation goes to CPU.
- Realistic meeting conditions: only STT fits, and TTS must run on CPU, where it
  will be far slower than 1188 ms.

`core/model_budget.py` makes that call automatically against live free VRAM. The
estimate it used for TTS (2300 MiB) proved close to the measured 2133 MiB and
conservative in the right direction.

### Caveats stated plainly

- The reference voice was a **public-domain speech clip, not the user's voice.**
  This measures cloning *mechanics* and latency, not how well the system will
  reproduce the user. NFR-7 (voice similarity) remains **UNVERIFIED**.
- First load took 625 s, almost all of it downloading ~1.9 GB. Warm-start load is
  measured separately.
- Synthesis latency scales with sentence length, so long utterances will exceed the
  p50 materially — the 2523 ms case is a 3.76 s sentence.

## 6. Provider matrix as implemented

| Stage | Provider | Local | Status |
|---|---|---|---|
| VAD | `SileroVad` | yes | Implemented (downloads 1.8 MB once) |
| VAD | `EnergyVad` | yes | Implemented, verified, zero-dependency fallback |
| STT | `FasterWhisperRecognizer` | yes | Implemented, **benchmarked on CPU and GPU** |
| Translation | `PassthroughTranslator` | yes | Implemented — English only, honest about it |
| Translation | `NllbTranslator` | yes | **Implemented, evaluated: 86% overall, 100% on Hinglish** |
| Translation | cloud LLM | no | **Not implemented** — interface ready |
| TTS | `FallbackSynthesizer` | yes | Implemented, always flagged as not-your-voice |
| TTS | `XttsSynthesizer` | yes | **Implemented and measured**: 1188 ms p50, 2133 MiB peak VRAM; consent enforced at synthesis time |

## 7. What would change these conclusions

- ~~Resolving the cuBLAS issue.~~ **Done.** GPU STT cut the 4 s-utterance cost from
  ~1050 ms to 586 ms while simultaneously allowing the larger `medium` model.
- ~~Benchmarking a cloning TTS at the real memory ceiling.~~ **Done.** XTTS-v2 runs
  at 1188 ms p50 using 2133 MiB peak. Total pipeline 2411 ms, inside budget.
- **Testing on the user's own recorded speech.** The 14 fixtures are
  representative, not exhaustive, and the register is personal. Romanised Hinglish
  in particular may be a much larger share of real usage than of this fixture set,
  which would change the local-vs-cloud answer.
