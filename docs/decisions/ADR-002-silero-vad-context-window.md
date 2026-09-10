# ADR-002 — Silero VAD requires a 576-sample input, not 512

**Status:** Accepted · **Date:** 2026-09-09 · **Verified on:** silero_vad.onnx (2,327,524 bytes), ONNX Runtime 1.29.0

## Context

Silero VAD v5 at 16 kHz is universally documented as operating on **512-sample**
windows (32 ms). The implementation was written to that specification: accumulate
caller frames, run inference on each 512 samples, carry the LSTM state forward.

## The problem, measured

The model loaded, ran at 0.081 ms per frame, raised no error — and reported
essentially zero speech probability on loud, clean speech:

```
Silero   silence-region mean prob 0.001 | speech-region mean prob 0.001
         frames>0.5 in speech region: 0%
Energy   silence-region mean prob 0.000 | speech-region mean prob 0.995
         frames>0.5 in speech region: 99%
```

The energy VAD, on identical audio, detected 99 % of speech frames. So the audio
was fine and the fault was in the Silero path.

The ONNX signature is no help — `input` is declared `[None, None]`, so **512 is
accepted without complaint**. Sweeping the window size found the truth:

| Window | Mean prob | Frames > 0.5 |
|---|---|---|
| 512 | 0.001 | **0 %** |
| **576** | **0.589** | **57 %** |
| 1024 | ONNX error | — |
| 1536 | ONNX error | — |

## Decision

Feed the model **576 samples**: 64 samples of context carried from the previous
window, followed by 512 new samples.

```python
padded = np.concatenate([self._context, window])   # 64 + 512
self._context = window[-64:].copy()                # for the next call
```

The context buffer is zero-filled on `reset()` so utterance boundaries do not leak
audio across segments.

## Verification

After the fix, on real speech with leading and trailing silence:

```
Silero   silence 0.005 | speech 0.690 | speech frames>0.5: 67%
         white noise: mean 0.014, frames>0.5: 0%
Energy   silence 0.000 | speech 0.995 | speech frames>0.5: 99%
         white noise: mean 0.250, frames>0.5: 0%
```

This also **confirms the reason Silero was chosen over the energy VAD** (TRD §6),
which until now was an assertion rather than a measurement: on white noise Silero
reports 0.014 against the energy VAD's 0.250 — roughly 18× better rejection. That
gap is the whole point, and it is what keeps keyboard clatter and fan noise from
opening an utterance.

Silero's lower speech-frame rate (67 % vs 99 %) is expected and correct: it marks
inter-word and inter-sentence pauses as non-speech, where the energy VAD's adaptive
floor stays hot. The segmenter's hangover (`SegmenterConfig.hangover_ms`) is what
bridges those gaps, which is precisely why segmentation policy lives in the
segmenter and not in the VAD.

## Consequences

- Per-frame cost rises slightly (0.081 → 0.166 ms mean). Irrelevant against a 20 ms
  frame budget.
- `reset()` must clear the context buffer as well as the LSTM state.
- **The energy VAD stays in the codebase permanently.** It is what proved the audio
  was fine and localised this bug to the Silero path in one step. A fallback that
  can be cross-checked against the primary is worth its maintenance cost.
- Any future upgrade of `silero_vad.onnx` must re-verify the context size, since
  the ONNX signature will not enforce it and the failure is silent.
