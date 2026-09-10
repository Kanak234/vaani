"""End-to-end latency measurement through the real pipeline (NFR-1, AC-13.4).

Feeds real speech through the actual VAD -> segmenter -> STT -> gate ->
translation -> TTS chain and reports the per-stage table. Uses the real
components, not stubs -- the point is to produce a number a user would experience.

Latency is measured from `speech_end_time` (the moment VAD decided the user
stopped talking) to audio being ready, because that is when the user starts
waiting.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, "src")

from vaani.ai.stt.whisper import FasterWhisperRecognizer          # noqa: E402
from vaani.ai.translate.passthrough import PassthroughTranslator  # noqa: E402
from vaani.ai.tts.fallback import FallbackSynthesizer             # noqa: E402
from vaani.ai.vad.silero import SileroVad                         # noqa: E402
from vaani.audio.segmenter import SegmenterConfig, UtteranceSegmenter  # noqa: E402
from vaani.context.engine import ContextEngine                    # noqa: E402
from vaani.core.gate import ConfidenceGate, GateConfig            # noqa: E402
from vaani.core.pipeline import TranslationPipeline               # noqa: E402
from vaani.core.state_machine import SessionState, SessionStateMachine  # noqa: E402
from vaani.core.types import PerformanceMode, Utterance           # noqa: E402

SR = 16000
FRAME_MS = 20
FRAME_N = SR * FRAME_MS // 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--mode", default="balanced",
                    choices=[m.value for m in PerformanceMode])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--translate", default="nllb", choices=["nllb", "passthrough"])
    args = ap.parse_args()

    mode = PerformanceMode(args.mode)
    audio, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"expected {SR} Hz, got {sr}")

    print("=" * 74)
    print(f"END-TO-END LATENCY  --  mode={mode.value}  input={len(audio)/SR:.1f}s")
    print("=" * 74)

    recognizer = FasterWhisperRecognizer.for_mode(mode)
    t0 = time.perf_counter()
    recognizer.warmup()
    print(f"STT: {recognizer.model_size} on {recognizer.device_used}/"
          f"{recognizer.compute_type_used}, beam={recognizer.beam_size} "
          f"(warmup {time.perf_counter()-t0:.1f}s)")

    vad = SileroVad()
    seg_cfg = SegmenterConfig.for_mode(mode)
    print(f"Segmenter: hangover={seg_cfg.hangover_ms}ms pre_roll={seg_cfg.pre_roll_ms}ms")
    print()

    if args.translate == "nllb":
        from vaani.ai.translate.nllb import NllbTranslator
        from vaani.core.model_budget import plan_placement
        plan = plan_placement()
        print(f"Model placement: {plan.summary()}")
        for note in plan.notes:
            print(f"  {note}")
        translator = NllbTranslator(device=plan.device_for("translation"))
        t0 = time.perf_counter()
        translator.warmup()
        print(f"Translation: NLLB-200 600M on {translator.device_used}/"
              f"{translator.compute_type_used} (warmup {time.perf_counter()-t0:.1f}s)")
    else:
        translator = PassthroughTranslator()
        print("Translation: pass-through")
    print()

    sm = SessionStateMachine()
    sm.transition(SessionState.INITIALIZING)
    sm.transition(SessionState.LISTENING)
    pipeline = TranslationPipeline(
        recognizer=recognizer, translator=translator,
        synthesizer=FallbackSynthesizer(), context=ContextEngine("bench"),
        gate=ConfidenceGate(GateConfig()), state_machine=sm)

    # Segment the real audio the way a live session would.
    segmenter = UtteranceSegmenter(config=seg_cfg, sample_rate=SR, frame_ms=FRAME_MS)
    padded = np.concatenate([np.zeros(SR // 2, dtype=np.float32), audio,
                             np.zeros(SR, dtype=np.float32)])
    segments = []
    for i in range(0, len(padded) - FRAME_N, FRAME_N):
        frame = padded[i:i + FRAME_N]
        s = segmenter.push(frame, vad.is_speech(frame))
        if s:
            segments.append(s)
    if not segments:
        s = segmenter.flush()
        if s:
            segments.append(s)
    print(f"VAD/segmenter produced {len(segments)} utterance(s): "
          + ", ".join(f"{s.audio.size/SR:.1f}s" for s in segments))
    print()

    all_runs = []
    for rep in range(args.repeats):
        for n, s in enumerate(segments, 1):
            u = Utterance(session_id="bench", seq=rep * 100 + n, audio=s.audio,
                          sample_rate=SR, speech_end_time=time.monotonic(),
                          speech_ms=s.speech_ms)
            r = pipeline.process(u)
            all_runs.append(r)
            if sm.can_transition(SessionState.LISTENING):
                sm.transition(SessionState.LISTENING)
            if rep == 0:
                status = ("SUPPRESSED: " + r.suppression_reason.value
                          if r.suppressed else "ok")
                print(f"  utterance {n} ({s.audio.size/SR:.1f}s): {status}")
                if r.transcript:
                    print(f"    \"{r.transcript.text[:66]}\"  conf={r.transcript.confidence:.2f}")

    good = [r for r in all_runs if not r.suppressed]
    if not good:
        print("\nAll utterances suppressed - no latency to report.")
        return 1

    print("\n" + "-" * 74)
    print(f"{'STAGE':<22}{'p50 (ms)':>12}{'p95 (ms)':>12}{'max (ms)':>12}")
    print("-" * 74)
    stages = ["stt", "gate_stt", "context", "translate", "gate_translation", "tts"]
    total_p50 = 0.0
    for stage in stages:
        vals = sorted(t.duration_ms for r in good for t in r.timings
                      if t.stage == stage)
        if not vals:
            continue
        p50 = statistics.median(vals)
        p95 = vals[min(len(vals) - 1, int(len(vals) * 0.95))]
        total_p50 += p50
        print(f"{stage:<22}{p50:>12.1f}{p95:>12.1f}{max(vals):>12.1f}")
    print("-" * 74)

    e2e = sorted(r.total_latency_ms for r in good)
    p50 = statistics.median(e2e)
    p95 = e2e[min(len(e2e) - 1, int(len(e2e) * 0.95))]
    print(f"{'PIPELINE TOTAL':<22}{total_p50:>12.1f}")
    print(f"{'MEASURED END-TO-END':<22}{p50:>12.1f}{p95:>12.1f}{max(e2e):>12.1f}")
    print()
    print(f"  + capture latency          22.3 ms  (measured separately)")
    print(f"  + VAD hangover        {seg_cfg.hangover_ms:>9} ms  (fixed cost before "
          f"the utterance even closes)")
    user_p50 = p50 + 22.3 + seg_cfg.hangover_ms
    print("-" * 74)
    print(f"  USER-PERCEIVED p50    {user_p50:>9.0f} ms")
    budget = {"low_latency": 2500, "balanced": 4000, "quality": 6000}[mode.value]
    print(f"  BUDGET (NFR-1/2)      {budget:>9} ms   "
          f"{'PASS' if user_p50 <= budget else 'OVER BUDGET'}")
    print("=" * 74)
    print(f"NOTE: translation={args.translate}, TTS=fallback synthesiser.")
    print("      A cloning TTS (XTTS-v2) will add materially to these numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
