"""Measure XTTS-v2 voice cloning on this machine: does it fit, and how slow is it?

Uses a public-domain speech clip as the reference voice. This tests the CLONING
MECHANICS only -- it is not the user's voice and says nothing about how well the
product will reproduce theirs.

Reports peak VRAM and per-utterance latency, which are the two numbers that decide
whether the product's core promise is deliverable on this hardware.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, "src")

from vaani.core.model_budget import free_vram_mb, observed_vram_mb  # noqa: E402

SENTENCES = [
    "Yes, I am saying that we can complete this project by next week.",
    "The authentication part is still pending.",
    "I need three days for this task.",
    "Schedule a meeting with the team on Thursday afternoon.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="reference voice wav")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    base_vram = observed_vram_mb()
    print("=" * 74)
    print("XTTS-v2 VOICE CLONING BENCHMARK")
    print("=" * 74)
    print(f"reference : {args.reference}")
    print(f"VRAM before: used {base_vram} MiB, free {free_vram_mb()} MiB")

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device    : {device}")

    from vaani.ai.tts.xtts import _allow_coqui_globals
    _allow_coqui_globals()
    import os
    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    t0 = time.perf_counter()
    try:
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    except Exception as exc:
        print(f"\nFAILED to load: {type(exc).__name__}: {str(exc)[:300]}")
        return 1
    load_s = time.perf_counter() - t0
    after_load = observed_vram_mb()
    print(f"model load : {load_s:.1f}s")
    if base_vram is not None and after_load is not None:
        print(f"VRAM after load: used {after_load} MiB "
              f"(+{after_load - base_vram} MiB for the model)")

    print("\n" + "-" * 74)
    print(f"{'sentence':<48}{'audio':>9}{'synth':>10}{'RTF':>7}")
    print("-" * 74)
    times, peak_vram = [], after_load or 0
    for text in SENTENCES:
        t0 = time.perf_counter()
        try:
            wav = tts.tts(text=text, speaker_wav=args.reference, language="en")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:120]}")
            return 1
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)
        audio = np.asarray(wav, dtype=np.float32)
        dur = audio.size / 24000
        peak_vram = max(peak_vram, observed_vram_mb() or 0)
        print(f"  {text[:46]:<46}{dur:>8.2f}s{ms:>9.0f}ms{ms/1000/dur:>7.2f}")

    sf.write("/tmp/claude-1000/xtts_sample.wav", np.asarray(wav, dtype=np.float32), 24000)

    print("-" * 74)
    p50 = statistics.median(times)
    print(f"synthesis p50 {p50:.0f}ms   min {min(times):.0f}ms   max {max(times):.0f}ms")
    print(f"peak VRAM     {peak_vram} MiB")
    print()
    print("Latency budget check (balanced mode, 4000 ms):")
    print(f"  capture 22 + hangover 500 + STT 578 + translate 123 + TTS {p50:.0f}"
          f" = {22 + 500 + 578 + 123 + p50:.0f} ms")
    total = 22 + 500 + 578 + 123 + p50
    print(f"  -> {'WITHIN BUDGET' if total <= 4000 else 'OVER BUDGET'}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
