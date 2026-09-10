"""Benchmark faster-whisper on THIS machine using REAL speech.

An earlier version of this script fed Whisper a synthetic formant stack. Those
numbers (RTF 5.8-22.7) were meaningless: on non-speech input Whisper hallucinates
and decodes to its token limit, so the benchmark measured worst-case degenerate
decoding rather than transcription. Real speech is the only valid input here.

Usage: bench_stt.py <16k mono wav> [--models small,medium] [--devices cuda,cpu]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np
import soundfile as sf


def load(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"expected 16 kHz, got {sr}")
    return audio.astype(np.float32), sr


def bench(model_size, device, compute_type, audio, sr, runs=3):
    from faster_whisper import WhisperModel
    t0 = time.perf_counter()
    try:
        m = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:
        print(f"\n=== {model_size} / {device} / {compute_type} ===")
        print(f"  UNAVAILABLE: {type(exc).__name__}: {str(exc)[:120]}")
        return None
    load_s = time.perf_counter() - t0
    print(f"\n=== {model_size} / {device} / {compute_type} ===")
    print(f"  model load: {load_s:.2f}s")

    dur = len(audio) / sr
    # Warm up so the first timed run does not pay graph-build cost.
    list(m.transcribe(audio, beam_size=1)[0])

    text = ""
    for beam in (1, 5):
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            segs, info = m.transcribe(audio, beam_size=beam, vad_filter=False,
                                      condition_on_previous_text=False)
            out = list(segs)
            times.append(time.perf_counter() - t)
            text = " ".join(s.text.strip() for s in out)
        med = statistics.median(times)
        print(f"  beam={beam}  {dur:.1f}s audio -> median {med * 1000:7.1f}ms  "
              f"best {min(times) * 1000:7.1f}ms  RTF {med / dur:.3f}")
    print(f"  detected language: {info.language} (p={info.language_probability:.2f})")
    print(f"  transcript: {text[:95]}...")
    del m
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--models", default="small,medium")
    ap.add_argument("--devices", default="cuda,cpu")
    args = ap.parse_args()

    audio, sr = load(args.audio)
    print(f"input: {args.audio}  {len(audio) / sr:.2f}s @ {sr}Hz")

    import ctranslate2
    print(f"ctranslate2 {ctranslate2.__version__}  "
          f"cuda devices: {ctranslate2.get_cuda_device_count()}")

    for size in args.models.split(","):
        for dev in args.devices.split(","):
            ct = "int8_float16" if dev == "cuda" else "int8"
            bench(size, dev, ct, audio, sr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
