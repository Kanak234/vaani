"""Audio-path soak test (NFR-4: zero underruns in a 60-minute session).

Drives the real capture and virtual-microphone path continuously, without the AI
stages, so it isolates the audio layer. AI stage failures are utterance-level and
recoverable; an audio underrun is a gap the meeting hears.

Reports underruns, frame-timing drift and RSS growth, since a slow memory leak
only shows up over an hour.

Usage: soak_test.py [--minutes 60] [--input DEVICE]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from vaani.audio.backend.pulse_backend import (  # noqa: E402
    PulseCaptureStream,
    PulsePlaybackStream,
)
from vaani.audio.segmenter import SegmenterConfig, UtteranceSegmenter  # noqa: E402
from vaani.ai.vad.energy import EnergyVad  # noqa: E402
from vaani.devices.manager import VirtualMicrophone  # noqa: E402

SR, FRAME_MS = 16000, 20
FRAME_N = SR * FRAME_MS // 1000


def rss_mb() -> float:
    try:
        with open(f"/proc/{os.getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--input", default=None)
    args = ap.parse_args()

    print("=" * 72)
    print(f"AUDIO SOAK TEST — {args.minutes:.0f} minutes (NFR-4: 0 underruns)")
    print("=" * 72)

    mic = VirtualMicrophone.create()
    capture = PulseCaptureStream(device=args.input, sample_rate=SR,
                                 frame_ms=FRAME_MS, stream_name="soak-in")
    sink = PulsePlaybackStream(device=mic.sink_name, sample_rate=SR,
                               frame_ms=FRAME_MS, stream_name="soak-out",
                               require_device=True)
    vad = EnergyVad()
    segmenter = UtteranceSegmenter(config=SegmenterConfig(), sample_rate=SR,
                                   frame_ms=FRAME_MS)

    deadline = time.monotonic() + args.minutes * 60
    frames = underruns = segments = 0
    gaps: list[float] = []
    rss0 = rss_mb()
    last = time.perf_counter()
    next_report = time.monotonic() + 60

    try:
        while time.monotonic() < deadline:
            frame = capture.read_frame()
            now = time.perf_counter()
            gaps.append((now - last) * 1000)
            last = now
            frames += 1

            if segmenter.push(frame, vad.is_speech(frame)) is not None:
                segments += 1
            try:
                sink.write(np.zeros(FRAME_N, dtype=np.float32))
            except Exception:
                underruns += 1

            if time.monotonic() >= next_report:
                next_report += 60
                recent = gaps[-3000:]
                print(f"  t+{(args.minutes*60-(deadline-time.monotonic()))/60:5.1f}m  "
                      f"frames {frames:7d}  underruns {underruns}  "
                      f"utterances {segments:4d}  "
                      f"frame gap p50 {statistics.median(recent):5.1f}ms  "
                      f"RSS {rss_mb():6.1f} MB")
    except KeyboardInterrupt:
        print("\n  interrupted by user")
    finally:
        capture.close()
        sink.close()
        mic.destroy()

    ordered = sorted(gaps)
    p50 = statistics.median(ordered)
    p99 = ordered[int(len(ordered) * 0.99)] if ordered else 0
    drift = abs(p50 - FRAME_MS)
    growth = rss_mb() - rss0

    print("-" * 72)
    print(f"  frames captured   {frames}")
    print(f"  underruns         {underruns}   (NFR-4 requires 0)")
    print(f"  utterances        {segments}")
    print(f"  frame gap         p50 {p50:.2f} ms  p99 {p99:.2f} ms  "
          f"(nominal {FRAME_MS} ms)")
    print(f"  timing drift      {drift:.2f} ms")
    print(f"  RSS               {rss0:.1f} -> {rss_mb():.1f} MB  ({growth:+.1f} MB)")
    ok = underruns == 0 and drift < 5.0 and growth < 100
    print("=" * 72)
    print("PASS" if ok else "FAIL")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
