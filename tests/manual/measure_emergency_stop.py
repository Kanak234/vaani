"""Measure emergency-stop latency against AC-12.2 (halt within 200 ms).

Needs real audio hardware. This measures the whole path a user experiences: from
calling emergency_stop() to the virtual microphone actually going silent, with a
long synthesised utterance already queued and playing.

Run: PYTHONPATH=src .venv/bin/python tests/manual/measure_emergency_stop.py
"""
from __future__ import annotations

import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from vaani.audio.backend.pulse_backend import PulsePlaybackStream  # noqa: E402
from vaani.core.state_machine import SessionState, SessionStateMachine  # noqa: E402
from vaani.devices.manager import VirtualMicrophone  # noqa: E402

SR, FRAME = 16000, 320
BUDGET_MS = 200.0


def one_trial(mic, sm) -> tuple[float, float]:
    """Returns (ms_of_tone_heard_after_stop, ms_of_tone_written_after_stop).

    The metric that matters is what the MEETING hears, so it is taken from the
    captured audio timeline, not from wall-clock around the call. Measuring the
    latter would mostly time our own silence-writing loop, which is not what a
    listener experiences.
    """
    proc = subprocess.Popen(
        ["parec", "--device", mic.node_name, "--format=s16le", "--rate", str(SR),
         "--channels=1", "--latency-msec=20"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    time.sleep(0.4)

    sink = PulsePlaybackStream(device=mic.sink_name, sample_rate=SR, frame_ms=20,
                               stream_name="estop-test", require_device=True)
    sm.force(SessionState.OUTPUTTING)
    t = np.arange(SR * 3, dtype=np.float32) / SR
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    # Play ~1 s of tone, then stop mid-utterance.
    for i in range(0, SR, FRAME):
        sink.write(tone[i:i + FRAME])

    sm.emergency_stop()
    sink.flush()          # drop everything already buffered downstream

    written_after = 0
    for i in range(SR, SR + SR // 2, FRAME):
        if sm.can_emit_audio:            # must be False -- the invariant
            sink.write(tone[i:i + FRAME])
            written_after += FRAME
        else:
            sink.write(np.zeros(FRAME, dtype=np.float32))
    sink.close()

    time.sleep(0.4)
    proc.terminate()
    raw = proc.stdout.read()
    proc.wait(timeout=5)

    got = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if got.size == 0:
        return float("nan"), written_after / SR * 1000

    # How much tone did the listener actually hear? We sent 1000 ms before the
    # stop; anything beyond that is audio the emergency stop failed to cut.
    loud = np.abs(got) > 0.02
    heard_ms = float(loud.sum()) / SR * 1000
    overrun_ms = max(0.0, heard_ms - 1000.0)
    return overrun_ms, written_after / SR * 1000


def main() -> int:
    print("=" * 70)
    print("EMERGENCY STOP LATENCY (AC-12.2: halt within 200 ms)")
    print("=" * 70)
    mic = VirtualMicrophone.create()
    sm = SessionStateMachine()
    try:
        stops, residuals = [], []
        for i in range(5):
            overrun_ms, written_ms = one_trial(mic, sm)
            stops.append(overrun_ms)
            residuals.append(written_ms)
            print(f"  trial {i+1}: tone heard past the stop {overrun_ms:6.1f} ms   "
                  f"tone written after stop: {written_ms:.1f} ms")
    finally:
        mic.destroy()

    p50 = statistics.median(stops)
    worst = max(stops)
    print("-" * 70)
    print(f"  audio heard past the stop: p50 {p50:.1f} ms   worst {worst:.1f} ms"
          f"   budget {BUDGET_MS:.0f} ms")
    print(f"  audio WRITTEN after the stop (must be 0): {max(residuals):.1f} ms")
    ok = worst <= BUDGET_MS and max(residuals) == 0
    print("=" * 70)
    print("PASS - emergency stop halts within budget and leaks no audio." if ok
          else "FAIL - see figures above.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
