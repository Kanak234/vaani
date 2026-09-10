"""Proves synthesised audio actually reaches a meeting application.

This is the test that matters most: everything else can pass while the virtual
microphone silently carries nothing. We write real synthesised speech into the
virtual source and simultaneously CAPTURE FROM IT the way Zoom would, then check
the recovered signal.

Run: PYTHONPATH=src .venv/bin/python tests/manual/verify_virtual_mic_audio.py
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

import numpy as np

from vaani.ai.tts.fallback import FallbackSynthesizer
from vaani.audio.backend.pulse_backend import PulsePlaybackStream
from vaani.devices.manager import VirtualMicrophone, list_sources

SR = 16000
FRAME = SR * 20 // 1000


def main() -> int:
    print("=" * 68)
    print("VIRTUAL MICROPHONE AUDIO PATH VERIFICATION")
    print("=" * 68)

    print("\n[1] Synthesising speech...")
    tts = FallbackSynthesizer(sample_rate=SR)
    tts.warmup()
    audio = tts.synthesize("I will complete this project by next week.").samples
    dur = len(audio) / SR
    print(f"    {len(audio)} samples, {dur:.2f}s, peak {np.max(np.abs(audio)):.3f}")

    print("\n[2] Creating virtual microphone...")
    mic = VirtualMicrophone.create()
    try:
        print(f"    source apps read: {mic.node_name}")
        print(f"    sink we write to: {mic.sink_name}")
        visible = [d for d in list_sources() if d.is_virtual]
        if not visible:
            print("    FAIL: not visible as a capture source")
            return 1
        print(f"    apps will list it as: {visible[0].display_name!r}")

        # Record from the virtual mic exactly as a meeting client would.
        print("\n[3] Recording FROM the virtual mic (as Zoom/Meet would)...")
        rec_seconds = dur + 1.2
        proc = subprocess.Popen(
            ["parec", "--device", mic.node_name, "--format=s16le",
             "--rate", str(SR), "--channels=1", "--latency-msec=20"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        time.sleep(0.4)  # let the recorder attach before we play

        print("[4] Writing synthesised audio into the virtual mic...")
        # Write to the SINK; apps read from mic.node_name.
        sink = PulsePlaybackStream(device=mic.sink_name, sample_rate=SR,
                                   frame_ms=20, stream_name="verify-out",
                                   require_device=True)
        written = 0
        try:
            # Lead-in silence, the audio, then trailing silence -- mirrors how the
            # output thread keeps the device continuously fed.
            for _ in range(10):
                sink.write(np.zeros(FRAME, dtype=np.float32))
            for i in range(0, len(audio), FRAME):
                block = audio[i:i + FRAME]
                if block.size < FRAME:
                    block = np.pad(block, (0, FRAME - block.size))
                sink.write(block)
                written += FRAME
            for _ in range(10):
                sink.write(np.zeros(FRAME, dtype=np.float32))
            lat = sink.latency_ms()
        finally:
            sink.close()
        print(f"    wrote {written} samples; playback latency {lat:.2f} ms")

        time.sleep(0.5)
        proc.terminate()
        raw = proc.stdout.read()
        proc.wait(timeout=5)

        print("\n[5] Analysing what a meeting app would have received...")
        captured = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if captured.size == 0:
            print("    FAIL: captured nothing from the virtual mic")
            return 1

        peak = float(np.max(np.abs(captured)))
        rms = float(np.sqrt(np.mean(captured ** 2)))
        active = int(np.sum(np.abs(captured) > 0.01))
        print(f"    captured {captured.size} samples ({captured.size / SR:.2f}s)")
        print(f"    peak {peak:.4f}   rms {rms:.5f}   non-silent samples {active}")

        if peak < 0.01:
            print("\n    FAIL: virtual mic carried only silence")
            return 1
        if active < SR * 0.2:
            print("\n    FAIL: too little signal recovered")
            return 1

        # Cross-correlate to prove this is OUR audio and not ambient noise.
        # Full FFT correlation rather than a stepped offset search: the recorder
        # starts at an unknown offset, and PipeWire resamples 16k->48k->16k, which
        # shifts phase enough that a coarse search misses the true alignment.
        sent = audio - audio.mean()
        got = captured - captured.mean()
        n = 1 << int(np.ceil(np.log2(sent.size + got.size)))
        xcorr = np.fft.irfft(np.fft.rfft(got, n) * np.conj(np.fft.rfft(sent, n)), n)
        norm = np.sqrt(np.sum(sent ** 2) * np.sum(got ** 2)) + 1e-12
        best = float(np.max(np.abs(xcorr)) / norm)
        lag = int(np.argmax(np.abs(xcorr)))
        print(f"    peak cross-correlation: {best:.3f} at lag {lag / SR * 1000:.0f} ms")

        # Envelope correlation is robust to resampling phase and confirms the
        # amplitude contour survived the round trip.
        def envelope(x, win=400):
            k = x.size // win
            return np.abs(x[:k * win]).reshape(k, win).mean(axis=1)
        es, eg = envelope(sent), envelope(got[lag:lag + sent.size])
        m = min(es.size, eg.size)
        env_corr = (float(abs(np.corrcoef(es[:m], eg[:m])[0, 1]))
                    if m > 4 else 0.0)
        print(f"    envelope correlation:   {env_corr:.3f}")

        print("\n" + "=" * 68)
        if best > 0.25 or env_corr > 0.7:
            print("PASS - synthesised audio verifiably reaches the virtual microphone.")
            print("       A meeting application selecting")
            print(f"       '{visible[0].display_name}' receives this audio.")
            print("=" * 68)
            return 0
        print(f"PARTIAL - signal present (peak {peak:.3f}) but correlation low.")
        print("=" * 68)
        return 1
    finally:
        mic.destroy()
        print("\n[cleanup] virtual mic unloaded; "
              f"remaining: {len([d for d in list_sources() if d.is_virtual])}")


if __name__ == "__main__":
    sys.exit(main())
