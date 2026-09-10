# ADR-001 — Virtual microphone uses two PipeWire modules, not one

**Status:** Accepted · **Date:** 2026-09-09 · **Verified on:** PipeWire 1.6.2, Ubuntu 26.04

## Context

Vaani must present a microphone device that Zoom, Google Meet, Teams and browsers
can select as an input, and write synthesised speech into it.

The obvious modern approach is a single module:

```bash
pactl load-module module-null-sink media.class=Audio/Source/Virtual \
      sink_name=VaaniVirtualMic channel_map=mono
```

This is widely recommended, and on first inspection it works: the node appears in
`pactl list short sources`, and every application lists it as a microphone.

## The problem, measured

That single-module form creates a **source node only**. No sink is created:

```
=== SINKS ===   (no Vaani entry)
=== SOURCES === 2401  VaaniVirtualMic  PipeWire  float32le 1ch 48000Hz
```

There is therefore nowhere to write audio *into*. A PulseAudio playback stream
targeting `VaaniVirtualMic` does **not** fail — `pa_simple_new` returns a valid
handle and connects to the **default sink** instead.

Measured result of the naive implementation:

```
[4] Writing synthesised audio into the virtual mic...
    wrote 47680 samples; playback latency 58.34 ms
[5] Analysing what a meeting app would have received...
    peak 0.0000   rms 0.00000   non-silent samples 0
    FAIL: virtual mic carried only silence
```

This failure mode is the worst possible shape for this product:

- Every status check passes. The device exists, the stream opens, writes succeed.
- The meeting receives **silence** — the user appears to have a broken mic.
- The translated speech, in the user's own cloned voice, plays out of the **laptop
  speakers** into the physical room instead.

Nothing in the application would have reported an error.

## Decision

Use two modules and treat the write target as distinct from the read target:

```
module-null-sink     sink_name=VaaniSink          <- Vaani writes here
         |
         v  (VaaniSink.monitor)
module-remap-source  source_name=VaaniVirtualMic  <- applications read here
```

Additionally, `PulsePlaybackStream` takes `require_device=True` and verifies the
named sink exists **before** opening, so a misconfigured device can never again
silently redirect audio to the speakers.

## Verification

Round trip, writing to the sink and capturing from the source exactly as a meeting
client would:

```
captured 67520 samples (4.22s)
peak 0.4580   rms 0.07582   non-silent samples 33394
peak cross-correlation: 0.930 at lag 627 ms
envelope correlation:   0.981
PASS - synthesised audio verifiably reaches the virtual microphone.
```

A raw tone test through the same path recovered exactly the amplitude sent
(0.500 → 0.500).

## Alternatives considered

| Option | Rejected because |
|---|---|
| Single `Audio/Source/Virtual` null-sink | Carries no audio; fails silently (above) |
| Expose `VaaniSink.monitor` directly | Many apps filter monitor devices out of mic lists; those that do not label it "Monitor of…", which the user must decode mid-meeting |
| `module-pipe-source` | Needs a FIFO and manual pacing; we would reimplement the timing the audio server already does |
| Native `libpipewire` filter node | Lowest latency and most control, but a much larger C-level surface via ctypes, and it forfeits PulseAudio compatibility for apps that use that API |
| Kernel `snd-aloop` | Requires root and a module load at boot; violates the no-sudo constraint (BRD C4) |

## Consequences

- Two modules to load, and **unload order matters**: the source must be unloaded
  before the sink it remaps, or it is left pointing at a missing master.
- `VaaniSink` is visible to the user as an output device. Acceptable, and it makes
  the routing inspectable when something goes wrong.
- Crash recovery must detect **both** modules; a half-loaded pair is treated as
  absent (`exists()` requires both).
- `tests/manual/verify_virtual_mic_audio.py` must stay in the suite. This bug was
  invisible to every test that did not actually capture from the virtual device,
  and it would return the moment someone "simplifies" back to one module.
