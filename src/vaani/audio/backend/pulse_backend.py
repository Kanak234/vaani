"""Capture and playback streams over libpulse-simple.

Audio format policy, decided once here so no other module has to think about it:
  * On the wire to/from the server: signed 16-bit little-endian, mono.
  * Inside the pipeline: float32 in [-1, 1].
The int16 <-> float32 conversion happens only at these boundaries.

16 kHz is used throughout the capture path because it is what Whisper wants; asking
the server for 16 kHz lets PipeWire do the resampling in its own graph, which is
better optimised than anything we would write and costs us no extra latency.
"""
from __future__ import annotations

import ctypes
import threading
import time

import numpy as np

from ...core.errors import ErrorCode, Severity, VaaniError
from .pulse_bindings import (
    PA_SAMPLE_S16LE,
    PA_STREAM_PLAYBACK,
    PA_STREAM_RECORD,
    PA_UNSPEC,
    PaBufferAttr,
    PaSampleSpec,
    PulseUnavailable,
    lib,
    strerror,
)

_INT16_SCALE = 32768.0


def _to_float32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / _INT16_SCALE


def _to_int16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * (_INT16_SCALE - 1)).astype("<i2").tobytes()


class _Stream:
    """Shared open/close/latency behaviour for both directions."""

    def __init__(self, *, direction: int, device: str | None, sample_rate: int,
                 channels: int, frame_bytes: int, stream_name: str,
                 app_name: str = "Vaani") -> None:
        self._lock = threading.Lock()
        self._handle: int | None = None
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_bytes = frame_bytes

        try:
            L = lib()
        except PulseUnavailable as exc:
            raise VaaniError(
                code=ErrorCode.AUDIO_BACKEND_UNAVAILABLE,
                message=str(exc),
                severity=Severity.FATAL,
                cause=exc,
            ) from exc

        spec = PaSampleSpec(PA_SAMPLE_S16LE, sample_rate, channels)
        if direction == PA_STREAM_RECORD:
            # fragsize is the capture quantum -> capture latency.
            attr = PaBufferAttr(PA_UNSPEC, PA_UNSPEC, PA_UNSPEC, PA_UNSPEC, frame_bytes)
        else:
            # tlength is the target playback buffer. Two frames balances
            # underrun resistance against added output latency.
            attr = PaBufferAttr(PA_UNSPEC, frame_bytes * 2, 0, frame_bytes, PA_UNSPEC)

        err = ctypes.c_int(0)
        handle = L.pa_simple_new(
            None,
            app_name.encode(),
            direction,
            device.encode() if device else None,
            stream_name.encode(),
            ctypes.byref(spec),
            None,
            ctypes.byref(attr),
            ctypes.byref(err),
        )
        if not handle:
            code = (ErrorCode.MIC_UNAVAILABLE if direction == PA_STREAM_RECORD
                    else ErrorCode.VIRTUAL_MIC_UNAVAILABLE)
            raise VaaniError(
                code=code,
                message=f"could not open {'capture' if direction == PA_STREAM_RECORD else 'playback'} "
                        f"stream on {device or 'default device'}: {strerror(err.value)}",
                severity=Severity.SESSION,
                detail={"device": device or "default", "pulse_error": err.value},
            )
        self._handle = handle

    @property
    def is_open(self) -> bool:
        return self._handle is not None

    def latency_ms(self) -> float:
        """Server-reported latency. Real measurement, not an estimate."""
        with self._lock:
            if self._handle is None:
                return 0.0
            err = ctypes.c_int(0)
            usec = lib().pa_simple_get_latency(self._handle, ctypes.byref(err))
            if err.value != 0:
                return 0.0
            return usec / 1000.0

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                lib().pa_simple_free(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class PulseCaptureStream(_Stream):
    """Blocking mono capture at a fixed frame size."""

    def __init__(self, *, device: str | None = None, sample_rate: int = 16000,
                 frame_ms: int = 20, stream_name: str = "capture") -> None:
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        super().__init__(
            direction=PA_STREAM_RECORD,
            device=device,
            sample_rate=sample_rate,
            channels=1,
            frame_bytes=self.frame_samples * 2,
            stream_name=stream_name,
        )
        self.frame_ms = frame_ms
        self._buf = (ctypes.c_char * self.frame_bytes)()

    def read_frame(self) -> np.ndarray:
        """Block until one frame is available. Returns float32 mono."""
        with self._lock:
            if self._handle is None:
                raise VaaniError(code=ErrorCode.DEVICE_DISCONNECTED,
                                 message="capture stream is closed",
                                 severity=Severity.SESSION)
            err = ctypes.c_int(0)
            rc = lib().pa_simple_read(self._handle, self._buf,
                                      self.frame_bytes, ctypes.byref(err))
            if rc < 0:
                raise VaaniError(
                    code=ErrorCode.DEVICE_DISCONNECTED,
                    message=f"capture read failed: {strerror(err.value)}",
                    severity=Severity.SESSION,
                    detail={"pulse_error": err.value},
                )
            return _to_float32(bytes(self._buf))


class PulsePlaybackStream(_Stream):
    """Blocking mono playback. Used for both the virtual mic and the monitor."""

    def __init__(self, *, device: str | None = None, sample_rate: int = 16000,
                 frame_ms: int = 20, stream_name: str = "playback",
                 require_device: bool = False) -> None:
        # pa_simple_new SUCCEEDS on a device name that does not exist -- it simply
        # connects to the default sink. For the virtual microphone that failure is
        # silent and severe: the app would look healthy while the user's translated
        # speech played out of the laptop speakers into the meeting room instead of
        # into the meeting. Verifying up front is the only way to catch it.
        if require_device and device is not None:
            from ...devices.manager import list_sinks
            if not any(d.key == device for d in list_sinks()):
                raise VaaniError(
                    code=ErrorCode.VIRTUAL_MIC_UNAVAILABLE,
                    message=f"playback device {device!r} does not exist; refusing to "
                            "open a stream that would silently play elsewhere",
                    severity=Severity.SESSION,
                    detail={"device": device},
                )
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        super().__init__(
            direction=PA_STREAM_PLAYBACK,
            device=device,
            sample_rate=sample_rate,
            channels=1,
            frame_bytes=self.frame_samples * 2,
            stream_name=stream_name,
        )
        self.frame_ms = frame_ms
        self.underruns = 0

    def write(self, samples: np.ndarray) -> None:
        """Block until `samples` have been handed to the server."""
        payload = _to_int16(np.asarray(samples, dtype=np.float32))
        with self._lock:
            if self._handle is None:
                raise VaaniError(code=ErrorCode.VIRTUAL_MIC_UNAVAILABLE,
                                 message="playback stream is closed",
                                 severity=Severity.SESSION)
            err = ctypes.c_int(0)
            rc = lib().pa_simple_write(self._handle, payload, len(payload),
                                       ctypes.byref(err))
            if rc < 0:
                raise VaaniError(
                    code=ErrorCode.VIRTUAL_MIC_UNAVAILABLE,
                    message=f"playback write failed: {strerror(err.value)}",
                    severity=Severity.SESSION,
                    detail={"pulse_error": err.value},
                )

    def write_silence(self, duration_ms: float) -> None:
        """Emit silence.

        This is the safe output for every failure path: when a stage fails we keep
        the virtual microphone alive and silent rather than letting it stall, so a
        meeting client never sees the device disappear mid-call.
        """
        n = int(self.sample_rate * duration_ms / 1000)
        self.write(np.zeros(n, dtype=np.float32))

    def drain(self) -> None:
        with self._lock:
            if self._handle is not None:
                err = ctypes.c_int(0)
                lib().pa_simple_drain(self._handle, ctypes.byref(err))

    def flush(self) -> None:
        """Discard buffered audio immediately -- used by emergency stop (AC-12.2)."""
        with self._lock:
            if self._handle is not None:
                err = ctypes.c_int(0)
                lib().pa_simple_flush(self._handle, ctypes.byref(err))
