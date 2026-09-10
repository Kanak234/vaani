"""Minimal ctypes binding to libpulse-simple.

WHY ctypes rather than the `sounddevice`/PortAudio stack:

1. `libportaudio2` is not installed on the target machine and installing it needs
   root. `libpulse-simple.so.0` is already present on any PipeWire/PulseAudio
   desktop. Requiring no sudo is a hard constraint (BRD C4).
2. PortAudio would sit on top of this same server anyway, adding a layer whose
   buffering we do not control.
3. Latency is a first-class requirement, and `fragsize`/`tlength` in `pa_buffer_attr`
   are exactly the knobs that set it. Binding directly means we set them ourselves.

Only the `simple` API is bound. It is blocking, which is what we want: each stream
lives on its own thread and blocks on read/write, giving natural pacing without a
callback system.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from typing import Final

# --- sample formats (pulse/sample.h) ---------------------------------------
PA_SAMPLE_S16LE: Final = 3
PA_SAMPLE_FLOAT32LE: Final = 5

# --- stream directions (pulse/def.h) ---------------------------------------
PA_STREAM_PLAYBACK: Final = 1
PA_STREAM_RECORD: Final = 2

#: Tells the server "pick a sensible value" for a buffer_attr field.
PA_UNSPEC: Final = 0xFFFFFFFF


class PaSampleSpec(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_int),
        ("rate", ctypes.c_uint32),
        ("channels", ctypes.c_uint8),
    ]


class PaBufferAttr(ctypes.Structure):
    """Buffer tuning. `fragsize` governs capture latency, `tlength` playback."""

    _fields_ = [
        ("maxlength", ctypes.c_uint32),
        ("tlength", ctypes.c_uint32),
        ("prebuf", ctypes.c_uint32),
        ("minreq", ctypes.c_uint32),
        ("fragsize", ctypes.c_uint32),
    ]


class PulseUnavailable(RuntimeError):
    """libpulse-simple could not be loaded."""


def _load() -> ctypes.CDLL:
    for name in ("libpulse-simple.so.0", "libpulse-simple.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    found = ctypes.util.find_library("pulse-simple")
    if found:
        try:
            return ctypes.CDLL(found)
        except OSError as exc:  # pragma: no cover
            raise PulseUnavailable(str(exc)) from exc
    raise PulseUnavailable(
        "libpulse-simple.so.0 not found. Vaani needs a PulseAudio or PipeWire "
        "session (install libpulse0)."
    )


_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    """Load and configure the library once, lazily."""
    global _lib
    if _lib is not None:
        return _lib

    L = _load()
    L.pa_simple_new.restype = ctypes.c_void_p
    L.pa_simple_new.argtypes = [
        ctypes.c_char_p,                    # server
        ctypes.c_char_p,                    # name
        ctypes.c_int,                       # direction
        ctypes.c_char_p,                    # device
        ctypes.c_char_p,                    # stream description
        ctypes.POINTER(PaSampleSpec),
        ctypes.c_void_p,                    # channel map (NULL = default)
        ctypes.POINTER(PaBufferAttr),
        ctypes.POINTER(ctypes.c_int),       # error out
    ]
    L.pa_simple_read.restype = ctypes.c_int
    L.pa_simple_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
    L.pa_simple_write.restype = ctypes.c_int
    L.pa_simple_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
    L.pa_simple_drain.restype = ctypes.c_int
    L.pa_simple_drain.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    L.pa_simple_flush.restype = ctypes.c_int
    L.pa_simple_flush.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    L.pa_simple_get_latency.restype = ctypes.c_uint64
    L.pa_simple_get_latency.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    L.pa_simple_free.restype = None
    L.pa_simple_free.argtypes = [ctypes.c_void_p]
    L.pa_strerror.restype = ctypes.c_char_p
    L.pa_strerror.argtypes = [ctypes.c_int]

    _lib = L
    return _lib


def strerror(code: int) -> str:
    try:
        msg = lib().pa_strerror(code)
        return msg.decode("utf-8", "replace") if msg else f"pulse error {code}"
    except Exception:  # pragma: no cover
        return f"pulse error {code}"


def available() -> bool:
    """True if the audio backend can be used at all."""
    try:
        lib()
        return True
    except PulseUnavailable:
        return False
