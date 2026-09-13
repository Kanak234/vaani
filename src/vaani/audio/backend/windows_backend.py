"""Windows audio streams backed by WASAPI through the ``soundcard`` package."""
from __future__ import annotations

import threading
import warnings

import numpy as np

from ...core.errors import ErrorCode, Severity, VaaniError



def _load_soundcard():
    try:
        import soundcard as sc
        return sc
    except ImportError as exc:
        raise VaaniError(
            code=ErrorCode.AUDIO_BACKEND_UNAVAILABLE,
            message="Windows audio support requires the 'soundcard' package; install Vaani with the Windows extras",
            severity=Severity.FATAL,
            cause=exc,
        ) from exc


def _match(devices, requested: str | None, kind: str):
    if requested is None:
        return devices[0] if devices else None
    wanted = requested.strip().lower()
    exact = [d for d in devices if d.name.lower() == wanted]
    if exact:
        return exact[0]
    partial = [d for d in devices if wanted in d.name.lower()]
    if len(partial) == 1:
        return partial[0]
    if not devices:
        raise VaaniError(
            code=ErrorCode.AUDIO_BACKEND_UNAVAILABLE,
            message=f"no Windows {kind} devices are available",
            severity=Severity.SESSION,
        )
    names = ", ".join(d.name for d in devices[:8])
    raise VaaniError(
        code=ErrorCode.MIC_UNAVAILABLE if kind == "microphone" else ErrorCode.VIRTUAL_MIC_UNAVAILABLE,
        message=f"could not uniquely resolve {kind} {requested!r}; available: {names}",
        severity=Severity.SESSION,
        detail={"device": requested},
    )


class WindowsCaptureStream:
    """Blocking mono capture from a Windows microphone or WASAPI loopback.

    Media Foundation/WASAPI can report timestamp discontinuities when another
    application changes an endpoint or the scheduler briefly falls behind. The
    soundcard package reports those as warnings rather than failed reads; they
    are not allowed to spam the meeting console or terminate the stream.
    """

    def __init__(self, *, device: str | None = None, sample_rate: int = 16000,
                 frame_ms: int = 20, stream_name: str = "capture",
                 include_loopback: bool = False) -> None:
        del stream_name
        sc = _load_soundcard()
        microphones = sc.all_microphones(include_loopback=include_loopback)
        self._device = _match(microphones, device, "microphone")
        if self._device is None:
            raise VaaniError(code=ErrorCode.MIC_UNAVAILABLE,
                             message="no Windows microphone is available",
                             severity=Severity.SESSION)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.channels = 1
        self.frame_bytes = self.frame_samples * 2
        self._lock = threading.Lock()
        self._recorder = None
        try:
            self._recorder = self._device.recorder(samplerate=sample_rate, channels=1)
            self._recorder.__enter__()
        except Exception as exc:
            raise VaaniError(
                code=ErrorCode.MIC_UNAVAILABLE,
                message=f"could not open Windows capture device {self._device.name!r}: {exc}",
                severity=Severity.SESSION,
                cause=exc,
            ) from exc

    @property
    def device_name(self) -> str:
        return self._device.name

    @property
    def is_open(self) -> bool:
        return self._recorder is not None

    def latency_ms(self) -> float:
        return 0.0

    def read_frame(self) -> np.ndarray:
        with self._lock:
            if self._recorder is None:
                raise VaaniError(code=ErrorCode.DEVICE_DISCONNECTED,
                                 message="capture stream is closed",
                                 severity=Severity.SESSION)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="data discontinuity in recording")
                    data = self._recorder.record(numframes=self.frame_samples)
            except Exception as exc:
                raise VaaniError(code=ErrorCode.DEVICE_DISCONNECTED,
                                 message=f"Windows capture read failed: {exc}",
                                 severity=Severity.SESSION,
                                 cause=exc) from exc
        samples = np.asarray(data, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = samples.reshape(-1)
        if samples.size == 0:
            raise VaaniError(code=ErrorCode.DEVICE_DISCONNECTED,
                             message="Windows capture returned an empty frame",
                             severity=Severity.SESSION)
        return samples

    def close(self) -> None:
        with self._lock:
            recorder, self._recorder = self._recorder, None
            if recorder is not None:
                try:
                    recorder.__exit__(None, None, None)
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class WindowsPlaybackStream:
    """Blocking mono playback to a Windows speaker endpoint."""

    def __init__(self, *, device: str | None = None, sample_rate: int = 16000,
                 frame_ms: int = 20, stream_name: str = "playback",
                 require_device: bool = False) -> None:
        del stream_name, require_device
        sc = _load_soundcard()
        speakers = sc.all_speakers()
        self._device = _match(speakers, device, "speaker")
        if self._device is None:
            raise VaaniError(code=ErrorCode.VIRTUAL_MIC_UNAVAILABLE,
                             message="no Windows speaker output is available",
                             severity=Severity.SESSION)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.channels = 1
        self.frame_bytes = self.frame_samples * 2
        self.underruns = 0
        self._lock = threading.Lock()
        self._speaker = None
        try:
            self._speaker = self._device.player(samplerate=sample_rate, channels=1)
            self._speaker.__enter__()
        except Exception as exc:
            raise VaaniError(
                code=ErrorCode.VIRTUAL_MIC_UNAVAILABLE,
                message=f"could not open Windows playback device {self._device.name!r}: {exc}",
                severity=Severity.SESSION,
                cause=exc,
            ) from exc

    @property
    def device_name(self) -> str:
        return self._device.name

    @property
    def is_open(self) -> bool:
        return self._speaker is not None

    def latency_ms(self) -> float:
        return 0.0

    def write(self, samples: np.ndarray) -> None:
        payload = np.asarray(samples, dtype=np.float32).reshape(-1, 1)
        payload = np.clip(payload, -1.0, 1.0)
        with self._lock:
            if self._speaker is None:
                raise VaaniError(code=ErrorCode.VIRTUAL_MIC_UNAVAILABLE,
                                 message="playback stream is closed",
                                 severity=Severity.SESSION)
            try:
                self._speaker.play(payload)
            except Exception as exc:
                self.underruns += 1
                raise VaaniError(code=ErrorCode.VIRTUAL_MIC_UNAVAILABLE,
                                 message=f"Windows playback failed: {exc}",
                                 severity=Severity.SESSION,
                                 cause=exc) from exc

    def write_silence(self, duration_ms: float) -> None:
        n = int(self.sample_rate * duration_ms / 1000)
        self.write(np.zeros(n, dtype=np.float32))

    def drain(self) -> None:
        return

    def flush(self) -> None:
        return

    def close(self) -> None:
        with self._lock:
            speaker, self._speaker = self._speaker, None
            if speaker is not None:
                try:
                    speaker.__exit__(None, None, None)
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
