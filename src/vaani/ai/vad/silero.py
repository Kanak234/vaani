"""Silero VAD via ONNX Runtime.

Chosen over WebRTC VAD (which is faster but markedly worse on music, keyboard
noise and non-speech transients) and over a torch build (which would pull ~2.5 GB
for a 1.8 MB model -- unacceptable given the machine has ~2.9 GiB free RAM).

Runs on CPU: ONNX Runtime on this machine reports only
['AzureExecutionProvider', 'CPUExecutionProvider'] -- there is no CUDA EP -- and at
1.8 MB the model costs well under a millisecond per frame, so CPU is correct anyway.

Silero requires exactly 512 samples per call at 16 kHz (32 ms). The caller's frame
size is decoupled from that by an internal accumulator, so the rest of the app can
keep using 20 ms frames.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ...core.errors import ErrorCode, Severity, VaaniError
from ...system.platform import data_dir

_MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
#: New samples consumed per inference call at 16 kHz.
_WINDOW = 512
#: Silero v5 prepends 64 samples of context from the PREVIOUS window, so the
#: tensor handed to the model is 576 long, not 512.
#:
#: This is not documented in the model's ONNX signature -- `input` is declared
#: [None, None] -- and getting it wrong fails SILENTLY: feeding exactly 512
#: samples runs without error and returns ~0.001 for loud, clean speech. Measured
#: on this machine: 512 -> mean prob 0.001 (0% of speech frames detected),
#: 576 -> mean prob 0.589 (57% detected). Verified against real speech.
_CONTEXT = 64


class SileroVad:
    frame_ms: int
    sample_rate: int

    def __init__(self, *, model_path: Path | None = None,
                 sample_rate: int = 16000, frame_ms: int = 20,
                 auto_download: bool = True) -> None:
        if sample_rate != 16000:
            raise VaaniError(
                code=ErrorCode.CONFIG_INVALID,
                message=f"Silero VAD requires 16 kHz input, got {sample_rate}",
                severity=Severity.FATAL,
            )
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._path = model_path or _default_model_path()
        self._session = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)
        self._last_prob = 0.0
        self._auto_download = auto_download

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise VaaniError(
                code=ErrorCode.MODEL_LOAD_FAILED,
                message="onnxruntime is not installed; install it or use the energy VAD",
                severity=Severity.FATAL, cause=exc,
            ) from exc

        if not self._path.exists():
            if not self._auto_download:
                raise VaaniError(
                    code=ErrorCode.MODEL_LOAD_FAILED,
                    message=f"Silero VAD model not found at {self._path}",
                    severity=Severity.FATAL,
                )
            self._download()

        opts = ort.SessionOptions()
        # One thread: this runs per 32 ms frame in the audio path, where thread
        # pool wake-up latency costs more than the arithmetic itself.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        try:
            self._session = ort.InferenceSession(
                str(self._path), sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise VaaniError(
                code=ErrorCode.MODEL_LOAD_FAILED,
                message=f"could not load the Silero VAD model: {exc}",
                severity=Severity.FATAL, cause=exc,
            ) from exc

    def _download(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".partial")
        try:
            with urllib.request.urlopen(_MODEL_URL, timeout=30) as resp:
                tmp.write_bytes(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            raise VaaniError(
                code=ErrorCode.NETWORK_UNAVAILABLE,
                message="could not download the Silero VAD model; "
                        "connect to the internet once, or switch to the energy VAD",
                severity=Severity.FATAL, cause=exc,
            ) from exc
        tmp.replace(self._path)

    def is_speech(self, frame: np.ndarray) -> float:
        """Speech probability for `frame`.

        Frames are accumulated to Silero's required 512-sample window. Until a full
        window is available the previous probability is returned, which keeps the
        segmenter's view of the signal continuous rather than dropping to zero
        between windows.
        """
        self._ensure_loaded()
        self._pending = np.concatenate([self._pending, np.asarray(frame, dtype=np.float32)])
        while self._pending.size >= _WINDOW:
            window = self._pending[:_WINDOW]
            self._pending = self._pending[_WINDOW:]
            # Prepend the tail of the previous window; keep this window's tail for
            # the next call. Without this the model silently returns ~0 (see _CONTEXT).
            padded = np.concatenate([self._context, window])
            self._context = window[-_CONTEXT:].copy()
            try:
                out, self._state = self._session.run(
                    None,
                    {
                        "input": padded.reshape(1, -1).astype(np.float32),
                        "state": self._state,
                        "sr": np.array(self.sample_rate, dtype=np.int64),
                    },
                )
            except Exception as exc:
                raise VaaniError(
                    code=ErrorCode.VAD_FAILURE,
                    message=f"VAD inference failed: {exc}",
                    severity=Severity.SESSION, cause=exc,
                ) from exc
            self._last_prob = float(out[0][0])
        return self._last_prob

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)
        self._last_prob = 0.0


def _default_model_path() -> Path:
    return data_dir() / "models" / "silero_vad.onnx"
