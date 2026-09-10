"""Energy + zero-crossing VAD.

This is the always-available baseline: pure numpy, no model download, no ONNX
runtime. It is genuinely worse than Silero on noisy input, and it is not the
default -- but it means the pipeline can always run, which matters for first-launch
and for diagnostics that need to isolate whether a problem is in the model layer.

The noise floor adapts during silence only, so a speaker holding a long vowel does
not get absorbed into the floor.
"""
from __future__ import annotations

import numpy as np


class EnergyVad:
    frame_ms: int
    sample_rate: int

    def __init__(self, *, sample_rate: int = 16000, frame_ms: int = 20,
                 initial_floor_db: float = -55.0,
                 threshold_above_floor_db: float = 12.0,
                 floor_adapt_rate: float = 0.02) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._floor_db = initial_floor_db
        self._threshold_db = threshold_above_floor_db
        self._adapt = floor_adapt_rate
        self._initial_floor = initial_floor_db

    @staticmethod
    def _rms_db(frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
        return 20.0 * np.log10(max(rms, 1e-9))

    @staticmethod
    def _zcr(frame: np.ndarray) -> float:
        if frame.size < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(np.signbit(frame).astype(np.int8)))))

    def is_speech(self, frame: np.ndarray) -> float:
        level = self._rms_db(frame)
        margin = level - self._floor_db

        if margin < self._threshold_db:
            # Adapt only while we believe this is silence.
            self._floor_db = (1 - self._adapt) * self._floor_db + self._adapt * level
            return 0.0

        # A high zero-crossing rate at low level is usually hiss, not voice.
        zcr = self._zcr(frame)
        if zcr > 0.35 and margin < self._threshold_db * 2:
            return 0.25

        # Map margin onto a probability that saturates ~20 dB above the floor.
        return float(min(1.0, 0.5 + 0.5 * (margin - self._threshold_db) / 20.0))

    def reset(self) -> None:
        self._floor_db = self._initial_floor
