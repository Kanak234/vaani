"""Turns a stream of VAD probabilities into utterance boundaries.

This module owns the product's dominant tunable latency cost. Segmentation policy
lives here rather than in the VAD provider so that the two can be changed
independently -- swapping Silero for the energy VAD must not change how long we
wait before deciding someone stopped talking.

The four knobs:

  speech_threshold   probability above which a frame counts as speech
  start_frames       consecutive speech frames needed to open an utterance.
                     Guards against a cough or a door opening a segment.
  hangover_ms        silence after speech before the utterance is closed.
                     THE key latency knob: it is added in full to every
                     utterance's user-perceived delay. Too short cuts people off
                     mid-sentence; too long makes conversation feel laggy.
  pre_roll_ms        audio retained from BEFORE speech was detected. Without it
                     the recogniser loses the utterance's first consonant, which
                     is exactly where Hindi's aspirated stops carry meaning
                     (क/ख, त/थ). Costs nothing in latency -- it is already buffered.
"""
from __future__ import annotations

import enum
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..core.types import PerformanceMode


class SegmenterState(enum.Enum):
    SILENCE = "silence"
    MAYBE_SPEECH = "maybe_speech"
    SPEECH = "speech"
    HANGOVER = "hangover"


@dataclass(slots=True)
class SegmenterConfig:
    speech_threshold: float = 0.5
    start_frames: int = 3
    hangover_ms: int = 500
    pre_roll_ms: int = 300
    min_utterance_ms: int = 250
    #: Hard cap. A monologue is force-split rather than growing without bound,
    #: which would blow both the latency budget and the STT memory footprint.
    max_utterance_ms: int = 25_000

    @classmethod
    def for_mode(cls, mode: PerformanceMode) -> SegmenterConfig:
        """Latency/accuracy presets.

        Only the hangover and the start guard really move: they are what trades
        responsiveness against not clipping people mid-thought.
        """
        if mode is PerformanceMode.LOW_LATENCY:
            # 250 ms is about the shortest that does not cut people mid-sentence:
            # normal speech has 150-200 ms gaps between words and after commas.
            return cls(speech_threshold=0.45, start_frames=2, hangover_ms=250,
                       pre_roll_ms=200, max_utterance_ms=12_000)
        if mode is PerformanceMode.QUALITY:
            return cls(speech_threshold=0.55, start_frames=4, hangover_ms=800,
                       pre_roll_ms=400, max_utterance_ms=30_000)
        # BALANCED: 400 ms rather than 500. The hangover is added in full to
        # every single utterance's perceived latency, so it is the cheapest
        # 100 ms in the whole pipeline to remove.
        return cls(hangover_ms=400)


@dataclass(slots=True)
class Segment:
    """A closed utterance, ready for recognition."""

    audio: np.ndarray
    sample_rate: int
    #: monotonic time when speech ENDED -- time zero for latency accounting.
    speech_end_time: float
    speech_ms: float
    truncated: bool = False


@dataclass(slots=True)
class UtteranceSegmenter:
    config: SegmenterConfig
    sample_rate: int = 16000
    frame_ms: int = 20

    _state: SegmenterState = field(default=SegmenterState.SILENCE, init=False)
    _pre_roll: deque = field(init=False, repr=False)
    _current: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _speech_run: int = field(default=0, init=False)
    _silence_ms: float = field(default=0.0, init=False)
    _speech_ms: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        n = max(1, int(self.config.pre_roll_ms / self.frame_ms))
        self._pre_roll = deque(maxlen=n)

    @property
    def state(self) -> SegmenterState:
        return self._state

    @property
    def is_capturing(self) -> bool:
        return self._state in (SegmenterState.SPEECH, SegmenterState.HANGOVER)

    def push(self, frame: np.ndarray, speech_prob: float) -> Segment | None:
        """Feed one frame. Returns a Segment when an utterance has just closed."""
        is_speech = speech_prob >= self.config.speech_threshold

        if self._state is SegmenterState.SILENCE:
            self._pre_roll.append(frame)
            if is_speech:
                self._speech_run = 1
                self._state = (SegmenterState.SPEECH
                               if self.config.start_frames <= 1
                               else SegmenterState.MAYBE_SPEECH)
                if self._state is SegmenterState.SPEECH:
                    self._open()
            return None

        if self._state is SegmenterState.MAYBE_SPEECH:
            self._pre_roll.append(frame)
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self.config.start_frames:
                    self._state = SegmenterState.SPEECH
                    self._open()
            else:
                # False start -- a cough, a click. Discard and go back to silence.
                self._speech_run = 0
                self._state = SegmenterState.SILENCE
            return None

        # SPEECH or HANGOVER: we are recording.
        self._current.append(frame)
        if is_speech:
            self._speech_ms += self.frame_ms
            self._state = SegmenterState.SPEECH
            self._silence_ms = 0.0
        else:
            self._silence_ms += self.frame_ms
            self._state = SegmenterState.HANGOVER
            if self._silence_ms >= self.config.hangover_ms:
                return self._close(truncated=False)

        if self._duration_ms() >= self.config.max_utterance_ms:
            return self._close(truncated=True)
        return None

    def _open(self) -> None:
        """Start recording, seeding with the pre-roll so no onset is lost."""
        self._current = list(self._pre_roll)
        self._pre_roll.clear()
        self._speech_ms = self.frame_ms * len(self._current)
        self._silence_ms = 0.0

    def _duration_ms(self) -> float:
        return len(self._current) * self.frame_ms

    def _close(self, *, truncated: bool) -> Segment | None:
        audio = (np.concatenate(self._current) if self._current
                 else np.zeros(0, dtype=np.float32))
        speech_ms = self._speech_ms
        self._reset_capture()

        if speech_ms < self.config.min_utterance_ms:
            # Too short to be a real utterance. Dropping it here, before STT, is
            # what stops a keyboard click from becoming a spoken sentence.
            return None

        # Trim the trailing hangover silence: it is dead weight for the recogniser
        # and for the latency budget, and it is not part of the utterance.
        if not truncated:
            keep = int(self.sample_rate * (self.config.hangover_ms * 0.5) / 1000)
            drop = max(0, int(self.sample_rate * self.config.hangover_ms / 1000) - keep)
            if drop and audio.size > drop:
                audio = audio[:-drop]

        return Segment(
            audio=audio,
            sample_rate=self.sample_rate,
            speech_end_time=time.monotonic(),
            speech_ms=speech_ms,
            truncated=truncated,
        )

    def _reset_capture(self) -> None:
        self._current = []
        self._speech_run = 0
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._state = SegmenterState.SILENCE
        self._pre_roll.clear()

    def flush(self) -> Segment | None:
        """Close any in-progress utterance -- used on pause/stop."""
        if self.is_capturing:
            return self._close(truncated=True)
        self._reset_capture()
        return None

    def reset(self) -> None:
        self._reset_capture()
