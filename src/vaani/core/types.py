"""Shared value types that cross module boundaries.

These are deliberately plain dataclasses with no provider-specific fields, so that
swapping a provider cannot leak its vocabulary into the rest of the application.
"""
from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field

import numpy as np


class PerformanceMode(enum.Enum):
    LOW_LATENCY = "low_latency"
    BALANCED = "balanced"
    QUALITY = "quality"


class SourceLanguageMode(enum.Enum):
    AUTO = "auto"
    HINDI = "hi"
    ENGLISH = "en"
    HINGLISH = "hinglish"


class SuppressionReason(enum.Enum):
    LOW_STT_CONFIDENCE = "low_stt_confidence"
    LOW_TRANSLATION_CONFIDENCE = "low_translation_confidence"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    STAGE_FAILURE = "stage_failure"
    USER_MANUAL = "user_manual"
    EMERGENCY_STOP = "emergency_stop"
    EMPTY_RESULT = "empty_result"


@dataclass(slots=True)
class AudioChunk:
    """Mono PCM. Always float32 in [-1, 1] once inside the pipeline.

    The backend hands us int16; conversion happens exactly once, at the capture
    boundary, so no downstream stage has to know about sample formats.
    """

    samples: np.ndarray
    sample_rate: int
    #: monotonic seconds when the FIRST sample of this chunk was captured
    timestamp: float

    @property
    def duration_ms(self) -> float:
        return len(self.samples) / self.sample_rate * 1000.0


@dataclass(slots=True)
class Utterance:
    """A single detected span of speech, carried through every stage."""

    session_id: str
    seq: int
    audio: np.ndarray
    sample_rate: int
    #: monotonic time at which VAD decided speech had ENDED. This is time zero for
    #: the user-perceived latency measurement (NFR-1), not the time capture began.
    speech_end_time: float
    speech_ms: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def latency_so_far_ms(self) -> float:
        return (time.monotonic() - self.speech_end_time) * 1000.0


@dataclass(slots=True)
class Transcript:
    text: str
    #: {"hi": 0.6, "en": 0.4} -- never a single hard label, so that code-mixed
    #: speech is representable (AC-04.2).
    language_distribution: dict[str, float]
    confidence: float
    is_final: bool = True
    duration_ms: float = 0.0

    @property
    def dominant_language(self) -> str:
        if not self.language_distribution:
            return "unknown"
        return max(self.language_distribution.items(), key=lambda kv: kv[1])[0]

    @property
    def is_code_mixed(self) -> bool:
        """True when no single language accounts for most of the utterance."""
        if len(self.language_distribution) < 2:
            return False
        top = sorted(self.language_distribution.values(), reverse=True)
        return top[1] >= 0.15


@dataclass(slots=True)
class Translation:
    text: str
    source_language: str
    target_language: str
    confidence: float
    used_context_turns: int = 0
    provider_key: str = ""


@dataclass(slots=True)
class SynthesizedAudio:
    samples: np.ndarray
    sample_rate: int
    voice_profile_id: str | None = None
    #: True when a generic voice was used because the personal voice was
    #: unavailable. The UI MUST surface this -- silently substituting a different
    #: voice would break the product's core promise.
    is_fallback_voice: bool = False


@dataclass(slots=True)
class StageTiming:
    stage: str
    duration_ms: float
    provider_key: str | None = None
    queued_ms: float | None = None
    succeeded: bool = True
    error_code: str | None = None


@dataclass(slots=True)
class UtteranceResult:
    """Everything that happened to one utterance. Written to the DB, shown in the UI."""

    utterance: Utterance
    transcript: Transcript | None = None
    translation: Translation | None = None
    audio: SynthesizedAudio | None = None
    timings: list[StageTiming] = field(default_factory=list)
    suppressed: bool = False
    suppression_reason: SuppressionReason | None = None
    error_code: str | None = None
    total_latency_ms: float | None = None

    def stage_ms(self, stage: str) -> float | None:
        for t in self.timings:
            if t.stage == stage:
                return t.duration_ms
        return None
