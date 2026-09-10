"""Provider interfaces (FR-13).

Every AI capability is defined here as a Protocol plus a small abstract base that
supplies the bookkeeping every provider needs (health, warmup, network policy).

The single most important rule in this file is `requires_network`. The local-only
guarantee (AC-17.2) is enforced by the registry refusing to hand out a provider
whose `requires_network` is True while the session is local-only -- it is not left
to each provider to remember to check.
"""
from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.errors import ErrorCode, Severity, VaaniError
from ..core.types import (
    SynthesizedAudio,
    Transcript,
    Translation,
)


class ProviderKind(enum.Enum):
    VAD = "vad"
    STT = "stt"
    TRANSLATION = "translation"
    TTS = "tts"


class HealthState(enum.Enum):
    UNKNOWN = "unknown"
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(slots=True)
class ProviderHealth:
    state: HealthState
    detail: str = ""
    last_error: ErrorCode | None = None


class Provider(abc.ABC):
    """Common base for every swappable AI component."""

    #: Stable identifier used in config, the DB and logs.
    key: str = "unset"
    kind: ProviderKind
    display_name: str = "Unset"
    #: If True this provider sends data off the machine. Read by the registry
    #: before the provider is ever handed to the pipeline.
    requires_network: bool = False

    def __init__(self) -> None:
        self._health = ProviderHealth(HealthState.UNKNOWN)

    @property
    def is_local(self) -> bool:
        return not self.requires_network

    @property
    def health(self) -> ProviderHealth:
        return self._health

    def _set_health(self, state: HealthState, detail: str = "",
                    last_error: ErrorCode | None = None) -> None:
        self._health = ProviderHealth(state, detail, last_error)

    @abc.abstractmethod
    def warmup(self) -> None:
        """Load models and make the first call fast.

        Must be safe to call more than once. Should raise VaaniError with
        MODEL_LOAD_FAILED or INSUFFICIENT_MEMORY rather than a bare exception,
        because the UI needs to tell the user which of those two happened.
        """

    def shutdown(self) -> None:
        """Release models and memory. Must be idempotent."""
        self._set_health(HealthState.UNKNOWN, "shut down")

    def _fail(self, code: ErrorCode, message: str, *,
              severity: Severity = Severity.UTTERANCE,
              cause: BaseException | None = None) -> VaaniError:
        self._set_health(HealthState.DEGRADED, message, code)
        return VaaniError(code=code, message=message, severity=severity,
                          provider_key=self.key, cause=cause)


# --------------------------------------------------------------------------- VAD

@runtime_checkable
class VadProvider(Protocol):
    """Frame-level speech detection.

    Deliberately frame-at-a-time rather than buffer-at-a-time: utterance
    segmentation policy (hangover, min/max duration) belongs to the segmenter in
    audio/, not to the model, so the two can be tuned independently.
    """

    frame_ms: int
    sample_rate: int

    def is_speech(self, frame: np.ndarray) -> float:
        """Return speech probability in [0, 1] for one frame."""
        ...

    def reset(self) -> None:
        """Clear internal recurrent state at an utterance boundary."""
        ...


# --------------------------------------------------------------------------- STT

class SpeechRecognizer(Provider):
    kind = ProviderKind.STT

    @abc.abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int, *,
                   language_hint: str | None = None,
                   context_prompt: str | None = None) -> Transcript:
        """Transcribe one complete utterance.

        `language_hint` may be None, which MUST mean "detect" and must not be
        silently coerced to a single language -- code-mixed speech depends on it
        (AC-04.2). `context_prompt` biases decoding toward recent vocabulary.
        """

    def supports_streaming(self) -> bool:
        return False


# ------------------------------------------------------------------- Translation

class TranslationEngine(Provider):
    kind = ProviderKind.TRANSLATION

    @abc.abstractmethod
    def translate(self, text: str, *, source_language: str, target_language: str,
                  context: list[tuple[str, str]] | None = None) -> Translation:
        """Translate one utterance.

        `context` is a list of (source, translated) pairs, oldest first, already
        trimmed to the configured window by the context engine. Providers must
        treat it as advisory and must not require it.
        """

    def supports_code_mixed(self) -> bool:
        """True if the engine handles mixed-script input directly.

        Engines that return False get romanised/normalised input from the
        pipeline instead of raw Hinglish.
        """
        return False


# --------------------------------------------------------------------------- TTS

class VoiceSynthesizer(Provider):
    kind = ProviderKind.TTS

    #: Whether this backend can speak in an enrolled personal voice at all.
    supports_voice_cloning: bool = False

    @abc.abstractmethod
    def synthesize(self, text: str, *, voice_profile_id: str | None = None,
                   speed: float = 1.0) -> SynthesizedAudio:
        """Render text to audio.

        If `voice_profile_id` is given but this backend cannot honour it, it MUST
        either raise VOICE_PROFILE_INVALID or return audio with
        `is_fallback_voice=True`. Silently returning a generic voice is a bug --
        the user would not know they are being represented by a stranger's voice.
        """

    def supports_streaming(self) -> bool:
        return False
