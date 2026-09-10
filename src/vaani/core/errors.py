"""Typed error taxonomy.

Every failure in the pipeline maps to one of these. The reason this is a closed
taxonomy rather than free-form exceptions: the product's central safety property
is "a failed stage never produces audio" (AC-12.4), and that is only auditable if
every failure is classified and every classification has a defined output policy.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Severity(enum.Enum):
    #: Utterance is lost, session continues. The common case.
    UTTERANCE = "utterance"
    #: Session must pause and recover (device lost, provider unhealthy).
    SESSION = "session"
    #: Cannot start or continue at all.
    FATAL = "fatal"


class ErrorCode(enum.Enum):
    """Stable codes. These appear in the DB and in user-facing detail views."""

    # --- devices / audio -------------------------------------------------
    MIC_UNAVAILABLE = "MIC_UNAVAILABLE"
    MIC_PERMISSION_DENIED = "MIC_PERMISSION_DENIED"
    DEVICE_DISCONNECTED = "DEVICE_DISCONNECTED"
    AUDIO_BACKEND_UNAVAILABLE = "AUDIO_BACKEND_UNAVAILABLE"
    VIRTUAL_MIC_CREATE_FAILED = "VIRTUAL_MIC_CREATE_FAILED"
    VIRTUAL_MIC_UNAVAILABLE = "VIRTUAL_MIC_UNAVAILABLE"
    BUFFER_UNDERRUN = "BUFFER_UNDERRUN"
    FEEDBACK_LOOP_DETECTED = "FEEDBACK_LOOP_DETECTED"

    # --- AI stages -------------------------------------------------------
    VAD_FAILURE = "VAD_FAILURE"
    STT_FAILURE = "STT_FAILURE"
    STT_TIMEOUT = "STT_TIMEOUT"
    STT_EMPTY = "STT_EMPTY"
    LANGID_UNSUPPORTED = "LANGID_UNSUPPORTED"
    TRANSLATION_FAILURE = "TRANSLATION_FAILURE"
    TRANSLATION_TIMEOUT = "TRANSLATION_TIMEOUT"
    TRANSLATION_EMPTY = "TRANSLATION_EMPTY"
    TTS_FAILURE = "TTS_FAILURE"
    TTS_TIMEOUT = "TTS_TIMEOUT"

    # --- voice profile ---------------------------------------------------
    VOICE_PROFILE_MISSING = "VOICE_PROFILE_MISSING"
    VOICE_PROFILE_INVALID = "VOICE_PROFILE_INVALID"
    VOICE_CONSENT_MISSING = "VOICE_CONSENT_MISSING"
    ENROLLMENT_QUALITY_FAILED = "ENROLLMENT_QUALITY_FAILED"
    ENROLLMENT_INSUFFICIENT_SPEECH = "ENROLLMENT_INSUFFICIENT_SPEECH"

    # --- providers / infra ----------------------------------------------
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_RATE_LIMITED = "PROVIDER_RATE_LIMITED"
    PROVIDER_AUTH_FAILED = "PROVIDER_AUTH_FAILED"
    NETWORK_UNAVAILABLE = "NETWORK_UNAVAILABLE"
    #: A network provider was selected while the session is local-only.
    LOCAL_ONLY_VIOLATION = "LOCAL_ONLY_VIOLATION"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"
    CONFIG_INVALID = "CONFIG_INVALID"
    STORAGE_FAILURE = "STORAGE_FAILURE"
    INTERNAL = "INTERNAL"


#: Codes that must never be retried automatically -- retrying either cannot help
#: or would repeat a privacy violation.
NON_RETRYABLE: frozenset[ErrorCode] = frozenset({
    ErrorCode.MIC_PERMISSION_DENIED,
    ErrorCode.VOICE_PROFILE_MISSING,
    ErrorCode.VOICE_CONSENT_MISSING,
    ErrorCode.LOCAL_ONLY_VIOLATION,
    ErrorCode.PROVIDER_AUTH_FAILED,
    ErrorCode.CONFIG_INVALID,
    ErrorCode.FEEDBACK_LOOP_DETECTED,
})


@dataclass(slots=True)
class VaaniError(Exception):
    """Base error. Carries everything the UI needs to explain itself."""

    code: ErrorCode
    message: str
    severity: Severity = Severity.UTTERANCE
    stage: str | None = None
    provider_key: str | None = None
    #: Non-sensitive structured detail. Must never contain audio, transcripts,
    #: or credentials -- it is written to the DB and to exported reports.
    detail: dict[str, Any] = field(default_factory=dict)
    cause: BaseException | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    @property
    def retryable(self) -> bool:
        return self.code not in NON_RETRYABLE

    def user_message(self) -> str:
        """Plain-language text for the UI. No codes, no stack traces."""
        return _USER_MESSAGES.get(self.code, self.message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        where = f" [{self.stage}]" if self.stage else ""
        return f"{self.code.value}{where}: {self.message}"


_USER_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.MIC_UNAVAILABLE: "The selected microphone is not available.",
    ErrorCode.MIC_PERMISSION_DENIED: "Microphone access was denied. Grant permission and try again.",
    ErrorCode.DEVICE_DISCONNECTED: "The audio device was disconnected.",
    ErrorCode.AUDIO_BACKEND_UNAVAILABLE: "No supported audio system was found.",
    ErrorCode.VIRTUAL_MIC_CREATE_FAILED: "Could not create the virtual microphone.",
    ErrorCode.VIRTUAL_MIC_UNAVAILABLE: "The virtual microphone is not available.",
    ErrorCode.BUFFER_UNDERRUN: "Audio output fell behind and a gap was inserted.",
    ErrorCode.FEEDBACK_LOOP_DETECTED: "Output is routed back into the input. Change the input device.",
    ErrorCode.STT_FAILURE: "Speech could not be recognised.",
    ErrorCode.STT_TIMEOUT: "Speech recognition took too long and was cancelled.",
    ErrorCode.STT_EMPTY: "No speech was recognised in that segment.",
    ErrorCode.LANGID_UNSUPPORTED: "That language is not supported yet.",
    ErrorCode.TRANSLATION_FAILURE: "The translation could not be produced.",
    ErrorCode.TRANSLATION_TIMEOUT: "Translation took too long and was cancelled.",
    ErrorCode.TTS_FAILURE: "Speech could not be generated.",
    ErrorCode.VOICE_PROFILE_MISSING: "No voice profile is set up. Enroll your voice first.",
    ErrorCode.VOICE_CONSENT_MISSING: "Voice enrollment requires consent.",
    ErrorCode.ENROLLMENT_INSUFFICIENT_SPEECH: "Not enough speech was recorded.",
    ErrorCode.ENROLLMENT_QUALITY_FAILED: "The recording quality was too low to build a voice profile.",
    ErrorCode.NETWORK_UNAVAILABLE: "No network connection.",
    ErrorCode.LOCAL_ONLY_VIOLATION: "That provider needs the internet, but local-only mode is on.",
    ErrorCode.INSUFFICIENT_MEMORY: "Not enough memory to load that model. Try a smaller one.",
    ErrorCode.MODEL_LOAD_FAILED: "The model could not be loaded.",
}
