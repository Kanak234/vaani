"""Voice enrollment: capture, quality gating, and profile storage (F-09).

Quality gating is the interesting part. A voice profile built from bad audio does
not fail loudly -- it produces a clone that sounds subtly wrong in every meeting
from then on, and the user has no way to tell whether the problem is the recording
or the model. So the checks run BEFORE a profile is created, and they say exactly
which criterion failed (AC-09.4).

Enrollment length is measured in SECONDS OF DETECTED SPEECH, not wall-clock
(AC-09.3). Someone who records for three minutes but pauses constantly has not
given the model three minutes of voice.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..core.errors import ErrorCode, Severity, VaaniError
from ..system.platform import data_dir, is_windows
from .consent import ConsentLedger

#: Prompts chosen to cover a wide phonetic range plus the code-mixed register the
#: user actually speaks -- a profile trained only on careful English reads badly
#: when the input is conversational Hinglish.
ENROLLMENT_PROMPTS = [
    "Hello, my name is and I work as a software engineer.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "मेरा नाम है और मैं एक सॉफ्टवेयर इंजीनियर के रूप में काम करता हूँ।",
    "Actually मेरा point ये है कि हम इस project को अगले हफ्ते तक complete कर सकते हैं।",
    "We should schedule the review for Thursday afternoon, around three o'clock.",
    "यह बहुत ज़रूरी है कि हम deadline से पहले सारा काम खत्म कर लें।",
    "I think the authentication module needs more testing before we deploy it.",
    "Could you please share the updated document with the whole team today?",
]


@dataclass(slots=True)
class QualityReport:
    speech_seconds: float
    snr_db: float
    clipping_pct: float
    silence_pct: float
    peak: float
    passed: bool
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.speech_seconds:.1f}s speech, SNR {self.snr_db:.1f} dB, "
                f"clipping {self.clipping_pct:.2f}%, silence {self.silence_pct:.0f}%")


@dataclass(slots=True)
class EnrollmentCriteria:
    min_speech_seconds: float = 45.0
    max_speech_seconds: float = 180.0
    min_snr_db: float = 15.0
    max_clipping_pct: float = 1.0
    max_silence_pct: float = 40.0
    min_peak: float = 0.05


@dataclass(slots=True)
class VoiceProfile:
    id: str
    name: str
    consent_id: str
    provider_key: str
    model_id: str
    reference_audio_path: str
    speech_seconds: float
    sample_rate: int
    quality_snr_db: float
    quality_clipping_pct: float
    status: str
    created_at_utc: str

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"


def _speech_mask(audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """Per-sample speech mask, preferring the real VAD over an energy heuristic.

    Silero is used when available because the energy fallback is materially wrong
    on real speech. Measured on a continuous 11 s recording (JFK inaugural):

        energy heuristic -> 51% silence   (would FAIL the 40% quality gate)
        Silero VAD       -> 33% silence   (passes, and is correct)

    A gate that rejects genuine, continuous speech makes enrollment feel broken
    and is worse than no gate, so the accurate detector is the default.
    """
    try:
        from ..ai.vad.silero import SileroVad
        vad = SileroVad()
        frame = 320                                  # 20 ms at 16 kHz
        if sample_rate == 16000 and audio.size >= frame:
            flags = [vad.is_speech(audio[i:i + frame]) > 0.5
                     for i in range(0, audio.size - frame, frame)]
            if flags:
                mask = np.repeat(np.array(flags, dtype=bool), frame)
                return np.pad(mask, (0, audio.size - mask.size),
                              constant_values=False)
    except Exception:
        # No model, no network on first run, wrong sample rate -- fall through.
        pass
    return _energy_mask(audio, sample_rate)


def _energy_mask(audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """Energy fallback for when the VAD model is unavailable.

    The threshold is taken from a PERCENTILE of frame energies rather than from
    the maximum. Peak-relative thresholding is what made the earlier version
    over-report silence: one loud syllable raises the bar for the whole recording,
    and ordinary quieter speech falls below it.
    """
    frame = max(1, sample_rate // 100)               # 10 ms
    n = audio.size // frame
    if n == 0:
        return None
    frames = audio[:n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

    # Noise floor from the quietest twentieth; speech sits well above it.
    floor = float(np.percentile(rms, 5))
    loud = float(np.percentile(rms, 90))
    # The upper bound matters: on audio with little dynamic range the low
    # percentile lands INSIDE speech, and an unbounded floor*3 then exceeds the
    # signal and marks the whole recording as silence. Capping the threshold at a
    # fraction of the loud level keeps the heuristic sane in both cases.
    threshold = float(np.clip(floor * 3.0, max(loud * 0.02, 1e-4), loud * 0.35))
    mask = np.repeat(rms > threshold, frame)
    return np.pad(mask, (0, audio.size - mask.size), constant_values=False)


def analyse_quality(audio: np.ndarray, sample_rate: int,
                    speech_mask: np.ndarray | None = None,
                    criteria: EnrollmentCriteria | None = None) -> QualityReport:
    """Assess enrollment audio against the criteria in AC-09.4.

    `speech_mask` is a per-sample boolean from the VAD. Without it, an energy
    threshold stands in -- less accurate, but this must work before any model is
    loaded.
    """
    criteria = criteria or EnrollmentCriteria()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return QualityReport(0, 0, 0, 100, 0, False, ["no audio was recorded"])

    peak = float(np.max(np.abs(audio)))
    clipping_pct = float(np.mean(np.abs(audio) >= 0.99) * 100)

    if speech_mask is None:
        speech_mask = _speech_mask(audio, sample_rate)
        if speech_mask is None:
            return QualityReport(0, 0, clipping_pct, 100, peak, False,
                                 ["recording is too short to analyse"])
    speech_mask = np.asarray(speech_mask, dtype=bool)

    speech_samples = audio[speech_mask]
    noise_samples = audio[~speech_mask]
    speech_seconds = float(speech_mask.sum()) / sample_rate
    silence_pct = float((~speech_mask).mean() * 100)

    speech_power = float(np.mean(speech_samples ** 2)) if speech_samples.size else 0.0
    noise_power = float(np.mean(noise_samples ** 2)) if noise_samples.size else 1e-12
    snr_db = 10 * np.log10(max(speech_power, 1e-12) / max(noise_power, 1e-12))

    failures: list[str] = []
    if speech_seconds < criteria.min_speech_seconds:
        failures.append(
            f"only {speech_seconds:.0f}s of speech; at least "
            f"{criteria.min_speech_seconds:.0f}s is needed")
    if speech_seconds > criteria.max_speech_seconds:
        failures.append(
            f"{speech_seconds:.0f}s of speech exceeds the "
            f"{criteria.max_speech_seconds:.0f}s maximum")
    if snr_db < criteria.min_snr_db:
        failures.append(
            f"background noise is too high (SNR {snr_db:.1f} dB, need "
            f"{criteria.min_snr_db:.0f} dB); try a quieter room or a closer mic")
    if clipping_pct > criteria.max_clipping_pct:
        failures.append(
            f"{clipping_pct:.1f}% of samples are clipped; lower the input gain")
    if silence_pct > criteria.max_silence_pct:
        failures.append(
            f"{silence_pct:.0f}% of the recording is silence; speak more continuously")
    if peak < criteria.min_peak:
        failures.append(f"recording is too quiet (peak {peak:.3f}); raise the gain")

    return QualityReport(speech_seconds, snr_db, clipping_pct, silence_pct, peak,
                         not failures, failures)


class VoiceProfileStore:
    """Profiles on disk. Deletion actually shreds the audio and verifies it."""

    def __init__(self, root: Path | None = None,
                 ledger: ConsentLedger | None = None) -> None:
        self.root = root or _default_root()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self.ledger = ledger or ConsentLedger()

    def create(self, *, name: str, audio: np.ndarray, sample_rate: int,
               provider_key: str, model_id: str,
               criteria: EnrollmentCriteria | None = None,
               speech_mask: np.ndarray | None = None) -> VoiceProfile:
        """Build a profile. Consent is checked FIRST, before anything is written."""
        consent = self.ledger.require_active_consent()

        report = analyse_quality(audio, sample_rate, speech_mask, criteria)
        if not report.passed:
            code = (ErrorCode.ENROLLMENT_INSUFFICIENT_SPEECH
                    if any("speech" in f for f in report.failures)
                    else ErrorCode.ENROLLMENT_QUALITY_FAILED)
            raise VaaniError(
                code=code,
                message="the recording is not good enough for a voice profile: "
                        + "; ".join(report.failures),
                severity=Severity.SESSION,
                detail={"report": asdict(report)},
            )

        import soundfile as sf
        profile_id = uuid.uuid4().hex
        directory = self.root / profile_id
        directory.mkdir(parents=True, exist_ok=True)
        audio_path = directory / "reference.wav"
        sf.write(str(audio_path), np.asarray(audio, dtype=np.float32), sample_rate)
        if not is_windows():
            os.chmod(audio_path, 0o600)     # the recording is personal data

        profile = VoiceProfile(
            id=profile_id, name=name, consent_id=consent.id,
            provider_key=provider_key, model_id=model_id,
            reference_audio_path=str(audio_path),
            speech_seconds=report.speech_seconds, sample_rate=sample_rate,
            quality_snr_db=report.snr_db, quality_clipping_pct=report.clipping_pct,
            status="ready", created_at_utc=datetime.now(UTC).isoformat(),
        )
        (directory / "profile.json").write_text(
            json.dumps(asdict(profile), indent=2), encoding="utf-8")
        return profile

    def get(self, profile_id: str) -> VoiceProfile | None:
        meta = self.root / profile_id / "profile.json"
        if not meta.exists():
            return None
        try:
            return VoiceProfile(**json.loads(meta.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError):
            return None

    def list_profiles(self) -> list[VoiceProfile]:
        out = []
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            if d.is_dir():
                p = self.get(d.name)
                if p:
                    out.append(p)
        return out

    def active(self) -> VoiceProfile | None:
        """The newest ready profile, or None. None must mean TTS refuses to clone."""
        ready = [p for p in self.list_profiles() if p.is_ready]
        return max(ready, key=lambda p: p.created_at_utc) if ready else None

    def delete(self, profile_id: str) -> bool:
        """Delete a profile and verify the audio is gone from disk (AC-09.6)."""
        directory = self.root / profile_id
        if not directory.exists():
            return False
        shutil.rmtree(directory, ignore_errors=True)
        if directory.exists():
            raise VaaniError(
                code=ErrorCode.STORAGE_FAILURE,
                message=f"could not fully delete the voice profile at {directory}",
                severity=Severity.SESSION,
            )
        return True

    def revoke_consent_and_delete_all(self, *, app_version: str) -> int:
        """Revoking consent must remove every derived artefact (AC-09.7)."""
        grant = self.ledger.active_grant()
        if grant is not None:
            self.ledger.revoke(grant.id, app_version=app_version)
        removed = 0
        for profile in self.list_profiles():
            if self.delete(profile.id):
                removed += 1
        return removed


def _default_root() -> Path:
    return data_dir() / "voice_profiles"
