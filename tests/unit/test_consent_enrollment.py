"""Consent ledger and enrollment quality gating.

Cloning is the one capability here that can do real harm, so these tests assert
the guard rails rather than the happy path.
"""
from __future__ import annotations

import numpy as np
import pytest

from vaani.core.errors import ErrorCode, VaaniError
from vaani.voice.consent import (
    CONFIRMATION_PHRASE,
    ConsentLedger,
    consent_text_hash,
)
from vaani.voice.enrollment import (
    EnrollmentCriteria,
    VoiceProfileStore,
    analyse_quality,
)

SR = 16000


@pytest.fixture
def ledger(tmp_path):
    return ConsentLedger(tmp_path / "consent.jsonl")


@pytest.fixture
def store(tmp_path, ledger):
    return VoiceProfileStore(tmp_path / "profiles", ledger)


def grant(ledger):
    return ledger.grant(subject_label="Kanak", confirmation=CONFIRMATION_PHRASE,
                        app_version="0.1.0")


def mask_for(audio, sr=SR):
    """Energy-derived mask for synthetic fixtures.

    These fixtures are harmonic tones, not speech, and Silero correctly refuses to
    call them speech. Since these tests exercise the quality CRITERIA rather than
    the detector, they supply the mask explicitly.
    """
    from vaani.voice.enrollment import _energy_mask
    return _energy_mask(audio, sr)


def speech(seconds=60.0, sr=SR, amp=0.3, noise=0.002, silence_frac=0.1):
    """Voice-like audio: harmonic stack, modest noise floor, a little silence."""
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / sr
    sig = sum(np.sin(2 * np.pi * f * t) * a
              for f, a in [(120, 1.0), (240, 0.5), (700, 0.3), (1500, 0.15)])
    sig = (sig / np.max(np.abs(sig)) * amp).astype(np.float32)
    sig += np.random.default_rng(0).standard_normal(n).astype(np.float32) * noise
    if silence_frac > 0:
        sig[: int(n * silence_frac)] = np.random.default_rng(1).standard_normal(
            int(n * silence_frac)).astype(np.float32) * noise
    return sig.astype(np.float32)


# --- consent ----------------------------------------------------------------

def test_no_consent_by_default(ledger):
    assert not ledger.has_active_consent()
    with pytest.raises(VaaniError) as e:
        ledger.require_active_consent()
    assert e.value.code is ErrorCode.VOICE_CONSENT_MISSING


def test_grant_requires_exact_phrase(ledger):
    for bad in ("yes", "i agree", "I CONSEN", ""):
        with pytest.raises(VaaniError):
            ledger.grant(subject_label="Kanak", confirmation=bad, app_version="0.1.0")
    assert not ledger.has_active_consent()


def test_grant_requires_a_subject(ledger):
    with pytest.raises(VaaniError):
        ledger.grant(subject_label="  ", confirmation=CONFIRMATION_PHRASE,
                     app_version="0.1.0")


def test_grant_records_the_exact_text_hash(ledger):
    r = grant(ledger)
    assert r.consent_text_hash == consent_text_hash()
    assert r.granted_at_utc.endswith("+00:00")     # UTC, per AC-09.2
    assert ledger.has_active_consent()


def test_revocation_is_a_new_record_not_a_deletion(ledger):
    g = grant(ledger)
    ledger.revoke(g.id, app_version="0.1.0")
    records = ledger.all_records()
    assert len(records) == 2                        # append-only
    assert records[0].id == g.id                    # original survives verbatim
    assert records[1].supersedes_id == g.id
    assert not ledger.has_active_consent()


def test_consent_survives_reload(tmp_path):
    path = tmp_path / "c.jsonl"
    grant(ConsentLedger(path))
    assert ConsentLedger(path).has_active_consent()


def test_regrant_after_revoke(ledger):
    g = grant(ledger)
    ledger.revoke(g.id, app_version="0.1.0")
    grant(ledger)
    assert ledger.has_active_consent()
    assert len(ledger.all_records()) == 3


def test_corrupt_line_does_not_hide_valid_records(ledger):
    grant(ledger)
    with ledger.path.open("a") as fh:
        fh.write("{ this is not json\n")
    assert ledger.has_active_consent()


# --- quality gating ---------------------------------------------------------

def test_good_recording_passes():
    a = speech(60.0)
    r = analyse_quality(a, SR, mask_for(a))
    assert r.passed, r.failures
    assert r.speech_seconds > 45


def test_too_short_is_rejected():
    a = speech(10.0)
    r = analyse_quality(a, SR, mask_for(a))
    assert not r.passed
    assert any("speech" in f for f in r.failures)


def test_clipped_recording_is_rejected():
    audio = speech(60.0)
    audio[: int(audio.size * 0.05)] = 1.0
    r = analyse_quality(audio, SR, mask_for(audio))
    assert not r.passed
    assert any("clip" in f.lower() for f in r.failures)


def test_too_quiet_is_rejected():
    a = speech(60.0, amp=0.01, noise=1e-5)
    r = analyse_quality(a, SR, mask_for(a))
    assert not r.passed
    assert any("quiet" in f.lower() for f in r.failures)


def test_mostly_silence_is_rejected():
    audio = speech(60.0)
    audio[int(audio.size * 0.5):] = 0.0
    r = analyse_quality(audio, SR, mask_for(audio))
    assert not r.passed


def test_empty_audio_is_rejected():
    r = analyse_quality(np.zeros(0, dtype=np.float32), SR)
    assert not r.passed


def test_failures_name_the_criterion():
    """AC-09.4: the user must be told which check failed, not just 'bad audio'."""
    a = speech(5.0)
    r = analyse_quality(a, SR, mask_for(a))
    assert r.failures and all(len(f) > 15 for f in r.failures)


# --- profile store ----------------------------------------------------------

def test_enrollment_blocked_without_consent(store):
    with pytest.raises(VaaniError) as e:
        a = speech(60.0)
        store.create(name="me", audio=a, sample_rate=SR,
                     provider_key="xtts", model_id="v2", speech_mask=mask_for(a))
    assert e.value.code is ErrorCode.VOICE_CONSENT_MISSING


def test_enrollment_succeeds_with_consent(store, ledger):
    grant(ledger)
    a = speech(60.0)
    p = store.create(name="me", audio=a, sample_rate=SR,
                     provider_key="xtts", model_id="v2", speech_mask=mask_for(a))
    assert p.is_ready
    assert store.active() is not None


def test_bad_audio_rejected_even_with_consent(store, ledger):
    grant(ledger)
    with pytest.raises(VaaniError) as e:
        a = speech(5.0)
        store.create(name="me", audio=a, sample_rate=SR,
                     provider_key="xtts", model_id="v2", speech_mask=mask_for(a))
    assert e.value.code in (ErrorCode.ENROLLMENT_INSUFFICIENT_SPEECH,
                            ErrorCode.ENROLLMENT_QUALITY_FAILED)


def test_delete_removes_audio_from_disk(store, ledger):
    from pathlib import Path
    grant(ledger)
    a = speech(60.0)
    p = store.create(name="me", audio=a, sample_rate=SR,
                     provider_key="xtts", model_id="v2", speech_mask=mask_for(a))
    audio_path = Path(p.reference_audio_path)
    assert audio_path.exists()
    assert store.delete(p.id)
    assert not audio_path.exists()          # AC-09.6: verified, not assumed
    assert store.active() is None


def test_revoking_consent_deletes_every_profile(store, ledger):
    """AC-09.7: revocation must not leave the clone behind."""
    grant(ledger)
    a = speech(60.0)
    store.create(name="me", audio=a, sample_rate=SR,
                 provider_key="xtts", model_id="v2", speech_mask=mask_for(a))
    assert store.revoke_consent_and_delete_all(app_version="0.1.0") == 1
    assert store.active() is None
    assert not ledger.has_active_consent()


def test_profile_records_the_consent_it_relied_on(store, ledger):
    g = grant(ledger)
    a = speech(60.0)
    p = store.create(name="me", audio=a, sample_rate=SR,
                     provider_key="xtts", model_id="v2", speech_mask=mask_for(a))
    assert p.consent_id == g.id


def test_reference_audio_is_not_world_readable(store, ledger):
    import stat
    from pathlib import Path
    grant(ledger)
    a = speech(60.0)
    p = store.create(name="me", audio=a, sample_rate=SR,
                     provider_key="xtts", model_id="v2", speech_mask=mask_for(a))
    mode = Path(p.reference_audio_path).stat().st_mode
    assert not (mode & stat.S_IROTH), "voice recording must not be world-readable"
