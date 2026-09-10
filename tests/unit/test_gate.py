"""Confidence gate tests.

The gate is the component that decides whether the user's own voice says something.
These tests encode the failure modes it exists to prevent.
"""
from __future__ import annotations

import pytest

from vaani.core.gate import ConfidenceGate, GateConfig, GatePolicy
from vaani.core.types import SuppressionReason, Transcript, Translation


def tr(text="hello world", conf=0.9, dist=None):
    return Transcript(text=text, language_distribution=dist or {"en": 1.0},
                      confidence=conf)


def tl(text="hello world", conf=0.9, target="en"):
    return Translation(text=text, source_language="hi", target_language=target,
                       confidence=conf)


@pytest.fixture
def gate():
    return ConfidenceGate(GateConfig())


# --- transcript stage -------------------------------------------------------

def test_good_transcript_passes(gate):
    assert gate.check_transcript(tr()).allowed


def test_empty_transcript_suppressed(gate):
    d = gate.check_transcript(tr(text="   "))
    assert d.suppressed and d.reason is SuppressionReason.EMPTY_RESULT


def test_low_confidence_transcript_suppressed(gate):
    d = gate.check_transcript(tr(conf=0.3))
    assert d.suppressed
    assert d.reason is SuppressionReason.LOW_STT_CONFIDENCE
    # The user must be able to see WHY, with numbers.
    assert d.actual_value == pytest.approx(0.3)
    assert d.threshold_value == pytest.approx(0.55)


def test_degenerate_repetition_suppressed(gate):
    """Whisper's classic failure: one token repeated to fill the window."""
    d = gate.check_transcript(tr(text="the the the the the the", conf=0.95))
    assert d.suppressed and d.reason is SuppressionReason.LOW_STT_CONFIDENCE


def test_code_mixed_transcript_is_not_penalised(gate):
    """Hinglish is the target use case -- it must never be gated for being mixed."""
    t = tr(text="Actually मेरा point ये है कि deadline tight है",
           conf=0.8, dist={"hi": 0.55, "en": 0.45})
    assert t.is_code_mixed
    assert gate.check_transcript(t).allowed


def test_confident_unsupported_language_suppressed(gate):
    d = gate.check_transcript(tr(text="bonjour tout le monde", conf=0.9,
                                 dist={"fr": 0.95}))
    assert d.suppressed and d.reason is SuppressionReason.UNSUPPORTED_LANGUAGE


def test_weak_unsupported_signal_is_not_suppressed(gate):
    """A weak third-language signal is noise, not a genuine language switch."""
    assert gate.check_transcript(
        tr(conf=0.9, dist={"fr": 0.4, "en": 0.35, "hi": 0.25})).allowed


# --- translation stage ------------------------------------------------------

def test_good_translation_passes(gate):
    assert gate.check_translation(tl()).allowed


def test_untranslated_devanagari_suppressed(gate):
    """AC-06.3: speaking Hindi to an English audience is a silent failure."""
    d = gate.check_translation(tl(text="मेरा कहना ये है कि project complete"))
    assert d.suppressed
    assert d.reason is SuppressionReason.LOW_TRANSLATION_CONFIDENCE
    assert "devanagari" in d.detail.lower()


def test_empty_translation_suppressed(gate):
    d = gate.check_translation(tl(text=""))
    assert d.suppressed and d.reason is SuppressionReason.EMPTY_RESULT


def test_low_confidence_translation_suppressed(gate):
    d = gate.check_translation(tl(conf=0.2))
    assert d.suppressed and d.reason is SuppressionReason.LOW_TRANSLATION_CONFIDENCE


def test_runaway_expansion_suppressed(gate):
    """A translation 4x its source is a hallucination, not a translation."""
    source = tr(text="हाँ मैं कल आऊंगा और काम पूरा करूंगा")
    d = gate.check_translation(tl(text="yes " * 60), transcript=source)
    assert d.suppressed


def test_short_input_exempt_from_ratio_check(gate):
    """'हाँ' -> 'Yes, absolutely.' is a legitimate expansion."""
    assert gate.check_translation(tl(text="Yes, absolutely."),
                                  transcript=tr(text="हाँ")).allowed


# --- policy -----------------------------------------------------------------

def test_suppress_is_the_default_policy():
    assert GateConfig().policy is GatePolicy.SUPPRESS


def test_speak_anyway_still_records_the_decision():
    """Policy changes whether we block, not whether we analyse."""
    g = ConfidenceGate(GateConfig(policy=GatePolicy.SPEAK_ANYWAY))
    d = g.check_translation(tl(conf=0.1))
    assert d.suppressed                 # detection still fires
    assert g.applies(d) is True         # but audio is allowed through


def test_suppress_policy_blocks_audio():
    g = ConfidenceGate(GateConfig(policy=GatePolicy.SUPPRESS))
    assert g.applies(g.check_translation(tl(conf=0.1))) is False


# --- untranslated passthrough (found by the Hinglish evaluation) -------------

def test_verbatim_echo_is_suppressed(gate):
    """An engine returning its input unchanged has not translated anything."""
    src = tr(text="haan bilkul, main kal call kar lunga",
             dist={"hi": 1.0}, conf=0.9)
    d = gate.check_translation(
        tl(text="Haan bilkul, main kal call kar lunga.", conf=0.86), src)
    assert d.suppressed
    assert "identical" in d.detail or "romanised" in d.detail


def test_romanised_hindi_output_is_suppressed(gate):
    """Measured: NLLB returns romanised Hinglish verbatim at confidence 0.86.

    There is no Devanagari to detect, so the script check cannot see it.
    """
    d = gate.check_translation(tl(text="mera point ye hai ki hum kal tak finish "
                                       "kar sakte hain", conf=0.81))
    assert d.suppressed
    assert d.reason is SuppressionReason.LOW_TRANSLATION_CONFIDENCE


def test_normal_english_is_not_flagged_as_romanised(gate):
    """The markers must not fire on ordinary English."""
    for text in (
        "I will complete the integration by Friday afternoon.",
        "The authentication module needs more testing before deployment.",
        "Yes, I am saying that we can complete this project by next week.",
        "Schedule a meeting with the team on Thursday.",
        "Database migration is having some issue.",
        "I need three days for this job.",
        "No, that's not my responsibility.",
    ):
        assert gate.check_translation(tl(text=text)).allowed, text


def test_single_marker_does_not_trip_the_check(gate):
    """One 'ki' or 'ho' appears in English; two distinct markers are required."""
    assert gate.check_translation(tl(text="The Ki framework handles this well.")).allowed


def test_legitimate_translation_of_short_input_is_not_verbatim(gate):
    src = tr(text="हाँ", dist={"hi": 1.0})
    assert gate.check_translation(tl(text="Yes, absolutely."), src).allowed


def test_english_passthrough_is_not_suppressed_as_verbatim(gate):
    """English in -> identical English out is CORRECT, not a failed translation.

    Without the source!=target guard this suppressed every English utterance.
    """
    text = "I will complete the integration by Friday afternoon."
    src = tr(text=text, dist={"en": 1.0}, conf=0.9)
    d = gate.check_translation(
        Translation(text=text, source_language="en", target_language="en",
                    confidence=0.95), src)
    assert d.allowed, d.detail


def test_verbatim_still_caught_across_a_language_change(gate):
    src = tr(text="haan bilkul, main kal call kar lunga", dist={"hi": 1.0}, conf=0.9)
    d = gate.check_translation(
        Translation(text="haan bilkul, main kal call kar lunga",
                    source_language="hi", target_language="en", confidence=0.86), src)
    assert d.suppressed


# --- found by the first real-voice validation run ---------------------------

def test_unsupported_dominant_language_is_suppressed_at_moderate_confidence(gate):
    """Regression from docs/10-USER-VALIDATION.md.

    A one-word "hello" was transcribed into Arabic script at p=0.6. The old
    threshold (>0.8) let it through, and the translator then raised
    LANGID_UNSUPPORTED as a hard error instead of a clean suppression.
    """
    d = gate.check_transcript(tr(text="هلو هلو", conf=0.56, dist={"ar": 0.6}))
    assert d.suppressed
    assert d.reason is SuppressionReason.UNSUPPORTED_LANGUAGE


def test_supported_languages_still_pass_at_the_same_confidence(gate):
    assert gate.check_transcript(
        tr(text="मुझे तीन दिन चाहिए", conf=0.75, dist={"hi": 1.0})).allowed
    assert gate.check_transcript(
        tr(text="I need three days", conf=0.75, dist={"en": 1.0})).allowed


def test_minority_unsupported_signal_still_does_not_suppress(gate):
    """The original guard must survive: a weak third-language reading is noise."""
    assert gate.check_transcript(
        tr(conf=0.9, dist={"fr": 0.4, "en": 0.35, "hi": 0.25})).allowed
