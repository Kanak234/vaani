"""End-to-end pipeline test with stub AI providers.

Uses stubs for STT/translation so the pipeline's control flow and the audio path
are tested independently of model behaviour and model download time. The real
providers are exercised separately in tests/manual.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from vaani.ai.tts.fallback import FallbackSynthesizer
from vaani.context.engine import ContextEngine
from vaani.core.errors import ErrorCode, Severity, VaaniError
from vaani.core.gate import ConfidenceGate, GateConfig
from vaani.core.pipeline import TranslationPipeline
from vaani.core.state_machine import SessionState, SessionStateMachine
from vaani.core.types import SuppressionReason, Transcript, Translation, Utterance

SR = 16000


class StubRecognizer:
    key = "stub_stt"

    def __init__(self, transcript=None, error=None):
        self._t = transcript
        self._error = error
        self.calls = 0
        self.last_prompt = None

    def transcribe(self, audio, sample_rate, *, language_hint=None, context_prompt=None):
        self.calls += 1
        self.last_prompt = context_prompt
        if self._error:
            raise self._error
        return self._t or Transcript(text="मैं कल आऊंगा",
                                     language_distribution={"hi": 1.0},
                                     confidence=0.9)


class StubTranslator:
    key = "stub_mt"

    def __init__(self, translation=None, error=None):
        self._t = translation
        self._error = error
        self.last_context = None

    def translate(self, text, *, source_language, target_language, context=None):
        self.last_context = context
        if self._error:
            raise self._error
        return self._t or Translation(text="I will come tomorrow.",
                                      source_language=source_language,
                                      target_language=target_language,
                                      confidence=0.9)


def build(recognizer=None, translator=None, synthesizer=None, gate_config=None):
    sm = SessionStateMachine()
    sm.transition(SessionState.INITIALIZING)
    sm.transition(SessionState.LISTENING)
    ctx = ContextEngine("test-session")
    p = TranslationPipeline(
        recognizer=recognizer or StubRecognizer(),
        translator=translator or StubTranslator(),
        synthesizer=synthesizer or FallbackSynthesizer(),
        context=ctx, gate=ConfidenceGate(gate_config or GateConfig()),
        state_machine=sm)
    return p, sm, ctx


def utt(seq=1, seconds=1.0):
    return Utterance(session_id="test-session", seq=seq,
                     audio=np.zeros(int(SR * seconds), dtype=np.float32),
                     sample_rate=SR, speech_end_time=time.monotonic(),
                     speech_ms=seconds * 1000)


# --- happy path -------------------------------------------------------------

def test_full_pipeline_produces_audio():
    p, sm, _ = build()
    r = p.process(utt())
    assert not r.suppressed
    assert r.transcript.text == "मैं कल आऊंगा"
    assert r.translation.text == "I will come tomorrow."
    assert r.audio is not None and r.audio.samples.size > 0
    assert sm.state is SessionState.OUTPUTTING


def test_every_stage_is_timed():
    """AC-13.1: per-stage durations must be recorded, measured not estimated."""
    p, _, _ = build()
    r = p.process(utt())
    stages = {t.stage for t in r.timings}
    assert {"stt", "translate", "tts", "context"} <= stages
    assert all(t.duration_ms >= 0 for t in r.timings)
    assert r.total_latency_ms is not None and r.total_latency_ms > 0


def test_fallback_voice_is_flagged():
    """The user must never be represented by a substitute voice silently."""
    r, _, _ = build()
    result = r.process(utt())
    assert result.audio.is_fallback_voice is True


def test_context_accumulates_across_utterances():
    p, sm, ctx = build()
    p.process(utt(seq=1))
    assert ctx.turn_count == 1
    sm.transition(SessionState.LISTENING)
    p.process(utt(seq=2))
    assert ctx.turn_count == 2


def test_context_is_passed_to_the_translator():
    translator = StubTranslator()
    p, sm, _ = build(translator=translator)
    p.process(utt(seq=1))
    sm.transition(SessionState.LISTENING)
    p.process(utt(seq=2))
    assert translator.last_context
    assert translator.last_context[0][1] == "I will come tomorrow."


def test_stt_receives_a_context_prompt_after_the_first_turn():
    rec = StubRecognizer()
    p, sm, _ = build(recognizer=rec)
    p.process(utt(seq=1))
    assert rec.last_prompt is None          # nothing to prime with yet
    sm.transition(SessionState.LISTENING)
    p.process(utt(seq=2))
    assert rec.last_prompt is not None


# --- failure paths: the safety property -------------------------------------

def test_stt_failure_produces_no_audio():
    rec = StubRecognizer(error=VaaniError(code=ErrorCode.STT_FAILURE,
                                          message="boom", severity=Severity.UTTERANCE))
    p, sm, _ = build(recognizer=rec)
    r = p.process(utt())
    assert r.suppressed and r.audio is None
    assert r.error_code == ErrorCode.STT_FAILURE.value
    assert sm.state is SessionState.SUPPRESSED
    assert not sm.can_emit_audio


def test_translation_failure_produces_no_audio():
    tr = StubTranslator(error=VaaniError(code=ErrorCode.TRANSLATION_FAILURE,
                                         message="boom", severity=Severity.UTTERANCE))
    p, sm, _ = build(translator=tr)
    r = p.process(utt())
    assert r.suppressed and r.audio is None
    assert not sm.can_emit_audio


def test_untranslated_devanagari_is_suppressed():
    """Speaking Hindi to an English audience is the silent-failure case."""
    tr = StubTranslator(translation=Translation(
        text="मैं कल आऊंगा", source_language="hi", target_language="en",
        confidence=0.95))
    p, _, _ = build(translator=tr)
    r = p.process(utt())
    assert r.suppressed
    assert r.suppression_reason is SuppressionReason.LOW_TRANSLATION_CONFIDENCE
    assert r.audio is None


def test_low_confidence_transcript_never_reaches_the_translator():
    tr = StubTranslator()
    rec = StubRecognizer(Transcript(text="something", language_distribution={"hi": 1.0},
                                    confidence=0.2))
    p, _, _ = build(recognizer=rec, translator=tr)
    r = p.process(utt())
    assert r.suppressed
    assert tr.last_context is None       # translator was never called
    assert r.audio is None


def test_suppressed_utterance_is_not_added_to_context():
    """A bad translation must not poison every subsequent utterance."""
    tr = StubTranslator(translation=Translation(
        text="मैं कल आऊंगा", source_language="hi", target_language="en",
        confidence=0.95))
    p, _, ctx = build(translator=tr)
    p.process(utt())
    assert ctx.turn_count == 0


def test_one_bad_utterance_does_not_end_the_session():
    """process() must never raise for an utterance-level failure."""
    rec = StubRecognizer(error=VaaniError(code=ErrorCode.STT_FAILURE,
                                          message="boom", severity=Severity.UTTERANCE))
    p, sm, _ = build(recognizer=rec)
    r = p.process(utt())
    assert r.suppressed
    sm.transition(SessionState.LISTENING)
    assert sm.is_active


def test_english_input_bypasses_the_translator():
    """Routing English through NMT risks paraphrasing what the user said."""
    tr = StubTranslator()
    rec = StubRecognizer(Transcript(text="I will come tomorrow",
                                    language_distribution={"en": 1.0},
                                    confidence=0.9))
    p, _, _ = build(recognizer=rec, translator=tr)
    r = p.process(utt())
    assert not r.suppressed
    assert tr.last_context is None                     # never called
    assert r.translation.provider_key == "passthrough"


def test_code_mixed_input_does_use_the_translator():
    tr = StubTranslator()
    rec = StubRecognizer(Transcript(text="Actually मेरा point ये है",
                                    language_distribution={"hi": 0.5, "en": 0.5},
                                    confidence=0.85))
    p, _, _ = build(recognizer=rec, translator=tr)
    r = p.process(utt())
    assert not r.suppressed
    assert r.translation.provider_key == "stub_mt"
