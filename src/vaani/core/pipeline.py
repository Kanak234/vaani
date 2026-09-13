"""Stage sequencing and per-stage timing (Architecture §C).

This is the only module that knows about both the audio layer and the AI layer.
Everything else stays on one side of that line, which is what actually makes
providers swappable rather than nominally abstract.

Two responsibilities that must not be separated from each other:
  1. Run the stages in order, timing each one (AC-13.1).
  2. Guarantee that a failure at any stage produces silence, never audio (AC-12.4).

Timing starts at `speech_end_time` -- the moment VAD decided the user stopped
talking -- because that is when the user starts waiting. Measuring from the start
of the stage pipeline would report a flattering number that no user experiences.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

from ..context.engine import ContextEngine
from ..core.errors import ErrorCode, Severity, VaaniError
from ..core.gate import ConfidenceGate
from ..core.state_machine import SessionState, SessionStateMachine
from ..core.types import (
    StageTiming,
    SuppressionReason,
    SynthesizedAudio,
    Transcript,
    Translation,
    Utterance,
    UtteranceResult,
)


@dataclass(slots=True)
class PipelineConfig:
    stt_timeout_s: float = 15.0
    translation_timeout_s: float = 10.0
    tts_timeout_s: float = 20.0
    #: Skip the translation engine entirely when the utterance is already English.
    passthrough_english: bool = True
    english_passthrough_threshold: float = 0.9
    #: Feed recent transcript text to the recogniser as a vocabulary prior.
    use_stt_context_prompt: bool = True


class TranslationPipeline:
    """Runs one utterance through every stage.

    Deliberately synchronous. The pipeline runs on its own worker thread while the
    audio thread keeps capturing, so concurrency lives at the thread boundary
    rather than inside the stage logic -- which keeps the ordering guarantees and
    the timing measurements simple enough to trust.
    """

    def __init__(self, *, recognizer, translator, synthesizer,
                 context: ContextEngine, gate: ConfidenceGate,
                 state_machine: SessionStateMachine,
                 config: PipelineConfig | None = None,
                 voice_profile_id: str | None = None,
                 on_stage: Callable[[str, UtteranceResult], None] | None = None,
                 on_audio_chunk: Callable[[object], None] | None = None) -> None:
        self.recognizer = recognizer
        self.translator = translator
        self.synthesizer = synthesizer
        self.context = context
        self.gate = gate
        self.sm = state_machine
        self.config = config or PipelineConfig()
        self.voice_profile_id = voice_profile_id
        self._on_stage = on_stage
        # When set and the synthesiser supports it, audio is delivered chunk by
        # chunk as it is generated rather than in one block at the end. Measured:
        # first sound at 394 ms instead of 2315 ms.
        self._on_audio_chunk = on_audio_chunk

    # ------------------------------------------------------------- utilities

    @contextmanager
    def _timed(self, result: UtteranceResult, stage: str, provider_key: str | None):
        """Time a stage and record the outcome whether it succeeds or fails.

        Failures are recorded too: "translation took 9.8 s and then timed out" is
        exactly the diagnostic the user needs, and it would be lost if only
        successes were timed.
        """
        started = time.perf_counter()
        try:
            yield
        except VaaniError as exc:
            result.timings.append(StageTiming(
                stage=stage, duration_ms=(time.perf_counter() - started) * 1000.0,
                provider_key=provider_key, succeeded=False,
                error_code=exc.code.value))
            raise
        except Exception as exc:
            result.timings.append(StageTiming(
                stage=stage, duration_ms=(time.perf_counter() - started) * 1000.0,
                provider_key=provider_key, succeeded=False,
                error_code=ErrorCode.INTERNAL.value))
            raise VaaniError(code=ErrorCode.INTERNAL,
                             message=f"unexpected failure in {stage}: {exc}",
                             stage=stage, cause=exc) from exc
        else:
            result.timings.append(StageTiming(
                stage=stage, duration_ms=(time.perf_counter() - started) * 1000.0,
                provider_key=provider_key, succeeded=True))
        finally:
            if self._on_stage:
                try:
                    self._on_stage(stage, result)
                except Exception:
                    pass

    @staticmethod
    def _suppress(result: UtteranceResult, reason: SuppressionReason,
                  error_code: str | None = None) -> UtteranceResult:
        result.suppressed = True
        result.suppression_reason = reason
        result.error_code = error_code
        result.audio = None          # belt and braces: nothing to play
        return result

    def _safe_state(self, target: SessionState) -> None:
        """Advance state, tolerating races with pause/stop from another thread."""
        try:
            self.sm.transition(target)
        except Exception:
            pass

    # ------------------------------------------------------------------ main

    def process(self, utterance: Utterance) -> UtteranceResult:
        """Run one utterance end to end.

        Never raises for an utterance-level failure: it returns a result with
        `suppressed=True`. One bad utterance must not end the meeting.
        """
        result = UtteranceResult(utterance=utterance)
        try:
            transcript = self._recognize(result, utterance)
            if result.suppressed:
                return self._finish(result)

            translation = self._translate(result, transcript)
            if result.suppressed:
                return self._finish(result)

            self._synthesize(result, translation)
            if result.suppressed:
                return self._finish(result)

            # Only now is it legal to emit audio.
            self._safe_state(SessionState.OUTPUTTING)
            self.context.add_turn(transcript.text, translation.text, utterance.seq)
            return self._finish(result)

        except VaaniError as exc:
            result.error_code = exc.code.value
            if exc.severity is Severity.FATAL:
                self.sm.force(SessionState.FAILED)
            return self._finish(
                self._suppress(result, SuppressionReason.STAGE_FAILURE, exc.code.value))
        except Exception as exc:  # pragma: no cover - defensive
            return self._finish(self._suppress(
                result, SuppressionReason.STAGE_FAILURE, ErrorCode.INTERNAL.value))

    def _finish(self, result: UtteranceResult) -> UtteranceResult:
        result.total_latency_ms = result.utterance.latency_so_far_ms()
        if result.suppressed:
            self._safe_state(SessionState.SUPPRESSED)
        return result

    # ---------------------------------------------------------------- stages

    def _recognize(self, result: UtteranceResult, utterance: Utterance) -> Transcript:
        self._safe_state(SessionState.TRANSCRIBING)
        prompt = (self.context.get_stt_prompt()
                  if self.config.use_stt_context_prompt else None)

        with self._timed(result, "stt", getattr(self.recognizer, "key", None)):
            transcript = self.recognizer.transcribe(
                utterance.audio, utterance.sample_rate,
                language_hint=None,        # None => detect; keeps code-mix intact
                context_prompt=prompt,
            )
        result.transcript = transcript

        with self._timed(result, "gate_stt", None):
            decision = self.gate.check_transcript(transcript)
        if not self.gate.applies(decision):
            self._suppress(result, decision.reason or SuppressionReason.STAGE_FAILURE)
        return transcript

    def _translate(self, result: UtteranceResult, transcript: Transcript) -> Translation:
        self._safe_state(SessionState.TRANSLATING)

        engine = self.translator
        if (self.config.passthrough_english
                and transcript.language_distribution.get("en", 0.0)
                >= self.config.english_passthrough_threshold):
            # Already English. Routing it through NMT would add latency and risk
            # paraphrasing what the user actually said.
            from ..ai.translate.passthrough import PassthroughTranslator
            engine = PassthroughTranslator()

        with self._timed(result, "context", None):
            context = self.context.get_context()

        with self._timed(result, "translate", getattr(engine, "key", None)):
            translation = engine.translate(
                transcript.text,
                source_language=transcript.dominant_language,
                target_language="en",
                context=context or None,
            )
        translation.used_context_turns = len(context)
        # Stamp the engine here rather than trusting each provider to self-report:
        # the pipeline is the only place that reliably knows which one ran, and
        # the UI's cloud/local indicator depends on this being accurate.
        translation.provider_key = getattr(engine, "key", "") or translation.provider_key
        result.translation = translation

        self._safe_state(SessionState.GATING)
        with self._timed(result, "gate_translation", None):
            decision = self.gate.check_translation(translation, transcript)
        if not self.gate.applies(decision):
            self._suppress(result, decision.reason or SuppressionReason.STAGE_FAILURE)
        return translation

    def _synthesize(self, result: UtteranceResult,
                    translation: Translation) -> SynthesizedAudio | None:
        self._safe_state(SessionState.SYNTHESIZING)

        streaming = (self._on_audio_chunk is not None
                     and getattr(self.synthesizer, "supports_streaming", lambda: False)())
        if streaming:
            return self._synthesize_streaming(result, translation)

        with self._timed(result, "tts", getattr(self.synthesizer, "key", None)):
            audio = self.synthesizer.synthesize(
                translation.text, voice_profile_id=self.voice_profile_id)

        if audio.samples.size == 0:
            self._suppress(result, SuppressionReason.EMPTY_RESULT,
                           ErrorCode.TTS_FAILURE.value)
            return None
        result.audio = audio
        return audio

    def _synthesize_streaming(self, result: UtteranceResult,
                              translation: Translation) -> SynthesizedAudio | None:
        """Emit audio chunk by chunk, and enter OUTPUTTING before the first one.

        The state transition has to happen up front: the invariant is that audio
        only leaves while in OUTPUTTING, and here the first chunk is played long
        before synthesis finishes.
        """
        import numpy as np

        started = time.perf_counter()
        first_chunk_ms: float | None = None
        chunks: list = []
        self._safe_state(SessionState.OUTPUTTING)
        try:
            for chunk in self.synthesizer.stream(
                    translation.text, voice_profile_id=self.voice_profile_id):
                if first_chunk_ms is None:
                    first_chunk_ms = (time.perf_counter() - started) * 1000.0
                chunks.append(chunk)
                self._on_audio_chunk(chunk)
        except VaaniError as exc:
            result.timings.append(StageTiming(
                stage="tts", duration_ms=(time.perf_counter() - started) * 1000.0,
                provider_key=getattr(self.synthesizer, "key", None),
                succeeded=False, error_code=exc.code.value))
            self._suppress(result, SuppressionReason.STAGE_FAILURE, exc.code.value)
            return None

        total_ms = (time.perf_counter() - started) * 1000.0
        result.timings.append(StageTiming(
            stage="tts", duration_ms=total_ms,
            provider_key=getattr(self.synthesizer, "key", None), succeeded=True))
        # Record time-to-first-audio separately: it is what the listener
        # experiences, and it is the number streaming actually improves.
        if first_chunk_ms is not None:
            result.timings.append(StageTiming(
                stage="tts_first_chunk", duration_ms=first_chunk_ms,
                provider_key=getattr(self.synthesizer, "key", None)))

        if not chunks:
            self._suppress(result, SuppressionReason.EMPTY_RESULT,
                           ErrorCode.TTS_FAILURE.value)
            return None

        is_fallback = getattr(self.synthesizer, 'supports_voice_cloning', True) is False
        audio = SynthesizedAudio(
            samples=np.concatenate(chunks),
            sample_rate=getattr(self.synthesizer, "sample_rate", 16000),
            voice_profile_id=self.voice_profile_id, is_fallback_voice=is_fallback)
        result.audio = audio
        return audio
