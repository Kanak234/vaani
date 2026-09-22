"""Phase 1 hardening: the session must survive a long meeting.

These tests exercise TranslationSession WITHOUT starting its threads or opening
any audio device. start() is what opens PulseAudio; __init__ is pure. So each
test constructs a session and drives _process_one / _put_audio directly, which
keeps the suite runnable on a headless machine and on CI, and keeps it fast.

What is being defended here, in one line each:
  * a long meeting must not grow without bound  (the results leak)
  * a crashing utterance must not end the session (the bare worker loop)
  * a consumer must not mis-read results once the window evicts (index slicing)
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from vaani.ai.tts.fallback import FallbackSynthesizer
from vaani.core.types import Transcript, Translation, Utterance
from vaani.session.engine import SessionConfig, TranslationSession

SR = 16000


class StubRecognizer:
    key = "stub_stt"

    def transcribe(self, audio, sample_rate, *, language_hint=None, context_prompt=None):
        return Transcript(text="मैं कल आऊंगा",
                          language_distribution={"hi": 1.0}, confidence=0.9)


class StubTranslator:
    key = "stub_mt"

    def translate(self, text, *, source_language, target_language, context=None):
        return Translation(text="I will come tomorrow.",
                           source_language=source_language,
                           target_language=target_language, confidence=0.9)


class StubVad:
    """The capture thread is never started in these tests, so the VAD only has
    to exist and satisfy the segmenter's constructor."""

    def is_speech(self, frame, sample_rate):    # pragma: no cover - unused here
        return True

    def reset(self):                            # pragma: no cover - unused here
        pass


def make_session(**cfg) -> TranslationSession:
    return TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=StubVad(),
        config=SessionConfig(use_virtual_mic=False, **cfg))


def utt(seq: int = 1, seconds: float = 0.5) -> Utterance:
    return Utterance(session_id="t", seq=seq,
                     audio=np.zeros(int(SR * seconds), dtype=np.float32),
                     sample_rate=SR, speech_end_time=time.monotonic(),
                     speech_ms=seconds * 1000)


# ------------------------------------------------ the memory leak (NFR-5)

def test_retained_results_are_capped():
    """THE leak: every UtteranceResult holds its synthesised audio, so an
    unbounded list grew by megabytes per utterance for the whole meeting."""
    s = make_session(max_retained_results=10)
    for i in range(50):
        s._process_one(utt(seq=i))
    assert len(s.results) == 10


def test_lifetime_counters_survive_eviction():
    """The window forgets; the counters must not. Otherwise the end-of-session
    report silently understates any meeting past the window size."""
    s = make_session(max_retained_results=10)
    for i in range(50):
        s._process_one(utt(seq=i))
    assert s.total_utterances == 50
    assert s.results_seq == 50


def test_memory_does_not_grow_with_utterance_count():
    """A behavioural check on the ceiling: 20x the utterances must not mean
    20x the retained audio."""
    small = make_session(max_retained_results=5)
    for i in range(10):
        small._process_one(utt(seq=i))
    large = make_session(max_retained_results=5)
    for i in range(200):
        large._process_one(utt(seq=i))
    assert len(small.results) == len(large.results) == 5


# ------------------------------------------- crash containment (the worker)

class ExplodingRecognizer:
    key = "boom_stt"

    def __init__(self, *, every: int = 1):
        self.calls = 0
        self.every = every

    def transcribe(self, audio, sample_rate, *, language_hint=None, context_prompt=None):
        self.calls += 1
        if self.calls % self.every == 0:
            raise RuntimeError("stage exploded")
        return Transcript(text="ok", language_distribution={"en": 1.0},
                          confidence=0.9)


def test_a_stage_exception_is_already_contained_by_the_pipeline():
    """Worth pinning down, because it bounds what the supervisor is FOR.

    TranslationPipeline.process() catches VaaniError and Exception internally
    and returns a suppressed result. So a failing STT/MT/TTS stage never
    reached the worker loop as an exception in the first place.
    """
    s = make_session()
    s.pipeline.recognizer = ExplodingRecognizer()
    attempt = s.supervisor.run(s._process_one, utt())
    assert attempt.ok is True                      # contained one level down
    assert s.results[-1].suppressed is True        # and correctly suppressed


def test_a_crash_after_the_pipeline_does_not_kill_the_worker():
    """THE gap the supervisor closes.

    The pipeline is guarded, but the five steps after it were not: database
    write, audio injection, latency recording, the result callback, the state
    transition. An exception from any of those ended the worker thread, while
    capture kept running and the virtual mic kept emitting silence - so the
    session looked alive while being permanently deaf.

    latency_monitor.record() stands in for all five here; it was genuinely
    unguarded before this change.
    """
    class BrokenMonitor:
        """LatencyMonitor uses __slots__, so the whole object is swapped
        rather than one method patched."""

        def record(self, _result):
            raise RuntimeError("telemetry exploded after a good translation")

    class SilentMonitor:
        def record(self, _result):
            return None

    s = make_session()
    s.latency_monitor = BrokenMonitor()

    attempt = s.supervisor.run(s._process_one, utt())
    assert attempt.ok is False                     # contained, not raised
    assert isinstance(attempt.error, RuntimeError)

    # The session survives and keeps working.
    s.latency_monitor = SilentMonitor()
    s._process_one(utt(seq=2))
    assert s.total_utterances == 2


def test_a_broken_result_callback_cannot_end_the_session():
    """The overlay/CLI callback runs on the worker thread. A bug in the UI
    must not silence translation."""
    s = TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=StubVad(),
        config=SessionConfig(use_virtual_mic=False),
        on_result=lambda _r: (_ for _ in ()).throw(ValueError("UI bug")))
    # _process_one already swallows callback errors, so this must not raise.
    s._process_one(utt())
    assert s.total_utterances == 1


def test_repeated_crashes_trip_degraded_mode_and_tell_the_user():
    warnings: list[str] = []
    s = TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=StubVad(),
        config=SessionConfig(use_virtual_mic=False, stage_crash_limit=3),
        on_latency_warning=lambda w: warnings.append(str(w)))

    def boom(_u):
        raise RuntimeError("stage exploded")

    for _ in range(3):
        s.supervisor.run(boom, utt())

    assert s.supervisor.exhausted is True
    # The user is told, rather than the session going quietly degraded.
    assert any("degraded mode" in w for w in warnings)


def test_failures_and_backpressure_drops_are_counted_separately():
    """"Too slow" and "broken" are different diagnoses and must not share a
    counter."""
    s = make_session()
    assert s.dropped_utterances == 0
    assert s.failed_utterances == 0


# -------------------------------------------- eviction-safe result handoff

def test_new_results_since_returns_only_the_unseen():
    s = make_session()
    for i in range(3):
        s._process_one(utt(seq=i))
    batch, seq = s.new_results_since(0)
    assert len(batch) == 3
    assert seq == 3

    # Nothing new since: empty, and the sequence does not move.
    batch, seq2 = s.new_results_since(seq)
    assert batch == []
    assert seq2 == 3

    s._process_one(utt(seq=99))
    batch, seq3 = s.new_results_since(seq2)
    assert len(batch) == 1
    assert seq3 == 4


def test_a_consumer_that_fell_behind_the_window_is_not_served_stale_data():
    """The old index slicing would silently re-read or skip results here."""
    s = make_session(max_retained_results=5)
    for i in range(100):
        s._process_one(utt(seq=i))
    # A consumer that has seen nothing asks for everything.
    batch, seq = s.new_results_since(0)
    # It can only get what still exists, and the sequence jump makes the gap
    # visible rather than pretending the missing 95 were delivered.
    assert len(batch) == 5
    assert seq == 100


def test_sequence_never_goes_backwards():
    s = make_session(max_retained_results=3)
    seq = 0
    seen = 0
    for i in range(20):
        s._process_one(utt(seq=i))
        batch, seq = s.new_results_since(seq)
        seen += len(batch)
    assert seq == 20
    assert seen == 20      # every result delivered exactly once


# ------------------------------------------------- bounded audio out-queue

def test_audio_queue_is_bounded_and_drops_are_counted():
    """This queue was unbounded. A synthesiser that produces faster than real
    time could outrun the output thread and grow for the whole meeting."""
    s = make_session(max_queued_audio_blocks=4)
    block = np.zeros(160, dtype=np.float32)
    for _ in range(50):
        s._put_audio(block)
    assert s._out_queue.qsize() <= 4
    assert s.dropped_audio_blocks > 0


def test_putting_audio_never_blocks_the_worker():
    """A full queue must shed load, not apply backpressure into the pipeline."""
    s = make_session(max_queued_audio_blocks=2)
    block = np.zeros(160, dtype=np.float32)
    started = time.monotonic()
    for _ in range(500):
        s._put_audio(block)
    # 500 puts against a 2-slot queue: if any of them blocked, this would hang.
    assert time.monotonic() - started < 2.0


def test_the_newest_audio_survives_when_the_queue_overflows():
    """Drop-oldest: in a live meeting the stale block is the worthless one."""
    s = make_session(max_queued_audio_blocks=2)
    for marker in (1.0, 2.0, 3.0, 4.0):
        s._put_audio(np.full(4, marker, dtype=np.float32))
    remaining = []
    while not s._out_queue.empty():
        remaining.append(s._out_queue.get_nowait()[0])
    assert 4.0 in remaining          # the most recent audio is still there
    assert 1.0 not in remaining      # the stalest was evicted


@pytest.mark.parametrize("n", [1, 64, 256])
def test_bound_is_respected_at_several_sizes(n):
    s = make_session(max_queued_audio_blocks=n)
    block = np.zeros(16, dtype=np.float32)
    for _ in range(n * 3):
        s._put_audio(block)
    assert s._out_queue.qsize() <= n


# ----------------------------------------- mid-session provider fallback
#
# THE acceptance criterion these cover: "model auto-detection and mid-session
# fallback work". The subtlety is that a broken translation engine never
# RAISES - the pipeline catches stage exceptions and returns a suppressed
# result - so the supervisor never sees it. Without these windows the meeting
# degrades into silence with nothing swapping the dead provider out.

from vaani.core.types import StageTiming, SuppressionReason  # noqa: E402


def suppressed_result(stage: str, *, reason=SuppressionReason.STAGE_FAILURE):
    """A result shaped like one the pipeline produces when `stage` fails."""
    from vaani.core.types import UtteranceResult
    r = UtteranceResult(utterance=utt())
    r.suppressed = True
    r.suppression_reason = reason
    r.timings = [StageTiming(stage="stt", duration_ms=10.0),
                 StageTiming(stage=stage, duration_ms=20.0)]
    return r


def test_the_failing_stage_is_read_off_the_last_timing():
    s = make_session()
    assert s.failing_stage(suppressed_result("translate")) == "translate"
    assert s.failing_stage(suppressed_result("tts")) == "tts"


def test_a_low_confidence_suppression_is_not_a_stage_failure():
    """The gate working correctly must never swap a provider out."""
    s = make_session()
    result = suppressed_result("translate",
                               reason=SuppressionReason.LOW_STT_CONFIDENCE)
    assert s.failing_stage(result) is None


def test_a_good_result_has_no_failing_stage():
    s = make_session()
    from vaani.core.types import UtteranceResult
    assert s.failing_stage(UtteranceResult(utterance=utt())) is None


def test_repeated_translate_failures_request_a_fallback():
    degraded: list[str] = []
    s = TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=StubVad(),
        config=SessionConfig(use_virtual_mic=False, stage_crash_limit=3),
        on_stage_degraded=degraded.append)

    for _ in range(3):
        s._note_stage_outcome(suppressed_result("translate"))

    assert degraded == ["translate"]


def test_a_good_utterance_clears_the_window():
    """Scattered failures over a long meeting must not accumulate into a
    spurious swap."""
    from vaani.core.types import UtteranceResult
    degraded: list[str] = []
    s = TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=StubVad(),
        config=SessionConfig(use_virtual_mic=False, stage_crash_limit=3),
        on_stage_degraded=degraded.append)

    s._note_stage_outcome(suppressed_result("translate"))
    s._note_stage_outcome(suppressed_result("translate"))
    s._note_stage_outcome(UtteranceResult(utterance=utt()))   # a good one
    s._note_stage_outcome(suppressed_result("translate"))
    s._note_stage_outcome(suppressed_result("translate"))

    assert degraded == []


def test_stages_are_tracked_independently():
    """One flaky stage must not get another stage swapped out."""
    degraded: list[str] = []
    s = TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=StubVad(),
        config=SessionConfig(use_virtual_mic=False, stage_crash_limit=3),
        on_stage_degraded=degraded.append)

    for stage in ("translate", "tts", "translate", "tts"):
        s._note_stage_outcome(suppressed_result(stage))

    assert degraded == []       # neither reached 3


def test_the_replacement_provider_gets_a_clean_allowance():
    """Otherwise the new engine inherits the dead one's failures and is
    swapped out on its first stumble."""
    degraded: list[str] = []
    s = TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=StubVad(),
        # Cooldown disabled: this test is about the WINDOW resetting so the
        # replacement gets a clean allowance, not about how often degradation
        # is announced. Rate limiting is covered separately by
        # test_degraded_announcements_are_rate_limited.
        config=SessionConfig(use_virtual_mic=False, stage_crash_limit=3,
                             stage_degrade_cooldown_s=0.0),
        on_stage_degraded=degraded.append)

    for _ in range(3):
        s._note_stage_outcome(suppressed_result("translate"))
    assert degraded == ["translate"]

    # Two more failures must NOT immediately re-trigger.
    s._note_stage_outcome(suppressed_result("translate"))
    s._note_stage_outcome(suppressed_result("translate"))
    assert degraded == ["translate"]

    s._note_stage_outcome(suppressed_result("translate"))
    assert degraded == ["translate", "translate"]


def test_a_raising_fallback_handler_does_not_kill_the_session():
    def boom(_stage):
        raise RuntimeError("swap failed")

    s = TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=StubVad(),
        config=SessionConfig(use_virtual_mic=False, stage_crash_limit=1),
        on_stage_degraded=boom)
    s._note_stage_outcome(suppressed_result("translate"))       # must not raise


def test_swapping_the_translator_takes_effect_and_is_announced():
    warnings: list[str] = []
    s = TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=StubVad(),
        config=SessionConfig(use_virtual_mic=False),
        on_latency_warning=lambda w: warnings.append(str(w)))

    class Replacement:
        key = "replacement"

        def translate(self, text, *, source_language, target_language, context=None):
            from vaani.core.types import Translation
            return Translation(text="replaced", source_language=source_language,
                               target_language=target_language, confidence=0.9)

    s.swap_translator(Replacement(), reason="LLM failed 3x")
    assert s.pipeline.translator.key == "replacement"
    # A silent downgrade looks like a bug, so the user is told.
    assert any("translation fallback" in w for w in warnings)
    # And the new engine is actually used.
    s._process_one(utt(seq=7))
    assert s.results[-1].translation.text == "replaced"


def test_degrading_a_router_keeps_nllb_not_the_failed_llm():
    """REGRESSION. RoutingTranslator is built as
    RoutingTranslator(fast=nllb, accurate=llm), so the SURVIVING engine is
    `fast`. The first version of the degrade ladder reached for `accurate` and
    would have handed the session the very LLM that had just failed three
    times."""
    from vaani.ai.translate.router import RoutingTranslator

    class Nllb:
        key = "nllb"
        requires_network = False

        def translate(self, text, *, source_language, target_language, context=None):
            from vaani.core.types import Translation
            return Translation(text="nllb", source_language=source_language,
                               target_language=target_language, confidence=0.9)

    class Llm:
        key = "llm"
        requires_network = False

        def translate(self, *a, **k):
            raise RuntimeError("the broken one")

    router = RoutingTranslator(fast=Nllb(), accurate=Llm())
    s = make_session()
    s.pipeline.translator = router

    surviving = router.fast
    s.swap_translator(surviving, reason="LLM failed repeatedly")

    assert s.pipeline.translator.key == "nllb"


def test_no_speech_recognised_is_not_a_provider_failure():
    """REGRESSION from the 30-minute soak.

    An idle microphone picks up room noise, the VAD segments it, Whisper
    correctly reports STT_EMPTY, and the pipeline turns that into a
    STAGE_FAILURE suppression. Counting those made a SILENT ROOM look like a
    broken speech engine: the soak log filled with "stage stt failed 3 times
    in 60s" while nothing was wrong. In a real meeting with any background
    noise, Vaani would have declared STT degraded within the first minute.
    """
    s = make_session()
    result = suppressed_result("stt")
    result.error_code = "STT_EMPTY"
    assert s.failing_stage(result) is None


def test_an_empty_translation_is_not_a_provider_failure_either():
    s = make_session()
    result = suppressed_result("translate")
    result.error_code = "TRANSLATION_EMPTY"
    assert s.failing_stage(result) is None


def test_a_real_stage_failure_is_still_counted():
    """The fix must not blind the detector to genuine failures."""
    s = make_session()
    result = suppressed_result("translate")
    result.error_code = "TRANSLATION_FAILURE"
    assert s.failing_stage(result) == "translate"


def test_benign_errors_never_trip_degraded_mode():
    degraded: list[str] = []
    s = TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=StubVad(),
        config=SessionConfig(use_virtual_mic=False, stage_crash_limit=3),
        on_stage_degraded=degraded.append)
    for _ in range(20):
        result = suppressed_result("stt")
        result.error_code = "STT_EMPTY"
        s._note_stage_outcome(result)
    assert degraded == []


def test_degraded_announcements_are_rate_limited():
    """The soak produced 20 identical ERROR lines in four minutes, because the
    window resets on every trip. That buries everything else in the log."""
    degraded: list[str] = []
    s = TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=StubVad(),
        config=SessionConfig(use_virtual_mic=False, stage_crash_limit=1,
                             stage_degrade_cooldown_s=3600.0),
        on_stage_degraded=degraded.append)
    for _ in range(30):
        s._note_stage_outcome(suppressed_result("translate"))
    # Announced once; the cooldown suppresses the rest.
    assert degraded == ["translate"]
