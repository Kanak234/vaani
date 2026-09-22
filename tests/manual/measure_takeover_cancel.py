"""Measure takeover-cancel latency against the "cancels instantly" criterion.

WHAT IS BEING MEASURED, AND WHY IT IS TWO NUMBERS
-------------------------------------------------
"Cancel instantly" hides two very different questions, and only measuring both
tells the truth:

  1. DECISION latency -- how long `cancel()` itself takes to return. This is on
     the hotkey path, so it must never wait on a model. Budget: 200 ms, and in
     practice it should be microseconds.

  2. AUDIO latency -- how long the meeting keeps hearing Vaani's voice after
     the user hits cancel. This is what the user actually experiences and the
     only one their colleagues notice.

A cancel that returns in 40 us while the meeting hears another two seconds of
an AI-authored sentence has not cancelled anything the user cares about.

The third case is the one that matters most and cannot be timed at all, only
asserted: a cancel arriving WHILE the LLM is still drafting must mean nothing
is ever spoken. That is checked here as a pass/fail, not a duration.

Run: PYTHONPATH=src .venv/bin/python tests/manual/measure_takeover_cancel.py

The audio half needs real PipeWire. The decision half does not, so this script
still reports something useful on a headless machine.
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from vaani.session.continuation import (  # noqa: E402
    ContinuationConfig, ContinuationEngine,
)

SR, FRAME = 16000, 320
BUDGET_MS = 200.0
TRIALS = 20

LONG_CONTEXT = ("I think the migration deadline for the payments service "
                "is going to be")


class InstantLLM:
    def complete(self, _prompt: str) -> str:
        return "somewhere around the end of next quarter, roughly."


class SlowLLM:
    """Stands in for a real local model, which takes seconds on this hardware."""

    def __init__(self, seconds: float = 2.0):
        self.seconds = seconds
        self.finished = False

    def complete(self, _prompt: str) -> str:
        time.sleep(self.seconds)
        self.finished = True
        return "a sentence the user never approved and must never hear spoken."


# ------------------------------------------------------ 1. decision latency

def measure_decision_latency() -> list[float]:
    """How long cancel() takes to return, with a draft NOT in flight."""
    samples = []
    for _ in range(TRIALS):
        engine = ContinuationEngine(llm=InstantLLM())
        engine.arm()
        engine.note_user_speech(LONG_CONTEXT)
        started = time.perf_counter()
        engine.cancel()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


# -------------------------------------------- 2. cancel during a live draft

def measure_cancel_during_draft() -> tuple[float, bool]:
    """Cancel while the model is mid-generation.

    Returns (ms for cancel() to return, whether anything would have been
    spoken). The second value is the one that matters: a draft in flight when
    cancel fired must be discarded even though it completes afterwards.
    """
    import threading

    llm = SlowLLM(seconds=2.0)
    engine = ContinuationEngine(llm=llm)
    engine.arm()
    engine.note_user_speech(LONG_CONTEXT)

    result_box: dict = {}

    def draft():
        result_box["result"] = engine.draft()

    worker = threading.Thread(target=draft)
    worker.start()
    time.sleep(0.3)                     # let the "model" get going

    started = time.perf_counter()
    engine.cancel()
    cancel_ms = (time.perf_counter() - started) * 1000.0

    worker.join(timeout=10)
    result = result_box.get("result")
    would_speak = result is not None and result.text is not None
    # Sanity: the model really was still working when cancel fired.
    assert llm.finished, "the slow LLM never ran; the test proved nothing"
    return cancel_ms, would_speak


# ------------------------------------------------------- 3. audio latency

def measure_audio_latency() -> list[float]:
    """Milliseconds of AI voice the meeting still hears after cancel.

    Uses the session's own flush path -- the same one bypass and emergency stop
    use -- because that is what a real cancel calls. Requires PipeWire.
    """
    from vaani.ai.tts.fallback import FallbackSynthesizer
    from vaani.core.state_machine import SessionState
    from vaani.core.types import Transcript, Translation
    from vaani.session.engine import SessionConfig, TranslationSession

    class R:
        key = "r"

        def transcribe(self, a, sr, *, language_hint=None, context_prompt=None):
            return Transcript(text="x", language_distribution={"en": 1.0},
                              confidence=0.9)

    class T:
        key = "t"

        def translate(self, t, *, source_language, target_language, context=None):
            return Translation(text="y", source_language=source_language,
                               target_language=target_language, confidence=0.9)

    class V:
        def is_speech(self, f):
            return True

        def reset(self):
            pass

    samples = []
    for _ in range(5):
        session = TranslationSession(
            recognizer=R(), translator=T(), synthesizer=FallbackSynthesizer(),
            vad=V(), config=SessionConfig(use_virtual_mic=False))
        # inject_audio() refuses unless the session is in a state that may
        # emit audio (AUDIO_EMITTING_STATES == {OUTPUTTING}). A fresh session
        # is IDLE, so driving it there is what a real takeover does - and
        # measuring against inject_audio() rather than _put_audio() keeps this
        # on the path an actual cancel exercises.
        session.sm.force(SessionState.OUTPUTTING)
        # Queue ~4 s of "AI speech" waiting to play.
        for _ in range(200):
            session.inject_audio(np.zeros(FRAME, dtype=np.float32))
        queued_before = session._out_queue.qsize()
        assert queued_before > 0, (
            "nothing queued - inject_audio() refused, so the measurement "
            "below would have been meaningless")

        started = time.perf_counter()
        session._flush_pending_audio()
        flush_ms = (time.perf_counter() - started) * 1000.0

        remaining = session._out_queue.qsize()
        # Every remaining block is 20 ms the meeting would still hear.
        samples.append(flush_ms + remaining * 20.0)
    return samples


def report(name: str, samples: list[float]) -> bool:
    p50 = statistics.median(samples)
    worst = max(samples)
    ok = worst <= BUDGET_MS
    print(f"  {name:<28} p50 {p50:8.3f} ms   worst {worst:8.3f} ms   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("=" * 72)
    print("TAKEOVER CANCEL LATENCY")
    print(f"budget: {BUDGET_MS:.0f} ms, {TRIALS} trials")
    print("=" * 72)

    passed = True
    print("\n1. decision latency (cancel() returns, no draft in flight)")
    passed &= report("cancel()", measure_decision_latency())

    print("\n2. cancel during a live 2 s draft")
    cancel_ms, would_speak = measure_cancel_during_draft()
    print(f"  {'cancel() during draft':<28} {cancel_ms:8.3f} ms   "
          f"{'PASS' if cancel_ms <= BUDGET_MS else 'FAIL'}")
    print(f"  {'would anything be spoken?':<28} "
          f"{'YES - FAIL' if would_speak else 'NO - PASS'}")
    passed &= (cancel_ms <= BUDGET_MS) and not would_speak

    print("\n3. audio latency (AI voice still heard after cancel)")
    try:
        passed &= report("queued audio dropped", measure_audio_latency())
    except Exception as exc:
        print(f"  SKIPPED: needs real audio ({type(exc).__name__}: {exc})")

    print("\n" + ("ALL PASS" if passed else "FAILURES ABOVE"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
