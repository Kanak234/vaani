"""Tests for session/continuation.py (Takeover A).

The risk being defended against is specific: this feature can be wrong about
what the USER THINKS, in the user's own cloned voice, to their colleagues. So
most of these tests are about the engine REFUSING to speak -- when it is not
armed, when it has no real context, when the sentence already landed, when the
other participant is talking, and above all when the user has cancelled.

Every timing test drives an injected clock rather than sleeping.
"""
from __future__ import annotations

import pytest

from vaani.session.continuation import (
    ContinuationConfig, ContinuationEngine, ContinuationTrigger,
)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, s):
        self.now += s


class FakeLLM:
    def __init__(self, reply="and that is why I think we should ship it."):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def engine(llm=None, clock=None, **cfg):
    return ContinuationEngine(llm=llm or FakeLLM(),
                              config=ContinuationConfig(**cfg),
                              clock=clock or FakeClock())


LONG_ENOUGH = "I think the deadline for the migration work is going to be"


# --------------------------------------------------------------- arming

def test_starts_disarmed():
    """Never speaks in the user's voice until they ask for the capability."""
    assert engine().armed is False


def test_a_stall_does_nothing_while_disarmed():
    clock = FakeClock()
    e = engine(clock=clock)
    e.note_user_speech(LONG_ENOUGH)
    clock.advance(60)
    ok, reason = e.should_continue()
    assert ok is False
    assert reason == "not armed"


def test_arming_enables_stall_continuation():
    clock = FakeClock()
    e = engine(clock=clock)
    e.arm()
    e.note_user_speech(LONG_ENOUGH)
    clock.advance(10)
    ok, _ = e.should_continue()
    assert ok is True


# ------------------------------------------------------- refusing to speak

def test_refuses_without_enough_context():
    """Guards the worst case: inventing an opinion out of "so"."""
    clock = FakeClock()
    e = engine(clock=clock)
    e.arm()
    e.note_user_speech("so")
    clock.advance(10)
    ok, reason = e.should_continue()
    assert ok is False
    assert "not enough context" in reason


def test_refuses_when_the_sentence_already_landed():
    """A stall after a full stop is someone who finished talking."""
    clock = FakeClock()
    e = engine(clock=clock)
    e.arm()
    e.note_user_speech("That is my view on the migration timeline.")
    clock.advance(30)
    ok, reason = e.should_continue()
    assert ok is False
    assert "finished" in reason


def test_refuses_while_the_other_participant_is_talking():
    """The user's silence is ordinary turn-taking, not a stall."""
    clock = FakeClock()
    e = engine(clock=clock)
    e.arm()
    e.note_user_speech(LONG_ENOUGH)
    e.note_other_speech("Sorry, can I jump in here?")
    clock.advance(30)
    ok, reason = e.should_continue()
    assert ok is False
    assert "other participant" in reason


def test_refuses_before_the_stall_threshold():
    clock = FakeClock()
    e = engine(clock=clock, stall_after_s=4.0)
    e.arm()
    e.note_user_speech(LONG_ENOUGH)
    clock.advance(1.5)
    ok, reason = e.should_continue()
    assert ok is False
    assert "1.5s of silence" in reason


def test_refuses_before_anything_was_said():
    e = engine()
    e.arm()
    ok, reason = e.should_continue()
    assert ok is False
    assert "nothing said yet" in reason


def test_speaking_again_resets_the_stall_timer():
    """The user picking their own thread back up must cancel the stall."""
    clock = FakeClock()
    e = engine(clock=clock, stall_after_s=4.0)
    e.arm()
    e.note_user_speech(LONG_ENOUGH)
    clock.advance(3.0)
    e.note_user_speech("the deadline is going to be")
    clock.advance(1.0)
    ok, _ = e.should_continue()
    assert ok is False


# ------------------------------------------------- unfinished detection

@pytest.mark.parametrize("text", [
    "I think the deadline is going to be",
    "we should do this because",
    "the main problem is that",
    "matlab yeh hai ki",          # Hinglish trailing connector
])
def test_trailing_connectors_read_as_unfinished(text):
    assert engine().looks_unfinished(text) is True


@pytest.mark.parametrize("text", [
    "That is my final answer.",
    "Does that make sense?",
    "Absolutely not!",
    "We shipped it last Tuesday.",
])
def test_completed_sentences_read_as_finished(text):
    assert engine().looks_unfinished(text) is False


def test_empty_speech_is_not_unfinished():
    assert engine().looks_unfinished("") is False


# ------------------------------------------------------------ cancelling

def test_cancel_prevents_a_draft():
    e = engine()
    e.arm()
    e.note_user_speech(LONG_ENOUGH)
    e.cancel()
    result = e.draft()
    assert result.text is None
    assert result.declined is True


def test_cancel_also_disarms():
    """The user hitting cancel means "stop", not "stop this one"."""
    e = engine()
    e.arm()
    e.cancel()
    assert e.armed is False


def test_a_cancel_during_the_draft_discards_the_result():
    """THE core promise. The LLM call takes seconds; a cancel arriving while
    the model is thinking must still prevent anything being spoken."""
    e = engine()

    class SlowLLM:
        def complete(self, _prompt):
            e.cancel()              # the user hits cancel mid-generation
            return "something I never agreed to say"

    e._llm = SlowLLM()
    e.arm()
    e.note_user_speech(LONG_ENOUGH)
    result = e.draft()
    assert result.text is None
    assert "cancelled" in result.reason


def test_rearming_does_not_let_a_stale_draft_through():
    """A draft in flight when cancel() fired carries the old generation. If
    the user re-arms before it returns, the flag alone would have been cleared
    and the stale text would be spoken."""
    e = engine()

    class SlowLLM:
        def complete(self, _prompt):
            e.cancel()
            e.arm()                 # re-armed before the draft returns
            return "stale text from the cancelled draft"

    e._llm = SlowLLM()
    e.arm()
    e.note_user_speech(LONG_ENOUGH)
    result = e.draft()
    assert result.text is None
    assert "cancelled" in result.reason


def test_cancel_is_safe_when_nothing_is_running():
    e = engine()
    e.cancel()
    e.cancel()


def test_cancel_does_not_call_the_model():
    """The hotkey has a 200 ms budget; it must not wait on anything."""
    class ExplodingLLM:
        def complete(self, _p):
            raise AssertionError("cancel must not reach the model")

    e = engine(llm=ExplodingLLM())
    e.arm()
    e.cancel()


# -------------------------------------------------------------- drafting

def test_a_hotkey_draft_works_without_arming():
    """An explicit key press IS the authorisation; requiring arming as well
    would mean the user pressing the key and nothing happening."""
    e = engine()
    e.note_user_speech(LONG_ENOUGH)
    result = e.draft(ContinuationTrigger.HOTKEY)
    assert result.text is not None


def test_a_stall_draft_requires_arming():
    e = engine()
    e.note_user_speech(LONG_ENOUGH)
    result = e.draft(ContinuationTrigger.STALL)
    assert result.text is None
    assert result.reason == "not armed"


def test_the_prompt_carries_the_conversation_context():
    llm = FakeLLM()
    e = engine(llm=llm)
    e.note_user_speech(LONG_ENOUGH)
    e.note_other_speech("When do you expect it?")
    e.note_user_speech("well the thing is that")
    e.draft()
    prompt = llm.prompts[-1]
    assert LONG_ENOUGH in prompt
    assert "When do you expect it?" in prompt
    assert "Other participant" in prompt


def test_the_prompt_forbids_inventing_facts():
    """The constraint that matters most: being wrong about what the user
    thinks, in their own voice."""
    llm = FakeLLM()
    e = engine(llm=llm)
    e.note_user_speech(LONG_ENOUGH)
    e.draft()
    prompt = llm.prompts[-1]
    assert "Do NOT introduce new facts" in prompt
    assert "FIRST PERSON" in prompt
    assert "ENGLISH only" in prompt


def test_the_model_can_decline():
    """An LLM admitting it cannot continue faithfully is exactly when silence
    is correct."""
    e = engine(llm=FakeLLM(reply="PASS"))
    e.note_user_speech(LONG_ENOUGH)
    result = e.draft()
    assert result.text is None
    assert result.declined is True


def test_an_llm_failure_is_not_fatal():
    class BrokenLLM:
        def complete(self, _p):
            raise RuntimeError("ollama died")

    e = engine(llm=BrokenLLM())
    e.note_user_speech(LONG_ENOUGH)
    result = e.draft()
    assert result.text is None
    assert "LLM failed" in result.reason


# ------------------------------------------------------------- cleaning

def test_wrapping_quotes_are_stripped():
    e = engine(llm=FakeLLM(reply='"and so we should ship it."'))
    e.note_user_speech(LONG_ENOUGH)
    assert e.draft().text == "and so we should ship it."


def test_a_speaker_label_is_stripped():
    """Otherwise the meeting hears the word "Speaker" spoken aloud."""
    e = engine(llm=FakeLLM(reply="Speaker: and so we should ship it."))
    e.note_user_speech(LONG_ENOUGH)
    assert e.draft().text == "and so we should ship it."


def test_the_word_cap_is_enforced_in_code_not_just_the_prompt():
    """A prompt is a request; this is a guarantee - and what is guaranteed is
    how much invented opinion can reach the meeting."""
    e = engine(llm=FakeLLM(reply=" ".join(["word"] * 200)), max_words=10)
    e.note_user_speech(LONG_ENOUGH)
    text = e.draft().text
    assert len(text.split()) <= 11      # 10 words plus the ellipsis marker


def test_an_empty_model_reply_is_declined_not_spoken():
    e = engine(llm=FakeLLM(reply="   "))
    e.note_user_speech(LONG_ENOUGH)
    result = e.draft()
    assert result.text is None
    assert result.declined is True


# ------------------------------------------------------------- bounding

def test_context_does_not_grow_without_bound():
    """A 30-minute meeting must not accumulate every utterance here."""
    e = engine(max_context_turns=4)
    for i in range(500):
        e.note_user_speech(f"sentence number {i} which is quite long and")
    assert len(e._recent) <= 8


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        ContinuationConfig(stall_after_s=0)
    with pytest.raises(ValueError):
        ContinuationConfig(max_words=0)
