"""Answer assistant: the approval gate is the whole point.

Every test here exists to make it impossible for a draft to reach audio without a
human explicitly approving it first.
"""
from __future__ import annotations

import pytest

from vaani.ai.assist import (
    AnswerAssistant,
    AssistDraft,
    AssistStyle,
    synthesize_if_approved,
)
from vaani.core.errors import ErrorCode, VaaniError


class StubLLM:
    requires_network = False

    def __init__(self, reply="I can finish this by next week."):
        self.reply = reply
        self.prompts: list[str] = []

    def warmup(self):
        pass

    def _generate(self, prompt, turns, src, tgt):
        self.prompts.append(prompt)
        return self.reply

    def generate_raw(self, prompt, *, model=None):
        self.prompts.append(prompt)
        return self.reply


class StubSynth:
    def __init__(self):
        self.spoken: list[str] = []

    def synthesize(self, text, *, voice_profile_id=None, speed=1.0):
        self.spoken.append(text)
        return object()


def draft_of(text="I can finish by next week."):
    return AssistDraft(id="d1", source_text="kal tak ho jayega",
                       draft_text=text, style=AssistStyle.POLISH)


# --- the approval gate ------------------------------------------------------

def test_a_new_draft_is_not_approved():
    d = draft_of()
    assert d.approved is False
    assert d.audio_allowed is False


def test_unapproved_draft_cannot_be_spoken():
    """The core safety property of this feature."""
    synth = StubSynth()
    with pytest.raises(VaaniError) as e:
        synthesize_if_approved(draft_of(), synth)
    assert e.value.code is ErrorCode.INTERNAL
    assert synth.spoken == []          # nothing reached the synthesiser


def test_approved_draft_is_spoken():
    synth = StubSynth()
    synthesize_if_approved(draft_of().approve(), synth)
    assert synth.spoken == ["I can finish by next week."]


def test_rejected_draft_cannot_be_spoken():
    synth = StubSynth()
    d = draft_of().approve().reject()
    with pytest.raises(VaaniError):
        synthesize_if_approved(d, synth)
    assert synth.spoken == []


def test_empty_draft_cannot_be_spoken_even_if_approved():
    synth = StubSynth()
    d = draft_of("   ")
    d.approve()
    with pytest.raises(VaaniError):
        synthesize_if_approved(d, synth)


def test_user_edit_replaces_the_text_and_is_recorded():
    """If the user rewrites it, those are their words -- and we note the edit."""
    d = draft_of().approve_with_edit("I will finish it by Friday instead.")
    assert d.approved and d.edited
    assert d.final_text == "I will finish it by Friday instead."
    synth = StubSynth()
    synthesize_if_approved(d, synth)
    assert synth.spoken == ["I will finish it by Friday instead."]


# --- drafting ---------------------------------------------------------------

def test_draft_returns_unapproved():
    a = AnswerAssistant(translator=StubLLM())
    assert a.draft("kal tak ho jayega").approved is False


def test_draft_rejects_empty_input():
    a = AnswerAssistant(translator=StubLLM())
    with pytest.raises(VaaniError):
        a.draft("   ")


def test_draft_raises_when_the_model_returns_nothing():
    a = AnswerAssistant(translator=StubLLM(reply=""))
    with pytest.raises(VaaniError) as e:
        a.draft("kuch bhi")
    assert e.value.code is ErrorCode.TRANSLATION_EMPTY


def test_model_chatter_is_stripped_from_drafts():
    a = AnswerAssistant(translator=StubLLM(
        reply="Sure! Here is the translation: I can finish by next week."))
    assert a.draft("kal tak").draft_text == "I can finish by next week."


@pytest.mark.parametrize("style", list(AssistStyle))
def test_every_style_has_an_instruction(style):
    llm = StubLLM()
    AnswerAssistant(translator=llm).draft("kuch", style=style)
    assert len(llm.prompts[0]) > 200


def test_prompt_forbids_inventing_commitments():
    """The assistant must never add a date or a promise the user did not make."""
    llm = StubLLM()
    AnswerAssistant(translator=llm).draft("kal tak")
    prompt = llm.prompts[0].lower()
    assert "never invent" in prompt
    assert "commitments" in prompt


def test_prompt_forbids_answering_on_the_users_behalf():
    llm = StubLLM()
    AnswerAssistant(translator=llm).draft("kya karna chahiye?")
    assert "on the speaker's behalf" in llm.prompts[0].lower()


def test_context_is_included_when_given():
    llm = StubLLM()
    AnswerAssistant(translator=llm).draft(
        "usme baaki hai",
        context=[("backend done", "We finished the backend API.")])
    assert "backend API" in llm.prompts[0]


def test_assistant_is_local_when_the_llm_is_local():
    assert AnswerAssistant(translator=StubLLM()).requires_network is False
