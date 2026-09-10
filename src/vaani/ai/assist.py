"""Answer assistance: draft a reply the user can approve before it is spoken.

WHAT THIS IS FOR
The user knows what they want to say but cannot find the English phrasing fast
enough in a live meeting. They speak a rough, partial or Hinglish thought; this
drafts a fluent English version; they approve it; it is spoken in their voice.

WHY APPROVAL IS REQUIRED, AND NOT OPTIONAL

There is a version of this feature where the AI simply answers for the user,
autonomously, in their cloned voice. That version is not built here, and the
reason is not squeamishness -- it is that it breaks the one property this whole
product depends on.

Everything else in Vaani translates something the user actually said. The words
are theirs; only the language changes. Participants hear the user's voice saying
the user's meaning, which is what makes it honest.

An AI generating original content and speaking it in the user's voice inverts
that. The other people in the meeting have no way to know they are hearing a
machine's opinion rather than a person's, they never agreed to it, and the user
becomes accountable for commitments -- dates, numbers, agreements -- that they
did not make and may not even have heard. A single "yes, we can ship by Friday"
generated on their behalf is a real professional problem.

So the draft is always shown and always requires an explicit approval before any
audio is produced. The user stays the author. That keeps the assistance genuinely
useful for the actual problem -- "I cannot find the words fast enough" -- without
making the tool speak for them.

`AssistDraft.approved` starts False and only `approve()` sets it. `synthesize`
callers must check it; `PendingDraft.audio_allowed` exists so that check is a
single obvious call rather than a convention people forget.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from ..core.errors import ErrorCode, Severity, VaaniError


class AssistStyle(Enum):
    #: Say the same thing, just fluently. The safest and the default.
    POLISH = "polish"
    #: Expand a terse thought into a complete sentence.
    EXPAND = "expand"
    #: Make it more formal for a client-facing call.
    FORMAL = "formal"


_STYLE_INSTRUCTION = {
    AssistStyle.POLISH: (
        "Rewrite the speaker's words as fluent, natural spoken English. Keep the "
        "SAME meaning and the same level of commitment. Do not add information, "
        "opinions, agreement or promises that are not already there."),
    AssistStyle.EXPAND: (
        "The speaker gave a terse or incomplete thought. Turn it into one or two "
        "complete, natural English sentences. Stay strictly within what they "
        "implied -- do not invent facts, dates, numbers or commitments."),
    AssistStyle.FORMAL: (
        "Rewrite the speaker's words in polite, professional English suitable for "
        "a client call. Keep the same meaning and the same commitments."),
}

_SYSTEM = """\
You help a person phrase what THEY want to say in an English-language meeting. \
You are not a participant and you have no opinions of your own.

Absolute rules:
1. Output ONLY the suggested phrasing. No preamble, no options, no commentary.
2. NEVER invent facts, dates, numbers, names, agreements or commitments. If the \
speaker did not say it, it does not appear.
3. NEVER answer a question on the speaker's behalf. You rephrase what they said; \
you do not decide what they think.
4. Preserve negation and hedging exactly. "maybe" must not become "yes".
5. Keep it short and speakable. This will be read aloud, not printed.\
"""


@dataclass(slots=True)
class AssistDraft:
    """A suggestion. Not speech until a human approves it."""

    id: str
    source_text: str
    draft_text: str
    style: AssistStyle
    created_at: float = field(default_factory=time.time)
    approved: bool = False
    edited: bool = False

    @property
    def audio_allowed(self) -> bool:
        """The single check every synthesis path must make."""
        return self.approved and bool(self.final_text.strip())

    @property
    def final_text(self) -> str:
        return self.draft_text

    def approve(self) -> AssistDraft:
        self.approved = True
        return self

    def approve_with_edit(self, text: str) -> AssistDraft:
        """The user rewrote it. Their words, even more clearly."""
        self.draft_text = text
        self.edited = True
        self.approved = True
        return self

    def reject(self) -> AssistDraft:
        self.approved = False
        return self


class AnswerAssistant:
    """Drafts phrasings with the local LLM. Never speaks on its own."""

    def __init__(self, *, translator=None, model: str = "qwen3-coder:latest",
                 max_context_turns: int = 6) -> None:
        # Reuses OllamaTranslator's transport: same loopback host, same
        # local-only guarantee, no second HTTP client to maintain.
        if translator is None:
            from .translate.ollama import OllamaTranslator
            translator = OllamaTranslator(model=model)
        self._llm = translator
        self.max_context_turns = max_context_turns
        self.requires_network = getattr(translator, "requires_network", False)

    def warmup(self) -> None:
        self._llm.warmup()

    def draft(self, source_text: str, *, style: AssistStyle = AssistStyle.POLISH,
              context: list[tuple[str, str]] | None = None) -> AssistDraft:
        """Produce a suggestion. Returns unapproved — nothing is spoken yet."""
        source_text = source_text.strip()
        if not source_text:
            raise VaaniError(code=ErrorCode.TRANSLATION_EMPTY,
                             message="nothing to rephrase",
                             severity=Severity.UTTERANCE)

        turns = (context or [])[-self.max_context_turns:]
        parts = [_SYSTEM, "", _STYLE_INSTRUCTION[style], ""]
        if turns:
            parts.append("Recent conversation, for context only:")
            for src, dst in turns:
                parts.append(f"  {dst or src}")
            parts.append("")
        parts += [f"The speaker said: {source_text}",
                  "Suggested English phrasing:"]

        raw = self._llm._generate("\n".join(parts), [], "hi", "en")
        from .translate.ollama import _strip_model_chatter
        text = _strip_model_chatter(raw)
        if not text:
            raise VaaniError(code=ErrorCode.TRANSLATION_EMPTY,
                             message="no suggestion produced",
                             severity=Severity.UTTERANCE)

        return AssistDraft(id=uuid.uuid4().hex[:8], source_text=source_text,
                           draft_text=text, style=style)


def synthesize_if_approved(draft: AssistDraft, synthesizer, *,
                           voice_profile_id: str | None = None):
    """The only supported way to turn a draft into audio.

    Raises rather than silently returning nothing, so a caller that forgets the
    approval step fails loudly in testing instead of quietly in a meeting.
    """
    if not draft.audio_allowed:
        raise VaaniError(
            code=ErrorCode.INTERNAL,
            message="refusing to speak a draft the user has not approved",
            severity=Severity.UTTERANCE,
            detail={"draft_id": draft.id},
        )
    return synthesizer.synthesize(draft.final_text,
                                  voice_profile_id=voice_profile_id)
