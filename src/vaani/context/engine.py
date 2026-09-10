"""Bounded short-term conversation memory (FR-12).

The problem this solves, from the PRD: "उसमें authentication वाला part अभी बाकी है"
cannot be translated correctly in isolation, because "उसमें" refers to something said
earlier. A sentence-at-a-time translator produces "in that, the authentication part
is still left", which is grammatical and useless.

Three properties matter more than cleverness here:

  1. BOUNDED. The window is capped in both turns and characters. An unbounded
     context would grow the translation prompt without limit, which raises latency
     on every utterance and, with a cloud provider, sends progressively more of the
     user's conversation off the machine.
  2. ISOLATED. Context never crosses a session boundary (AC-06.5). Yesterday's
     standup must not leak into today's client call.
  3. DISCARDABLE. Reset takes effect within one utterance (AC-06.4).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Turn:
    source_text: str
    translated_text: str
    seq: int


@dataclass(slots=True)
class ContextConfig:
    #: Turns of history offered to the translator. 0 disables context entirely.
    max_turns: int = 6
    #: Hard character cap across the whole window. Guards latency and, on cloud
    #: providers, egress volume -- regardless of how long individual turns were.
    max_chars: int = 1200
    #: Turns longer than this are stored truncated; a rambling turn should not
    #: consume the entire window on its own.
    max_turn_chars: int = 400


class ContextEngine:
    """Per-session rolling window of recent turns."""

    def __init__(self, session_id: str, config: ContextConfig | None = None) -> None:
        self.session_id = session_id
        self.config = config or ContextConfig()
        self._turns: deque[Turn] = deque(maxlen=max(1, self.config.max_turns))

    def add_turn(self, source_text: str, translated_text: str, seq: int) -> None:
        """Record a completed utterance.

        Only successful translations are added. A suppressed or failed utterance is
        deliberately NOT recorded: feeding a known-bad translation back as context
        would let one error corrupt every subsequent utterance.
        """
        if self.config.max_turns <= 0:
            return
        if not source_text.strip() or not translated_text.strip():
            return
        self._turns.append(Turn(
            source_text=_truncate(source_text, self.config.max_turn_chars),
            translated_text=_truncate(translated_text, self.config.max_turn_chars),
            seq=seq,
        ))

    def get_context(self) -> list[tuple[str, str]]:
        """Recent (source, translation) pairs, oldest first, within the char cap.

        Trimming drops the OLDEST turns first: recent turns are far more likely to
        contain the antecedent of a pronoun in the current utterance.
        """
        if self.config.max_turns <= 0 or not self._turns:
            return []

        selected: list[Turn] = []
        used = 0
        for turn in reversed(self._turns):
            cost = len(turn.source_text) + len(turn.translated_text)
            if used + cost > self.config.max_chars and selected:
                break
            selected.append(turn)
            used += cost
        selected.reverse()
        return [(t.source_text, t.translated_text) for t in selected]

    def get_stt_prompt(self) -> str | None:
        """Recent SOURCE text, to bias the recogniser's vocabulary.

        Whisper's `initial_prompt` conditions decoding on prior text, which helps
        with recurring proper nouns and domain jargon ("Kubernetes", a colleague's
        name) that it would otherwise mangle differently each time.
        """
        if not self._turns:
            return None
        recent = [t.source_text for t in list(self._turns)[-3:]]
        prompt = " ".join(recent).strip()
        return _truncate(prompt, 400) or None

    def reset(self) -> None:
        """Discard all context (AC-06.4)."""
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def __len__(self) -> int:
        return len(self._turns)


def _truncate(text: str, limit: int) -> str:
    """Trim to `limit`, preferring a word boundary so context stays readable."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.6 else cut).rstrip()
