"""Meeting takeover policy for hands-free conversation support.

This module deliberately separates *when Vaani may take the floor* from *what
Vaani says*.  The latter remains the responsibility of the existing local LLM
and voice pipeline.

Takeover is session-scoped and explicit: it must be armed before automatic
handoff can happen, and STOP immediately disables it.  Silence alone is not
treated as a medical or psychological diagnosis.  Instead, the controller uses
observable conversation signals: a pending incoming question, prolonged user
silence, repeated hesitation, or an explicit takeover phrase.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum


class TakeoverState(Enum):
    OFF = "off"
    ARMED = "armed"
    ACTIVE = "active"


class TakeoverAction(Enum):
    WAIT = "wait"
    USER = "user"
    TAKEOVER = "takeover"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class TakeoverConfig:
    """Conservative defaults for automatic handoff.

    `silence_after_question_s` is intentionally configurable because meeting
    latency varies.  A short value makes Vaani proactive; a longer value gives
    the user more time to answer themselves.
    """

    silence_after_question_s: float = 4.0
    min_hesitation_events: int = 2
    max_hesitation_words: int = 4
    explicit_phrases: tuple[str, ...] = (
        "vaani take over",
        "vaani takeover",
        "take over vaani",
        "vani take over",
        "vani takeover",
    )

    def __post_init__(self) -> None:
        if self.silence_after_question_s <= 0:
            raise ValueError("silence_after_question_s must be positive")
        if self.min_hesitation_events < 1:
            raise ValueError("min_hesitation_events must be >= 1")
        if self.max_hesitation_words < 1:
            raise ValueError("max_hesitation_words must be >= 1")


@dataclass(frozen=True, slots=True)
class TakeoverDecision:
    action: TakeoverAction
    reason: str


class MeetingTakeoverController:
    """Thread-safe-by-ownership controller for automatic meeting handoff.

    The session worker should own one instance and feed it meeting/user events
    in timestamp order.  It never generates speech and never changes the audio
    device itself; it only authorizes the next response.
    """

    def __init__(self, *, config: TakeoverConfig | None = None) -> None:
        self.config = config or TakeoverConfig()
        self.state = TakeoverState.OFF
        self.pending_question: str | None = None
        self.hesitation_events = 0
        self.last_user_activity = time.monotonic()
        self.last_question_time: float | None = None

    @property
    def active(self) -> bool:
        return self.state is TakeoverState.ACTIVE

    @property
    def armed(self) -> bool:
        return self.state in (TakeoverState.ARMED, TakeoverState.ACTIVE)

    def arm(self) -> None:
        if self.state is TakeoverState.OFF:
            self.state = TakeoverState.ARMED
        self._reset_turn_state()

    def stop(self) -> TakeoverDecision:
        """Immediate user stop.  This always wins over automatic takeover."""
        self.state = TakeoverState.OFF
        self._reset_turn_state()
        return TakeoverDecision(TakeoverAction.STOP, "user_stop")

    def take_over(self, *, reason: str = "explicit_takeover") -> TakeoverDecision:
        if self.state is TakeoverState.OFF:
            return TakeoverDecision(TakeoverAction.WAIT, "takeover_not_armed")
        self.state = TakeoverState.ACTIVE
        return TakeoverDecision(TakeoverAction.TAKEOVER, reason)

    def observe_question(self, text: str, *, now: float | None = None) -> None:
        """Register a question or conversational turn from the other party."""
        if not self.armed or not text.strip():
            return
        self.pending_question = text.strip()
        self.last_question_time = time.monotonic() if now is None else now
        self.hesitation_events = 0

    def observe_user_speech(
        self,
        text: str,
        *,
        hesitation: bool = False,
        now: float | None = None,
    ) -> TakeoverDecision:
        """Record the user's attempt to answer.

        Any substantive user speech keeps Vaani from stealing the turn.  Short
        hesitant fragments can accumulate as evidence that the user is unable
        to continue, but this is only a conversation signal, not a diagnosis.
        """
        if self.state is TakeoverState.OFF:
            return TakeoverDecision(TakeoverAction.USER, "takeover_off")

        current = time.monotonic() if now is None else now
        self.last_user_activity = current
        text = text.strip()

        if self._is_explicit_takeover(text):
            return self.take_over(reason="explicit_phrase")

        if text:
            if hesitation and len(text.split()) <= self.config.max_hesitation_words:
                self.hesitation_events += 1
            else:
                self.hesitation_events = 0
                # A real answer means the user owns this turn.
                if self.state is TakeoverState.ACTIVE:
                    return TakeoverDecision(TakeoverAction.USER, "user_speaking")
                return TakeoverDecision(TakeoverAction.USER, "user_answering")

        if self.state is TakeoverState.ACTIVE:
            return TakeoverDecision(TakeoverAction.TAKEOVER, "takeover_active")
        return TakeoverDecision(TakeoverAction.WAIT, "waiting_for_user")

    def poll(self, *, now: float | None = None) -> TakeoverDecision:
        """Check whether a pending question has waited long enough for handoff."""
        if self.state is TakeoverState.OFF:
            return TakeoverDecision(TakeoverAction.WAIT, "takeover_off")
        if self.state is TakeoverState.ACTIVE:
            return TakeoverDecision(TakeoverAction.TAKEOVER, "takeover_active")
        if self.pending_question is None or self.last_question_time is None:
            return TakeoverDecision(TakeoverAction.WAIT, "no_pending_question")

        current = time.monotonic() if now is None else now
        silence = current - max(self.last_question_time, self.last_user_activity)
        if (
            silence >= self.config.silence_after_question_s
            and self.hesitation_events >= self.config.min_hesitation_events
        ):
            self.state = TakeoverState.ACTIVE
            return TakeoverDecision(TakeoverAction.TAKEOVER, "hesitation_timeout")
        return TakeoverDecision(TakeoverAction.WAIT, "user_still_has_turn")

    def consume_question(self) -> str | None:
        """Return the question Vaani should answer and clear the pending turn."""
        question = self.pending_question
        self.pending_question = None
        self.last_question_time = None
        self.hesitation_events = 0
        return question

    def _reset_turn_state(self) -> None:
        self.pending_question = None
        self.hesitation_events = 0
        self.last_question_time = None
        self.last_user_activity = time.monotonic()

    def _is_explicit_takeover(self, text: str) -> bool:
        normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return any(phrase in normalized for phrase in self.config.explicit_phrases)
