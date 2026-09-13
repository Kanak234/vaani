"""Runtime bridge between meeting takeover policy and Vaani's local AI/voice stack.

This module deliberately does not touch audio devices. A host session supplies
remote-party transcript events and user speech events; once the takeover policy
authorizes a response, the bridge asks the existing local LLM for a draft and
then uses the existing synthesizer. Device playback remains the session's
responsibility, preserving the existing state-machine boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .takeover import MeetingTakeoverController, TakeoverAction, TakeoverConfig
from ..core.errors import ErrorCode, Severity, VaaniError


@dataclass(frozen=True, slots=True)
class TakeoverResponse:
    question: str
    text: str
    audio: Any
    reason: str


class MeetingTakeoverRuntime:
    """Owns one takeover controller and turns authorized turns into speech."""

    def __init__(self, *, assistant, synthesizer, config: TakeoverConfig | None = None,
                 voice_profile_id: str | None = None) -> None:
        self.controller = MeetingTakeoverController(config=config)
        self.assistant = assistant
        self.synthesizer = synthesizer
        self.voice_profile_id = voice_profile_id
        self.last_question: str | None = None

    @property
    def active(self) -> bool:
        return self.controller.active

    def arm(self) -> None:
        self.controller.arm()

    def stop(self) -> None:
        self.controller.stop()

    def observe_remote_question(self, text: str, *, now: float | None = None) -> None:
        """Feed a detected incoming question into the takeover policy."""
        self.controller.observe_question(text, now=now)
        self.last_question = text.strip() or None

    def observe_user_speech(self, text: str, *, hesitation: bool = False,
                            now: float | None = None) -> TakeoverAction:
        return self.controller.observe_user_speech(text, hesitation=hesitation, now=now).action

    def poll(self, *, now: float | None = None) -> TakeoverAction:
        return self.controller.poll(now=now).action

    def respond_if_authorized(self, *, now: float | None = None,
                              style=None) -> TakeoverResponse | None:
        """Generate and synthesize only after the controller grants the turn."""
        decision = self.controller.poll(now=now)
        if decision.action is not TakeoverAction.TAKEOVER:
            return None

        question = self.controller.consume_question() or self.last_question
        if not question:
            raise VaaniError(
                code=ErrorCode.TRANSLATION_EMPTY,
                message="takeover was authorized but no incoming question is available",
                severity=Severity.UTTERANCE,
                stage="takeover",
            )

        # AnswerAssistant intentionally returns an unapproved draft. For automatic
        # takeover the policy itself is the explicit authorization boundary: the
        # response is generated only after TAKEOVER, never while merely ARMED.
        kwargs = {"style": style} if style is not None else {}
        draft = self.assistant.draft(question, **kwargs)
        draft.approved = True
        audio = self.synthesizer.synthesize(
            draft.final_text, voice_profile_id=self.voice_profile_id)
        self.last_question = None
        return TakeoverResponse(
            question=question,
            text=draft.final_text,
            audio=audio,
            reason=decision.reason,
        )
