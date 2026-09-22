"""Takeover A: finish the user's own sentence when they stall.

HOW THIS DIFFERS FROM session/takeover.py
-----------------------------------------
``takeover.py`` is Takeover B: the OTHER participant asked something and the
user froze, so Vaani answers on their behalf. This module is Takeover A: the
user was mid-point, ran out of words, and wants their own thought carried to
the end. The distinction is not cosmetic -- it changes the prompt, the voice,
and above all the risk.

B can be wrong about a fact. A can be wrong about what the USER THINKS, in the
user's own cloned voice, to their colleagues. That is a much worse failure, and
every default here is set accordingly:

  * it is OFF until armed, per session, and arming never persists;
  * it needs real context -- it will not invent a position from one word;
  * it is short by design, because a long invented monologue is harder to
    disown than a single sentence;
  * it is cancellable at any instant, including mid-draft;
  * it never speaks without being visibly tagged as AI.

WHY A STALL IS NOT A DIAGNOSIS
------------------------------
Silence means many things: thinking, being interrupted, muting to cough,
someone else talking. So a stall alone is necessary but NOT sufficient -- the
engine also requires that the user was demonstrably mid-thought (unfinished
trailing speech) and that the session is armed. The alternative, speaking
whenever the user pauses, would make Vaani interrupt its own user, which is the
single most embarrassing thing this product could do in a meeting.

WHERE THE DATA FLOWS NEXT
-------------------------
:meth:`ContinuationEngine.draft` returns text only. The caller synthesises it
with the user's cloned voice and injects it through
``TranslationSession.inject_audio``. The AI-speaking indicator is raised BEFORE
synthesis starts and lowered only after playback ends, so the badge can never
lag behind the audio.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class ContinuationTrigger(Enum):
    """Why a continuation was requested. Recorded so the overlay and the logs
    can distinguish "I asked for this" from "Vaani decided"."""
    HOTKEY = "hotkey"
    STALL = "stall"


@dataclass(frozen=True, slots=True)
class ContinuationConfig:
    #: How long the user must be silent before a stall counts. Deliberately
    #: longer than an ordinary between-sentence pause: at 1-2 s Vaani would
    #: talk over someone who was simply breathing.
    stall_after_s: float = 4.0

    #: Minimum characters of recent user speech before a continuation is
    #: allowed. Guards the worst case -- inventing an opinion out of "so".
    min_context_chars: int = 40

    #: How many recent turns are shown to the LLM.
    max_context_turns: int = 6

    #: Cap on what gets spoken. Short on purpose: a single sentence that lands
    #: wrong is recoverable in a meeting; a paragraph is not.
    max_words: int = 40

    #: Trailing words that suggest an unfinished thought. A stall right after
    #: one of these is a genuine "lost the thread"; a stall after a full stop
    #: is just someone who finished talking and must NOT be continued.
    #: WHY A WORD LIST rather than a parser: the failure mode of a MISSING word
    #: is that Vaani stays quiet, which is the safe direction. The failure mode
    #: of over-matching is Vaani talking over its own user. So this list is
    #: allowed to be incomplete and must never be aggressive.
    #: It started without the auxiliaries and immediately missed the most
    #: ordinary English stall there is - "the deadline is going to be" - so
    #: auxiliaries, modals and prepositions are included now.
    unfinished_endings: tuple[str, ...] = (
        # conjunctions and subordinators
        "and", "but", "so", "because", "if", "when", "while", "although",
        "that", "which", "who", "than",
        # articles and determiners
        "the", "a", "an", "this", "these", "those", "some", "any",
        # pronouns and copulas
        "we", "i", "it", "they", "you", "he", "she",
        "is", "are", "was", "were", "be", "been", "being", "am",
        # auxiliaries and modals
        "have", "has", "had", "do", "does", "did",
        "will", "would", "can", "could", "should", "might", "must", "may",
        # prepositions and particles
        "to", "of", "in", "on", "at", "for", "with", "about", "from", "by",
        "into", "over", "under", "as", "like",
        # Hinglish connectors - the register this product exists for
        "matlab", "toh", "aur", "ki", "jo", "kyunki", "lekin", "phir",
        "agar", "jab", "hai", "hain", "tha", "the", "karna", "ke", "ka", "se",
    )

    def __post_init__(self) -> None:
        if self.stall_after_s <= 0:
            raise ValueError("stall_after_s must be positive")
        if self.max_words < 1:
            raise ValueError("max_words must be >= 1")


#: The prompt. Every clause is a defence against a specific observed failure.
_SYSTEM = (
    "You finish the speaker's own sentence in a live meeting.\n"
    # First person, because the audio will be their cloned voice. Third person
    # here produces "he was saying that...", which is incoherent coming out of
    # the user's own mouth.
    "Continue in FIRST PERSON as the speaker, never about them.\n"
    # The single most important constraint. See the module docstring: being
    # wrong about what the user thinks is the failure that matters.
    "Do NOT introduce new facts, numbers, dates, names or commitments. "
    "Only complete the thought already begun.\n"
    "Output ENGLISH only. No Devanagari.\n"
    # LLMs otherwise emit "Sure! Here's how I'd finish that:" which would be
    # spoken aloud into the meeting.
    "Output ONLY the continuation text. No preamble, no quotes, no labels.\n"
    "Be brief: one short sentence. If you cannot continue the thought "
    "faithfully, output exactly: PASS"
)

#: The model's way of declining. Treated as "say nothing" rather than spoken,
#: because an LLM admitting it cannot continue faithfully is exactly the case
#: where silence is correct.
PASS_SENTINEL = "PASS"


@dataclass
class ContinuationResult:
    text: str | None
    trigger: ContinuationTrigger
    reason: str
    #: True when the engine declined on purpose (no context, model passed,
    #: cancelled). Distinguished from an error so the caller can stay quiet
    #: instead of showing a failure.
    declined: bool = False
    context_turns: int = 0


class OllamaCompletion:
    """Adapts OllamaTranslator to the tiny `complete(prompt) -> str` interface
    this module needs.

    WHY AN ADAPTER rather than calling the translator directly: the engine must
    stay testable with a five-line fake, and it must not inherit the
    translation system prompt. OllamaTranslator.generate_raw() exists for
    exactly this - a raw prompt with no translation framing wrapped around it.

    WHY NOT duck-type on generate_raw inside the engine: an explicit adapter
    makes the dependency visible at the call site instead of hiding a
    getattr() fallback in the hot path.
    """

    def __init__(self, translator) -> None:
        self._translator = translator

    def complete(self, prompt: str) -> str:
        return self._translator.generate_raw(prompt)


class ContinuationEngine:
    """Decides when the user has stalled, and drafts the finish of their point.

    Owns no audio and no threads of its own. The caller polls :meth:`poll` on
    whatever loop it already has, which keeps this class deterministic and
    testable with an injected clock.
    """

    def __init__(self, *, llm, config: ContinuationConfig | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config or ContinuationConfig()
        self._llm = llm
        self._clock = clock
        self._armed = False
        self._recent: list[tuple[str, str]] = []
        self._last_speech_at: float | None = None
        self._last_text: str = ""
        # Set by cancel(). Checked immediately before the LLM call and again
        # immediately after, so a cancel arriving DURING a multi-second draft
        # still prevents anything being spoken. A lock is not enough here: the
        # point is to invalidate work already in flight, not to serialise it.
        self._cancelled = threading.Event()
        #: Bumped on every cancel. A draft that started before a cancel carries
        #: the old value and is discarded even if it finishes afterwards --
        #: the flag alone would be cleared by the next arm and let a stale
        #: draft through.
        self._generation = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------- arming

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self) -> None:
        """Allow continuations. Never persisted across sessions on purpose."""
        self._armed = True
        self._cancelled.clear()

    def disarm(self) -> None:
        self._armed = False

    def cancel(self) -> None:
        """Stop immediately and refuse anything already in flight.

        Must be safe to call from any thread, at any time, including when
        nothing is running. Cheap by design: the hotkey path has a 200 ms
        budget and must not wait on a model.
        """
        with self._lock:
            self._generation += 1
        self._cancelled.set()
        self._armed = False

    # ------------------------------------------------- observing the user

    def note_user_speech(self, text: str, *, now: float | None = None) -> None:
        """Record what the user just said. Resets the stall timer.

        WHERE IT COMES FROM: each finished UtteranceResult's transcript, so the
        context is what the user actually said in their own words -- not the
        English translation, which would make the LLM continue a paraphrase.
        """
        text = (text or "").strip()
        if not text:
            return
        self._last_speech_at = self._clock() if now is None else now
        self._last_text = text
        self._recent.append(("user", text))
        # Bounded: a long meeting must not grow this list forever, and the LLM
        # only ever sees max_context_turns anyway.
        if len(self._recent) > self.config.max_context_turns * 2:
            del self._recent[:-self.config.max_context_turns]

    def note_other_speech(self, text: str) -> None:
        """Remote participant speech, for context only.

        Also treated as a reason NOT to continue: if someone else is talking,
        the user's silence is ordinary turn-taking, not a stall.
        """
        text = (text or "").strip()
        if not text:
            return
        self._recent.append(("other", text))
        self._last_speech_at = self._clock()

    # ------------------------------------------------------- the decision

    def looks_unfinished(self, text: str | None = None) -> bool:
        """Whether the user's last utterance trails off mid-thought.

        This is what separates "lost the thread" from "finished talking". A
        stall after "...and so the deadline is" wants finishing; a stall after
        "That's my view." emphatically does not.
        """
        candidate = (text if text is not None else self._last_text).strip()
        if not candidate:
            return False
        # Terminal punctuation means the speaker landed the sentence.
        if candidate[-1] in ".!?":
            return False
        last_word = candidate.split()[-1].strip(",;:").lower()
        return last_word in self.config.unfinished_endings

    def context_chars(self) -> int:
        return sum(len(t) for role, t in self._recent if role == "user")

    def should_continue(self, *, now: float | None = None) -> tuple[bool, str]:
        """Is a stall-triggered continuation warranted right now?

        Returns (yes, reason) so the reason can be logged and shown rather than
        the user wondering why Vaani spoke.
        """
        if not self._armed:
            return False, "not armed"
        if self._cancelled.is_set():
            return False, "cancelled"
        if self._last_speech_at is None:
            return False, "nothing said yet"
        if self._recent and self._recent[-1][0] == "other":
            # Ordinary turn-taking, not a stall.
            return False, "the other participant spoke last"
        if self.context_chars() < self.config.min_context_chars:
            return False, "not enough context to continue a point"
        if not self.looks_unfinished():
            return False, "the last sentence sounded finished"
        elapsed = (self._clock() if now is None else now) - self._last_speech_at
        if elapsed < self.config.stall_after_s:
            return False, f"only {elapsed:.1f}s of silence"
        return True, f"stalled mid-sentence for {elapsed:.1f}s"

    # ---------------------------------------------------------- drafting

    def draft(self, trigger: ContinuationTrigger = ContinuationTrigger.HOTKEY,
              ) -> ContinuationResult:
        """Ask the LLM to finish the user's point.

        Cancellation is checked three times -- before the call, after it, and
        via the generation counter -- because the call itself can take seconds
        and the whole promise of this feature is that the user can stop it
        instantly. A draft that was in flight when cancel() fired is discarded
        rather than spoken.
        """
        if trigger is ContinuationTrigger.STALL and not self._armed:
            return ContinuationResult(None, trigger, "not armed", declined=True)
        if self._cancelled.is_set():
            return ContinuationResult(None, trigger, "cancelled", declined=True)
        if self.context_chars() < self.config.min_context_chars:
            return ContinuationResult(
                None, trigger, "not enough context to continue a point",
                declined=True)

        with self._lock:
            generation = self._generation

        turns = self._recent[-self.config.max_context_turns:]
        prompt = self._build_prompt(turns)

        try:
            raw = self._llm.complete(prompt)
        except Exception as exc:
            # Never fatal: failing to finish a sentence must not end a meeting.
            return ContinuationResult(None, trigger, f"LLM failed: {exc}",
                                      declined=False, context_turns=len(turns))

        # Cancelled while the model was thinking: discard, do not speak.
        if self._cancelled.is_set():
            return ContinuationResult(None, trigger, "cancelled during draft",
                                      declined=True, context_turns=len(turns))
        with self._lock:
            if generation != self._generation:
                return ContinuationResult(None, trigger,
                                          "cancelled during draft",
                                          declined=True,
                                          context_turns=len(turns))

        text = self._clean(raw)
        if not text or text.upper().startswith(PASS_SENTINEL):
            return ContinuationResult(None, trigger,
                                      "the model declined to continue",
                                      declined=True, context_turns=len(turns))
        return ContinuationResult(text, trigger, "continued the user's point",
                                  context_turns=len(turns))

    def _build_prompt(self, turns: list[tuple[str, str]]) -> str:
        lines = [_SYSTEM, ""]
        if turns:
            lines.append("Conversation so far:")
            for role, text in turns:
                speaker = "Speaker" if role == "user" else "Other participant"
                lines.append(f"{speaker}: {text}")
            lines.append("")
        lines.append("Finish the Speaker's unfinished thought, in their voice:")
        return "\n".join(lines)

    def _clean(self, raw: str) -> str:
        """Strip the wrappers LLMs add, and enforce the word cap.

        The cap is enforced HERE rather than trusted to the prompt, because a
        prompt is a request and this is a guarantee -- and the thing being
        guaranteed is how much invented opinion can reach the meeting.
        """
        text = (raw or "").strip()
        if not text:
            return ""
        # Common wrappers: surrounding quotes, a "Speaker:" label.
        if text[0] in "\"'" and text[-1] == text[0] and len(text) > 1:
            text = text[1:-1].strip()
        for label in ("Speaker:", "Continuation:", "Assistant:"):
            if text.startswith(label):
                text = text[len(label):].strip()
        words = text.split()
        if len(words) > self.config.max_words:
            text = " ".join(words[:self.config.max_words]).rstrip(",;:") + "..."
        return text
