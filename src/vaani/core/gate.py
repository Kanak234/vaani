"""The confidence gate (FR-15, AC-12.3).

This is the most important safety component in the product.

The primary persona's stated failure mode is not "the translation was imperfect" --
it is "the tool confidently said something I did not say, in my own voice, to my
colleagues, and I had to publicly correct it." Silence is recoverable; the user can
simply repeat themselves. A confident mistranslation in the user's own voice is not.

So the default policy is SUPPRESS, and every suppression is recorded with the
threshold and the actual value, so the user can see exactly why it stayed quiet.
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass

from ..core.types import SuppressionReason, Transcript, Translation


class GatePolicy(enum.Enum):
    #: Emit nothing when confidence is low. Default.
    SUPPRESS = "suppress"
    #: Speak anyway. For users who prefer coverage over precision.
    SPEAK_ANYWAY = "speak_anyway"
    #: Surface it and let the user decide. Costs a round trip of human latency.
    ASK = "ask"


@dataclass(slots=True)
class GateConfig:
    stt_confidence_threshold: float = 0.55
    translation_confidence_threshold: float = 0.50
    policy: GatePolicy = GatePolicy.SUPPRESS
    #: Languages we can actually translate. Anything else is suppressed rather
    #: than guessed at.
    supported_languages: frozenset[str] = frozenset({"hi", "en"})
    #: Below this, a "translation" is almost certainly a decode artefact.
    min_output_chars: int = 2


@dataclass(slots=True)
class GateDecision:
    allowed: bool
    reason: SuppressionReason | None = None
    threshold_value: float | None = None
    actual_value: float | None = None
    detail: str = ""

    @property
    def suppressed(self) -> bool:
        return not self.allowed


#: Devanagari left in the "English" output means translation silently failed.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

#: Common romanised-Hindi function words. Romanised Hinglish ("haan bilkul, main
#: kal call kar lunga") is all-Latin, so the Devanagari check cannot see it, and
#: NLLB frequently returns it VERBATIM with high confidence. Measured: that exact
#: sentence came back unchanged at confidence 0.86 and passed the gate.
#: Speaking untranslated Hindi to an English audience is precisely the silent
#: failure this gate exists to prevent, so function words are checked directly.
#: The list deliberately EXCLUDES Hindi function words that collide with common
#: English ones -- "the", "hi", "ho", "main", "par", "se", "ka", "ke", "ki", "to",
#: "ya". An early version included "the" (Hindi: were) and suppressed the perfectly
#: good English sentence "The Ki framework handles this well." False-suppressing
#: correct English is a worse failure than missing some romanised Hindi, because
#: the user loses an utterance they said correctly.
_ROMANISED_HINDI = re.compile(
    r"\b(?:hai|hain|hoga|hogi|tha|thi|raha|rahe|rahi|gaya|gayi|"
    r"karo|karna|karke|karta|karti|kiya|karenge|karunga|"
    r"kyun|kyunki|nahi|nahin|haan|bilkul|matlab|lekin|magar|agar|phir|"
    r"mera|meri|mere|tera|teri|uska|uski|unka|unke|hamara|hamari|"
    r"mujhe|tumhe|aapko|humein|yeh|woh|jaise|waise|"
    r"thoda|zyada|accha|theek|sirf|bahut|kuch|abhi|"
    r"wala|wale|wali|sakta|sakte|sakti|chahiye|liye|"
    r"lunga|lungi|denge|hona|hone|kaam|baat|log)\b",
    re.IGNORECASE,
)
#: Whisper's classic failure: the same token repeated to fill the window.
_DEGENERATE_REPEAT = re.compile(r"\b(\w+)(?:\s+\1\b){4,}", re.IGNORECASE)


def _normalise(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _is_verbatim(source: str, output: str) -> bool:
    """True when the engine echoed its input back unchanged.

    Exact match after stripping punctuation and case. A near-match threshold was
    considered and rejected: legitimate translations of short utterances can share
    most of their tokens with the source when it is already mostly English.
    """
    src, out = _normalise(source), _normalise(output)
    return bool(src) and src == out


class ConfidenceGate:
    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = config or GateConfig()

    def check_transcript(self, transcript: Transcript) -> GateDecision:
        """Gate after recognition, before we spend time translating."""
        c = self.config

        if not transcript.text.strip():
            return GateDecision(False, SuppressionReason.EMPTY_RESULT,
                                detail="recogniser returned no text")

        if _DEGENERATE_REPEAT.search(transcript.text):
            return GateDecision(
                False, SuppressionReason.LOW_STT_CONFIDENCE,
                detail="degenerate repetition in transcript",
            )

        if transcript.confidence < c.stt_confidence_threshold:
            return GateDecision(
                False, SuppressionReason.LOW_STT_CONFIDENCE,
                threshold_value=c.stt_confidence_threshold,
                actual_value=transcript.confidence,
                detail=f"recognition confidence {transcript.confidence:.2f} "
                       f"below {c.stt_confidence_threshold:.2f}",
            )

        # An unsupported language must be suppressed whenever it is the DOMINANT
        # reading, not only when it is overwhelming.
        #
        # The original threshold here was 0.8, on the reasoning that a weak
        # third-language signal is noise. That reasoning was wrong for the
        # dominant case: measured on real input, Whisper transcribed a short
        # "hello" as Arabic script at probability 0.6, the gate allowed it, and
        # the translator then raised LANGID_UNSUPPORTED as a hard error instead of
        # a clean suppression. If we cannot translate the dominant language, there
        # is nothing to gain by continuing.
        dominant = transcript.dominant_language
        share = transcript.language_distribution.get(dominant, 0.0)
        if dominant not in c.supported_languages and share >= 0.5:
            return GateDecision(
                False, SuppressionReason.UNSUPPORTED_LANGUAGE,
                actual_value=share,
                detail=f"detected unsupported language '{dominant}'",
            )

        return GateDecision(True)

    def check_translation(self, translation: Translation,
                          transcript: Transcript | None = None) -> GateDecision:
        """Gate after translation, immediately before synthesis."""
        c = self.config
        text = translation.text.strip()

        if len(text) < c.min_output_chars:
            return GateDecision(False, SuppressionReason.EMPTY_RESULT,
                                detail="translation was empty or trivially short")

        if translation.target_language == "en" and _DEVANAGARI.search(text):
            # AC-06.3: English output must not contain Devanagari. If it does, the
            # engine passed input through rather than translating it, and speaking
            # it would emit Hindi to an English audience.
            return GateDecision(
                False, SuppressionReason.LOW_TRANSLATION_CONFIDENCE,
                detail="untranslated Devanagari remains in the English output",
            )

        if _DEGENERATE_REPEAT.search(text):
            return GateDecision(False, SuppressionReason.LOW_TRANSLATION_CONFIDENCE,
                                detail="degenerate repetition in translation")

        # Untranslated passthrough. Two independent signals, because either alone
        # produces false positives: an engine that returns its input verbatim, and
        # romanised Hindi surviving into "English" output.
        #
        # The verbatim check applies ONLY across a language change. For English
        # input, identical output is the CORRECT result -- the pass-through engine
        # is supposed to return what the user said. An earlier version omitted this
        # guard and suppressed "I will complete the integration by Friday
        # afternoon.", i.e. it would have silenced every English utterance.
        if (transcript and translation.target_language == "en"
                and translation.source_language != "en"):
            if _is_verbatim(transcript.text, text):
                return GateDecision(
                    False, SuppressionReason.LOW_TRANSLATION_CONFIDENCE,
                    detail="output is identical to the input; nothing was translated",
                )

        if translation.target_language == "en":
            hits = len(set(m.group(0).lower() for m in _ROMANISED_HINDI.finditer(text)))
            words = max(1, len(text.split()))
            # Two distinct markers AND a meaningful density. A single "ki" or "ho"
            # appears in English text (proper nouns, "ho ho"), so one is not enough.
            if hits >= 2 and hits / words > 0.15:
                return GateDecision(
                    False, SuppressionReason.LOW_TRANSLATION_CONFIDENCE,
                    actual_value=round(hits / words, 3),
                    detail=f"output looks like untranslated romanised Hindi "
                           f"({hits} markers in {words} words)",
                )

        if translation.confidence < c.translation_confidence_threshold:
            return GateDecision(
                False, SuppressionReason.LOW_TRANSLATION_CONFIDENCE,
                threshold_value=c.translation_confidence_threshold,
                actual_value=translation.confidence,
                detail=f"translation confidence {translation.confidence:.2f} "
                       f"below {c.translation_confidence_threshold:.2f}",
            )

        # A translation many times longer than its source is a runaway generation,
        # not a translation. Guarded only for inputs long enough for the ratio to
        # be meaningful.
        if transcript and len(transcript.text) > 20:
            ratio = len(text) / max(1, len(transcript.text))
            if ratio > 4.0:
                return GateDecision(
                    False, SuppressionReason.LOW_TRANSLATION_CONFIDENCE,
                    actual_value=ratio,
                    detail=f"output {ratio:.1f}x longer than input; likely hallucinated",
                )

        return GateDecision(True)

    def applies(self, decision: GateDecision) -> bool:
        """Whether a failed check should actually block audio.

        Separated from the checks themselves so that policy can change without
        touching detection logic -- and so SPEAK_ANYWAY still *records* the
        decision rather than skipping the analysis.
        """
        if decision.allowed:
            return True
        return self.config.policy is not GatePolicy.SUPPRESS and \
            self.config.policy is not GatePolicy.ASK
