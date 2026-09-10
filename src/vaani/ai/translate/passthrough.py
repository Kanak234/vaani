"""Pass-through 'translation' for English input, plus shared text normalisation.

When the user already spoke English there is nothing to translate, and routing it
through an NMT model would add latency and risk paraphrasing what they said. This
engine handles that path and applies the light normalisation (F-07) that every
engine's output needs before it reaches TTS.
"""
from __future__ import annotations

import re

from ...core.types import Translation
from ..base import HealthState, TranslationEngine

_FILLERS = re.compile(
    r"\b(?:um+|uh+|erm+|hmm+|matlab|yaani|aisa hai ki)\b[,\s]*",
    re.IGNORECASE,
)
_REPEATED_WORD = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)
_MULTISPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")


def normalize_for_speech(text: str, *, strip_fillers: bool = True) -> str:
    """Prepare translated text for synthesis (F-07).

    Kept conservative on purpose. Aggressive rewriting would change what the user
    said, and the product's whole premise is that the output faithfully represents
    them. Numbers and dates are explicitly left alone -- AC-06.6 requires they
    survive untouched, and "helpful" reformatting is exactly how "next week"
    becomes something else.
    """
    if not text:
        return ""
    out = text.strip()
    if strip_fillers:
        out = _FILLERS.sub("", out)
    out = _REPEATED_WORD.sub(r"\1", out)      # stutter artefacts from STT
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _MULTISPACE.sub(" ", out).strip()
    if out and out[-1] not in ".!?,:;":
        # Terminal punctuation drives TTS prosody; without it the synthesiser
        # tends to trail off flatly, which reads as uncertainty.
        out += "."
    return out[:1].upper() + out[1:] if out else out


class PassthroughTranslator(TranslationEngine):
    key = "passthrough"
    display_name = "Pass-through (English in, English out)"
    requires_network = False

    def warmup(self) -> None:
        self._set_health(HealthState.READY, "no model required")

    def translate(self, text: str, *, source_language: str, target_language: str,
                  context: list[tuple[str, str]] | None = None) -> Translation:
        cleaned = normalize_for_speech(text)
        return Translation(
            text=cleaned,
            source_language=source_language,
            target_language=target_language,
            # High but not 1.0: the input is unchanged, yet upstream STT may
            # still have misheard it, and confidence here should not paper over that.
            confidence=0.95,
            used_context_turns=0,
            provider_key=self.key,
        )

    def supports_code_mixed(self) -> bool:
        return False
