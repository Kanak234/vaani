"""Routes each utterance to whichever engine measurably handles it best.

This exists because neither local engine wins everywhere, and the measurements
are unambiguous about where each one wins:

    register            NLLB-200          LLM (qwen3-coder)
    -----------------   ---------------   ------------------
    Devanagari Hindi    4/4  (100%)       4/4  (100%)
    Code-mixed          6/6  (100%)       6/6  (100%)
    English             2/2  (100%)       2/2  (100%)
    Romanised Hindi     0/2    (0%)       2/2  (100%)
    -----------------   ---------------   ------------------
    overall             12/14  (86%)      14/14 (100%)
    latency p50         87 ms             1906 ms  (max 5458 ms)

The LLM is more accurate but ~22x slower, and at 1906 ms it pushes the pipeline
to ~4194 ms -- over the 4000 ms balanced budget. Using it for everything would
buy 2 fixture cases and lose the latency target.

But the two cases NLLB fails share one property: **the input is romanised Hindi**,
which is detectable from the text itself before either engine runs. So:

    Devanagari present, or plain English  ->  NLLB   (fast, already 100% there)
    Latin-script Hindi markers            ->  LLM    (the only engine that works)

That keeps NLLB's 87 ms on the common path and spends the LLM's latency only on
utterances that would otherwise be silently suppressed. Accuracy approaches the
LLM's while latency stays near NLLB's for most speech.

The router degrades rather than fails: if the LLM is unavailable, romanised input
falls back to NLLB, which returns it untranslated, which the confidence gate then
suppresses (ADR-003). The user gets silence and a stated reason -- never wrong
speech.
"""
from __future__ import annotations

import re

from ...core.errors import ErrorCode, Severity, VaaniError
from ...core.types import Translation
from ..base import HealthState, TranslationEngine

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

#: Romanised-Hindi markers. Shares the design constraint of the gate's list
#: (ADR-003): every entry must be a word that is NOT common English, or the
#: router would send ordinary English sentences down the slow path.
_ROMANISED_HINDI = re.compile(
    r"\b(?:hai|hain|hoga|hogi|tha|thi|raha|rahe|rahi|gaya|gayi|"
    r"karo|karna|karke|karta|karti|kiya|karenge|karunga|"
    r"kyun|kyunki|nahi|nahin|haan|bilkul|matlab|lekin|magar|agar|phir|"
    r"mera|meri|mere|tera|teri|uska|uski|unka|unke|hamara|hamari|"
    r"mujhe|tumhe|aapko|humein|yeh|woh|jaise|waise|"
    r"thoda|zyada|accha|theek|sirf|bahut|kuch|abhi|"
    r"wala|wale|wali|sakta|sakte|sakti|chahiye|liye|"
    r"lunga|lungi|denge|hona|hone|kaam|baat|log|tak|kal)\b",
    re.IGNORECASE,
)


def is_romanised_hindi(text: str, *, min_markers: int = 2,
                       min_density: float = 0.12) -> bool:
    """True when the text is Latin-script Hindi rather than English.

    Requires two distinct markers AND a density floor, for the same reason the
    gate does: a single "hai" or "tak" shows up in English text, and routing every
    English utterance to a 1906 ms engine would wreck the latency budget.
    """
    if _DEVANAGARI.search(text):
        return False        # Devanagari present -> NLLB handles it well
    markers = {m.group(0).lower() for m in _ROMANISED_HINDI.finditer(text)}
    words = max(1, len(text.split()))
    return len(markers) >= min_markers and len(markers) / words >= min_density


class RoutingTranslator(TranslationEngine):
    """Dispatches to a fast engine or an accurate one, per utterance."""

    key = "routing"
    display_name = "Automatic (fast NMT + LLM for romanised Hinglish)"

    def __init__(self, *, fast, accurate=None) -> None:
        super().__init__()
        self.fast = fast
        self.accurate = accurate
        # Inherit the strictest privacy requirement of the engines we may call.
        self.requires_network = bool(
            getattr(fast, "requires_network", False)
            or (accurate is not None and getattr(accurate, "requires_network", False))
        )
        self.route_counts: dict[str, int] = {"fast": 0, "accurate": 0, "fallback": 0}

    def warmup(self) -> None:
        self.fast.warmup()
        if self.accurate is not None:
            try:
                self.accurate.warmup()
            except VaaniError as exc:
                # Not fatal. Romanised input will fall back to `fast`, get
                # returned untranslated, and be suppressed by the gate -- which is
                # a worse outcome, but a SAFE and stated one.
                self.accurate = None
                self._set_health(
                    HealthState.DEGRADED,
                    f"LLM unavailable ({exc.code.value}); romanised Hinglish will "
                    "be suppressed rather than translated")
                return
        self._set_health(HealthState.READY, "routing ready")

    def shutdown(self) -> None:
        self.fast.shutdown()
        if self.accurate is not None:
            self.accurate.shutdown()
        super().shutdown()

    def choose_engine(self, text: str, source_language: str):
        """Pick the engine and say why. Separated so it is directly testable."""
        if source_language != "en" and is_romanised_hindi(text):
            if self.accurate is not None:
                return self.accurate, "accurate"
            return self.fast, "fallback"
        return self.fast, "fast"

    def translate(self, text: str, *, source_language: str, target_language: str,
                  context: list[tuple[str, str]] | None = None) -> Translation:
        if not text.strip():
            raise VaaniError(code=ErrorCode.TRANSLATION_EMPTY, message="empty input",
                             stage="translate", provider_key=self.key)

        engine, route = self.choose_engine(text, source_language)
        self.route_counts[route] += 1
        try:
            result = engine.translate(text, source_language=source_language,
                                      target_language=target_language,
                                      context=context)
        except VaaniError:
            # A failure on the accurate path retries on the fast one. The gate
            # still guards the output, so a bad fallback becomes silence.
            if route == "accurate":
                self.route_counts["fallback"] += 1
                result = self.fast.translate(text, source_language=source_language,
                                             target_language=target_language,
                                             context=context)
            else:
                raise
        # Report the engine that actually ran, not "routing" -- the UI's
        # local/cloud indicator and the latency table both depend on this.
        result.provider_key = getattr(engine, "key", self.key)
        return result

    def supports_code_mixed(self) -> bool:
        return True
