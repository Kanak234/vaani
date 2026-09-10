"""Routing translator: send each utterance to the engine that measurably wins.

Measured basis for the routing rule (docs/07 §7):
    register          NLLB      LLM
    Devanagari        4/4       4/4
    Code-mixed        6/6       6/6
    English           2/2       2/2
    Romanised         0/2       2/2     <- the only divergence
    latency p50       87 ms     1906 ms
"""
from __future__ import annotations

import pytest

from vaani.ai.translate.router import RoutingTranslator, is_romanised_hindi
from vaani.core.errors import ErrorCode, Severity, VaaniError
from vaani.core.types import Translation


class StubEngine:
    def __init__(self, key, *, error=None, text=None):
        self.key = key
        self.requires_network = False
        self._error = error
        self._text = text
        self.calls = 0
        self.last_context = None

    def warmup(self):
        if self._error:
            raise self._error

    def shutdown(self):
        pass

    def translate(self, text, *, source_language, target_language, context=None):
        self.calls += 1
        self.last_context = context
        if self._error:
            raise self._error
        return Translation(text=self._text or f"[{self.key}] {text}",
                           source_language=source_language,
                           target_language=target_language, confidence=0.9)


def build(accurate_error=None):
    fast = StubEngine("nllb")
    accurate = StubEngine("llm", error=accurate_error)
    return RoutingTranslator(fast=fast, accurate=accurate), fast, accurate


# --- the classifier ---------------------------------------------------------

@pytest.mark.parametrize("text", [
    "haan bilkul, main kal call kar lunga",
    "mera point ye hai ki hum kal tak finish kar sakte hain",
    "mujhe teen din chahiye is kaam ke liye",
])
def test_romanised_hindi_detected(text):
    assert is_romanised_hindi(text)


@pytest.mark.parametrize("text", [
    "I will complete the integration by Friday afternoon.",
    "The authentication module needs more testing before deployment.",
    "Schedule a meeting with the team on Thursday.",
    "Actually मेरा point ये है कि deadline tight है",   # Devanagari present
    "मुझे तीन दिन चाहिए",
])
def test_english_and_devanagari_not_misrouted(text):
    """Misrouting English to a 1906 ms engine would wreck the latency budget."""
    assert not is_romanised_hindi(text)


def test_single_marker_is_not_enough():
    """'tak' and 'hai' appear in English text; one marker must not trigger."""
    assert not is_romanised_hindi("The Hai Corporation filed by Thursday.")


# --- routing ----------------------------------------------------------------

def test_devanagari_goes_to_the_fast_engine():
    r, fast, accurate = build()
    r.translate("मुझे तीन दिन चाहिए", source_language="hi", target_language="en")
    assert fast.calls == 1 and accurate.calls == 0


def test_english_goes_to_the_fast_engine():
    r, fast, accurate = build()
    r.translate("I will finish by Friday.", source_language="en", target_language="en")
    assert fast.calls == 1 and accurate.calls == 0


def test_romanised_goes_to_the_accurate_engine():
    r, fast, accurate = build()
    r.translate("haan bilkul, main kal call kar lunga",
                source_language="hi", target_language="en")
    assert accurate.calls == 1 and fast.calls == 0


def test_route_counts_are_tracked():
    r, _, _ = build()
    r.translate("मुझे तीन दिन चाहिए", source_language="hi", target_language="en")
    r.translate("haan bilkul main kal kar lunga", source_language="hi",
                target_language="en")
    assert r.route_counts["fast"] == 1
    assert r.route_counts["accurate"] == 1


def test_result_reports_the_engine_that_actually_ran():
    """The UI's provider indicator and latency table depend on this."""
    r, _, _ = build()
    out = r.translate("haan bilkul, main kal call kar lunga",
                      source_language="hi", target_language="en")
    assert out.provider_key == "llm"


def test_context_is_forwarded():
    r, _, accurate = build()
    ctx = [("पहले वाला", "the earlier one")]
    r.translate("haan bilkul main kal kar lunga", source_language="hi",
                target_language="en", context=ctx)
    assert accurate.last_context == ctx


# --- degradation ------------------------------------------------------------

def test_llm_unavailable_at_warmup_degrades_safely():
    """Romanised input then goes to NLLB, returns untranslated, and the GATE
    suppresses it. Silence with a reason -- never wrong speech."""
    err = VaaniError(code=ErrorCode.PROVIDER_UNAVAILABLE, message="ollama down",
                     severity=Severity.FATAL)
    r, fast, _ = build(accurate_error=err)
    r.warmup()
    assert r.accurate is None
    assert "suppressed" in r.health.detail.lower()
    r.translate("haan bilkul main kal kar lunga", source_language="hi",
                target_language="en")
    assert fast.calls == 1
    assert r.route_counts["fallback"] == 1


def test_llm_failure_mid_session_falls_back_to_fast():
    r, fast, accurate = build()
    accurate._error = VaaniError(code=ErrorCode.TRANSLATION_TIMEOUT,
                                 message="timeout", severity=Severity.UTTERANCE)
    r.translate("haan bilkul main kal kar lunga", source_language="hi",
                target_language="en")
    assert fast.calls == 1
    assert r.route_counts["fallback"] == 1


def test_fast_engine_failure_is_not_swallowed():
    r, fast, _ = build()
    fast._error = VaaniError(code=ErrorCode.TRANSLATION_FAILURE, message="boom",
                             severity=Severity.UTTERANCE)
    with pytest.raises(VaaniError):
        r.translate("मुझे तीन दिन चाहिए", source_language="hi", target_language="en")


def test_empty_input_rejected():
    r, _, _ = build()
    with pytest.raises(VaaniError) as e:
        r.translate("   ", source_language="hi", target_language="en")
    assert e.value.code is ErrorCode.TRANSLATION_EMPTY


# --- privacy ----------------------------------------------------------------

def test_router_is_local_when_both_engines_are_local():
    r, _, _ = build()
    assert not r.requires_network


def test_router_inherits_network_requirement():
    """If either engine needs the network, so does the router."""
    fast = StubEngine("nllb")
    accurate = StubEngine("cloud")
    accurate.requires_network = True
    assert RoutingTranslator(fast=fast, accurate=accurate).requires_network
