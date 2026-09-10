"""Ollama translator: output cleaning, confidence, and the privacy claim.

Network calls are not made here; these cover the logic around them.
"""
from __future__ import annotations

import pytest

from vaani.ai.translate.ollama import (
    OllamaTranslator,
    _confidence,
    _is_loopback,
    _strip_model_chatter,
)


# --- privacy ----------------------------------------------------------------

def test_loopback_host_is_not_network_egress():
    """Ollama on localhost keeps the local-only guarantee intact (AC-17.2)."""
    assert not OllamaTranslator().requires_network
    assert not OllamaTranslator(host="http://127.0.0.1:11434").requires_network


def test_remote_host_is_declared_as_network():
    """A remote Ollama really does send data off the machine; do not lie."""
    assert OllamaTranslator(host="http://192.168.1.50:11434").requires_network
    assert OllamaTranslator(host="https://ollama.example.com").requires_network


@pytest.mark.parametrize("host,expected", [
    ("http://localhost:11434", True),
    ("http://127.0.0.1:11434", True),
    ("http://[::1]:11434", True),
    ("http://10.0.0.5:11434", False),
    ("https://example.com", False),
])
def test_loopback_detection(host, expected):
    assert _is_loopback(host) is expected


# --- output cleaning --------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Sure! Here is the translation: I will call tomorrow.", "I will call tomorrow."),
    ("Here's the translation:\nI will call tomorrow.", "I will call tomorrow."),
    ("English: I will call tomorrow.", "I will call tomorrow."),
    ('"I will call tomorrow."', "I will call tomorrow."),
    ("I will call tomorrow.", "I will call tomorrow."),
    ("<think>The user said kal...</think>I will call tomorrow.",
     "I will call tomorrow."),
])
def test_model_chatter_is_stripped(raw, expected):
    """Speaking 'Sure! Here is the translation:' into a meeting would be worse
    than most translation errors."""
    assert _strip_model_chatter(raw) == expected


def test_only_the_first_line_is_kept():
    raw = "I will call tomorrow.\n\nNote: 'kal' can mean yesterday too."
    assert _strip_model_chatter(raw) == "I will call tomorrow."


def test_empty_response_yields_empty():
    assert _strip_model_chatter("") == ""
    assert _strip_model_chatter("   \n  ") == ""


# --- confidence -------------------------------------------------------------

def test_normal_output_is_confident():
    assert _confidence("main kal call karunga", "I will call tomorrow.") > 0.7


def test_untranslated_echo_across_languages_is_penalised():
    text = "haan bilkul main kal call kar lunga"
    assert _confidence(text, text, same_language=False) < 0.3


def test_english_echo_is_not_penalised():
    """Regression: English in -> identical English out is CORRECT.

    Without the same_language flag this scored 0.20 and the gate suppressed a
    perfectly good English utterance -- the same class of bug as ADR-003.
    """
    text = "The authentication module needs more testing before deployment."
    assert _confidence(text, text, same_language=True) > 0.7


def test_runaway_output_is_penalised():
    assert _confidence("haan", "yes " * 60) < 0.6


def test_truncated_output_is_penalised():
    assert _confidence("mera point ye hai ki hum kal tak finish kar sakte hain",
                       "Yes") < 0.6


def test_empty_output_has_zero_confidence():
    assert _confidence("kuch bhi", "") == 0.0


# --- prompt -----------------------------------------------------------------

def test_prompt_includes_context_turns():
    t = OllamaTranslator()
    p = t._build_prompt("usme authentication baaki hai",
                        [("We completed the backend API.", "We completed the backend API.")],
                        "hi", "en")
    assert "backend API" in p
    assert "context only" in p.lower()


def test_prompt_forbids_answering_the_input():
    """An instruction-tuned model will otherwise REPLY to a question instead of
    translating it -- dangerous when the user is mid-meeting."""
    p = OllamaTranslator()._build_prompt("kya time hai?", [], "hi", "en")
    assert "never answer" in p.lower()


def test_prompt_demands_number_and_date_fidelity():
    p = OllamaTranslator()._build_prompt("kal", [], "hi", "en")
    assert "tomorrow" in p.lower() and "next week" in p.lower()
