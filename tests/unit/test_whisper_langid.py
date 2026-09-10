"""Language-distribution estimation, including short-utterance behaviour."""
from __future__ import annotations

from vaani.ai.stt.whisper import _language_distribution


class Info:
    def __init__(self, language, probability):
        self.language = language
        self.language_probability = probability


def test_short_utterance_ignores_an_exotic_language_guess():
    """Regression: a one-word "hello" was detected as Arabic and rendered in
    Arabic script. On short text the script ratio is evidence; the guess is not."""
    d = _language_distribution("Hello?", Info("ar", 0.6))
    assert set(d) <= {"hi", "en"}
    assert d.get("en", 0) > 0.9


def test_short_devanagari_is_still_hindi():
    d = _language_distribution("हाँ", Info("hi", 0.9))
    assert d.get("hi", 0) > 0.9


def test_non_latin_non_devanagari_script_is_reported_for_suppression():
    """A third script must reach the gate as itself, so it can be suppressed."""
    d = _language_distribution("هلو هلو مرحبا بكم", Info("ar", 0.85))
    assert "ar" in d


def test_long_code_mixed_text_uses_the_script_ratio():
    d = _language_distribution(
        "Actually मेरा point ये है कि deadline थोड़ी tight है", Info("hi", 0.8))
    assert 0.1 < d["hi"] < 0.9
    assert 0.1 < d["en"] < 0.9


def test_confident_hindi_on_romanised_text_is_split():
    """Romanised Hinglish is all-Latin; the model's prior is the only signal."""
    d = _language_distribution(
        "mera point ye hai ki hum kal tak finish kar sakte hain", Info("hi", 0.9))
    assert d.get("hi", 0) >= 0.4
