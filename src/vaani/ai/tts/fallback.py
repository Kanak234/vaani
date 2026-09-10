"""Dependency-free fallback synthesiser.

This is NOT the product's voice. It is a formant synthesiser that produces
intelligible-ish, obviously synthetic speech using nothing but numpy.

It exists for three reasons:

  1. The pipeline must be end-to-end runnable and testable on day one, before any
     large model is downloaded, and on machines that cannot host one at all.
  2. When the personal-voice backend fails mid-meeting, the choice is between
     silence and an obviously-synthetic voice. Which of those is right depends on
     the user, so it is a setting -- but the fallback must exist to be chosen.
  3. Diagnostics need a way to test the audio output path (virtual mic routing,
     buffering, underruns) without involving a neural model at all.

It ALWAYS sets `is_fallback_voice=True`. The product's core promise is that output
sounds like the user; substituting a different voice without saying so would break
that promise silently, which is worse than breaking it loudly.
"""
from __future__ import annotations

import numpy as np

from ...core.types import SynthesizedAudio
from ..base import HealthState, VoiceSynthesizer

#: Crude per-character formant table. Vowels carry F1/F2; consonants are shaped
#: noise bursts. Enough for the listener to tell that words are being produced.
_VOWELS: dict[str, tuple[float, float]] = {
    "a": (730, 1090), "e": (530, 1840), "i": (390, 1990),
    "o": (570, 840),  "u": (440, 1020), "y": (440, 1500),
}
_VOICED = set("bdgjlmnrvwz")
_BASE_F0 = 118.0


class FallbackSynthesizer(VoiceSynthesizer):
    key = "fallback_formant"
    display_name = "Fallback voice (not your voice)"
    requires_network = False
    supports_voice_cloning = False

    def __init__(self, *, sample_rate: int = 16000, wpm: float = 165.0) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.wpm = wpm

    def warmup(self) -> None:
        self._set_health(HealthState.READY, "no model required")

    def synthesize(self, text: str, *, voice_profile_id: str | None = None,
                   speed: float = 1.0) -> SynthesizedAudio:
        # A voice profile was requested but this backend cannot honour it. Returning
        # audio flagged as fallback (rather than raising) lets the caller decide,
        # but the flag is never omitted.
        samples = self._render(text, speed)
        return SynthesizedAudio(
            samples=samples,
            sample_rate=self.sample_rate,
            voice_profile_id=None,
            is_fallback_voice=True,
        )

    # ------------------------------------------------------------- rendering

    def _render(self, text: str, speed: float) -> np.ndarray:
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        sr = self.sample_rate
        # Roughly 5 characters per word at the configured speaking rate.
        char_s = max(0.02, (60.0 / (self.wpm * 5.0)) / max(0.25, speed))
        out: list[np.ndarray] = []
        phrase_pos = 0.0

        for ch in text.lower():
            low = ch
            if low in _VOWELS:
                f1, f2 = _VOWELS[low]
                out.append(self._voiced(f1, f2, char_s * 1.4, phrase_pos))
                phrase_pos += char_s * 1.4
            elif low.isalpha():
                voiced = low in _VOICED
                out.append(self._consonant(char_s * 0.7, voiced, phrase_pos))
                phrase_pos += char_s * 0.7
            elif low in ",;:":
                out.append(np.zeros(int(sr * 0.14), dtype=np.float32))
                phrase_pos += 0.14
            elif low in ".!?":
                out.append(np.zeros(int(sr * 0.28), dtype=np.float32))
                phrase_pos = 0.0                      # reset declination
            elif low.isspace():
                out.append(np.zeros(int(sr * 0.05), dtype=np.float32))
                phrase_pos += 0.05

        if not out:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(out)
        return self._finish(audio)

    def _f0(self, phrase_pos: float) -> float:
        """Falling declination across a phrase -- flat pitch reads as robotic."""
        return _BASE_F0 * (1.0 - min(0.18, phrase_pos * 0.06))

    def _voiced(self, f1: float, f2: float, dur_s: float,
                phrase_pos: float) -> np.ndarray:
        n = max(1, int(self.sample_rate * dur_s))
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        f0 = self._f0(phrase_pos)
        # Glottal source: a few harmonics with 1/k rolloff.
        src = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
        sig = (0.55 * np.sin(2 * np.pi * f1 * t)
               + 0.30 * np.sin(2 * np.pi * f2 * t)
               + 0.35 * src)
        return (sig * _envelope(n) * 0.30).astype(np.float32)

    def _consonant(self, dur_s: float, voiced: bool,
                   phrase_pos: float) -> np.ndarray:
        n = max(1, int(self.sample_rate * dur_s))
        noise = np.random.default_rng().standard_normal(n).astype(np.float32)
        # One-pole high-pass gives the noise a fricative character.
        filtered = np.diff(noise, prepend=noise[0])
        sig = filtered * 0.16
        if voiced:
            t = np.arange(n, dtype=np.float32) / self.sample_rate
            sig = sig * 0.6 + np.sin(2 * np.pi * self._f0(phrase_pos) * t) * 0.16
        return (sig * _envelope(n)).astype(np.float32)

    def _finish(self, audio: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.55        # headroom, avoids clipping downstream
        fade = min(int(self.sample_rate * 0.01), audio.size // 2)
        if fade > 0:
            audio[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            audio[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        return audio.astype(np.float32)


def _envelope(n: int) -> np.ndarray:
    """Short attack/release so segment joins do not click."""
    env = np.ones(n, dtype=np.float32)
    edge = max(1, n // 8)
    env[:edge] = np.linspace(0, 1, edge, dtype=np.float32)
    env[-edge:] = np.linspace(1, 0, edge, dtype=np.float32)
    return env
