"""Translation quality against the Hinglish fixtures.

Skipped unless the NLLB model is already downloaded, so a fresh clone's test run
does not silently pull 2.5 GB. Run explicitly with:

    PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_translation_quality.py -v

These assert the MEASURED baseline. If a change to the gate, the tokenizer or the
model drops Hinglish below 100% or lets romanised Hindi through, that is a
regression in the product's core promise and the suite should say so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from hinglish_cases import CASES, by_register  # noqa: E402

from vaani.core.gate import ConfidenceGate, GateConfig  # noqa: E402
from vaani.core.types import Transcript  # noqa: E402


def _model_present() -> bool:
    from pathlib import Path as P
    hub = P.home() / ".cache" / "huggingface" / "hub"
    return any(hub.glob("*nllb-200-distilled-600M-ctranslate2*")) if hub.exists() else False


pytestmark = pytest.mark.skipif(
    not _model_present(),
    reason="NLLB model not downloaded; run scripts/eval_translation.py first",
)


@pytest.fixture(scope="module")
def engine():
    from vaani.ai.translate.nllb import NllbTranslator
    e = NllbTranslator()
    e.warmup()
    return e


@pytest.fixture(scope="module")
def gate():
    return ConfidenceGate(GateConfig())


def _translate(engine, case):
    src = "en" if case.register == "english" else "hi"
    return engine.translate(case.text, source_language=src, target_language="en")


def _check(case, text: str) -> bool:
    low = text.lower()
    return (all(t.lower() in low for t in case.must_contain)
            and not any(t.lower() in low for t in case.must_not_contain))


@pytest.mark.parametrize("case", by_register("devanagari"),
                         ids=lambda c: c.text[:28])
def test_pure_hindi_translates(engine, case):
    assert _check(case, _translate(engine, case).text)


@pytest.mark.parametrize("case", by_register("hinglish"),
                         ids=lambda c: c.text[:28])
def test_code_mixed_hinglish_translates(engine, case):
    """The product's core register. Measured at 100%; must not regress."""
    assert _check(case, _translate(engine, case).text)


@pytest.mark.parametrize("case", by_register("english"),
                         ids=lambda c: c.text[:28])
def test_english_survives_unchanged(engine, case):
    assert _check(case, _translate(engine, case).text)


@pytest.mark.parametrize("case", by_register("romanised"),
                         ids=lambda c: c.text[:28])
def test_romanised_hindi_is_suppressed_not_spoken(engine, gate, case):
    """NLLB cannot translate romanised Hindi; the gate MUST catch it.

    The requirement is not that this succeeds -- it is that it never emits
    untranslated Hindi to an English audience (ADR-003).
    """
    result = _translate(engine, case)
    decision = gate.check_translation(
        result,
        Transcript(text=case.text, language_distribution={"hi": 1.0}, confidence=0.9),
    )
    translated_ok = _check(case, result.text)
    assert translated_ok or decision.suppressed, (
        f"untranslated output was allowed through: {result.text!r}")


def test_english_is_never_falsely_suppressed(engine, gate):
    """Regression: the verbatim check once silenced every English utterance."""
    for case in by_register("english"):
        result = _translate(engine, case)
        decision = gate.check_translation(
            result,
            Transcript(text=case.text, language_distribution={"en": 1.0},
                       confidence=0.9),
        )
        assert decision.allowed, f"{case.text!r} was suppressed: {decision.detail}"


def test_commitments_preserve_their_numbers_and_dates(engine):
    """AC-06.6. Getting a date wrong is the failure that causes real harm."""
    failures = []
    for case in CASES:
        if case.group != "commitment" or case.register == "romanised":
            continue
        if not _check(case, _translate(engine, case).text):
            failures.append(case.text[:40])
    assert not failures, f"commitment cases drifted: {failures}"


def test_overall_accuracy_does_not_regress(engine):
    """Measured baseline: 12/14 (86%). Fail if it drops below that."""
    passed = sum(1 for c in CASES if _check(c, _translate(engine, c).text))
    assert passed >= 12, f"accuracy regressed to {passed}/{len(CASES)}"


# --- offline script mode (Phase 5) ------------------------------------------
#
# WHY THIS IS HERE rather than in tests/unit/test_script_mode.py: the unit
# tests stub the translator, and a stub accepts ANY source_language string.
# That is exactly how script mode shipped its first bug - it passed
# source_language="auto", which the real NLLB translator rejects with
# LANGID_UNSUPPORTED, so every chunk of a real run failed while every unit
# test passed. Only the real provider can catch that class of mistake.

def test_script_mode_translates_fixtures_with_the_real_translator():
    """Drive ScriptJob's own code path with a real translator.

    The recognizer is stubbed (this is about translation, and real Whisper
    would need real audio), but it returns a genuine Transcript carrying a
    real language_distribution - which is what ScriptJob reads to decide the
    source language.
    """
    import numpy as np

    from vaani.ai.translate.nllb import NllbTranslator
    from vaani.core.types import Transcript
    from vaani.session.script_mode import ScriptConfig, ScriptJob

    cases = [c for c in CASES if c.register == "devanagari"][:3]
    if not cases:
        pytest.skip("no Devanagari fixtures")

    class FixtureRecognizer:
        """Returns one fixture per chunk, in order."""

        def __init__(self, texts):
            self.texts = list(texts)
            self.index = 0

        def transcribe(self, audio, sample_rate, *, language_hint=None,
                       context_prompt=None):
            text = self.texts[min(self.index, len(self.texts) - 1)]
            self.index += 1
            return Transcript(text=text, language_distribution={"hi": 1.0},
                              confidence=0.9)

    translator = NllbTranslator()
    translator.warmup()

    # One chunk per fixture: 30 s of audio at a 10 s target.
    audio = np.zeros(16000 * 30, dtype=np.float32)
    job = ScriptJob(audio=audio, sample_rate=16000,
                    recognizer=FixtureRecognizer(c.text for c in cases),
                    translator=translator,
                    config=ScriptConfig(chunk_s=10.0))
    result = job.run()

    # The bug this test exists for: every chunk failing on a language code.
    assert result.failed_chunks == 0, (
        f"chunks failed: {[s.error for s in result.segments if s.error]}")

    script = result.script_text().lower()
    for case in cases:
        for needle in case.must_contain:
            assert needle.lower() in script, (
                f"{needle!r} missing for {case.text!r}; script was {script!r}")
        for banned in case.must_not_contain:
            assert banned.lower() not in script


def test_script_mode_output_is_english_not_devanagari():
    """AC-06.3 applies to the written script too: a presentation script with
    Devanagari in it is useless for its stated purpose."""
    import numpy as np

    from vaani.ai.translate.nllb import NllbTranslator
    from vaani.core.types import Transcript
    from vaani.session.script_mode import ScriptConfig, ScriptJob

    class R:
        def transcribe(self, audio, sample_rate, **kw):
            return Transcript(text="मैं कल मीटिंग में आऊंगा",
                              language_distribution={"hi": 1.0}, confidence=0.9)

    translator = NllbTranslator()
    translator.warmup()
    job = ScriptJob(audio=np.zeros(16000 * 12, dtype=np.float32),
                    sample_rate=16000, recognizer=R(), translator=translator,
                    config=ScriptConfig(chunk_s=10.0))
    script = job.run().script_text()

    assert script.strip()
    # No Devanagari codepoints anywhere in the English script.
    assert not any("ऀ" <= ch <= "ॿ" for ch in script), script
