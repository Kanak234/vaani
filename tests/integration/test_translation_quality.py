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
