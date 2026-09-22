"""Tests for ai/model_detect.py (Phase 3 auto model detection).

This module shipped as rescued work-in-progress with NO tests at all, which
matters more than usual here: it decides which LLM translates the user's speech,
and its ranking encodes measured claims about which models get dates wrong.

Nothing here touches the network. `list_models` and `probe_model` are the only
two functions that do I/O, and both are substituted, so the resolution logic is
tested against every Ollama state -- absent, empty, wedged, healthy -- including
the ones that are hard to produce on a real machine on purpose.
"""
from __future__ import annotations

import json
import urllib.error

import pytest

from vaani.ai import model_detect as md


# ------------------------------------------------------------- name handling

@pytest.mark.parametrize("raw,expected", [
    ("phi4-mini:latest", "phi4-mini"),
    ("phi4-mini", "phi4-mini"),
    ("  Phi4-Mini:3B  ", "phi4-mini"),
    ("llama3.1:8b", "llama3.1"),
])
def test_base_name_strips_the_ollama_tag(raw, expected):
    assert md.base_name(raw) == expected


def test_a_derived_model_ranks_as_its_base():
    """THE case this machine actually has: `rishi-phi4-mini-latest:latest` is a
    phi4-mini and must rank as one. Equality matching would call it unknown and
    rank the tested-good model below an untested one."""
    assert md.rank_of("rishi-phi4-mini-latest:latest") == md.rank_of("phi4-mini")


def test_coder_builds_rank_last():
    """Measured WRONG on the date fixture, so they lose to everything else --
    but are still usable, because a flawed translation the gate can inspect
    beats no translation."""
    assert md.rank_of("qwen2.5-coder:3b") > md.rank_of("phi4-mini")
    assert md.rank_of("qwen2.5-coder:3b") > md.rank_of("some-unknown-model")
    assert md.is_discouraged("qwen2.5-coder:3b") is True
    assert md.is_discouraged("rishi-qwen2.5-coder-3b:latest") is True


def test_embedding_models_are_discouraged_too():
    """An embedding model cannot translate at all; it must never be picked over
    a real one."""
    assert md.is_discouraged("nomic-embed-text:latest") is True


def test_an_unknown_model_outranks_a_coder_build():
    """An unknown model that works is strictly better than one measured wrong."""
    assert md.rank_of("brand-new-llm:7b") < md.rank_of("qwen2.5-coder:3b")


def test_a_known_good_model_outranks_an_unknown_one():
    assert md.rank_of("phi4-mini") < md.rank_of("brand-new-llm:7b")


def test_ranking_puts_the_measured_best_first():
    installed = ["qwen2.5-coder:3b", "llama3.1:8b", "phi4-mini:latest",
                 "mystery:1b"]
    assert md.rank_models(installed)[0] == "phi4-mini:latest"
    assert md.rank_models(installed)[-1] == "qwen2.5-coder:3b"


def test_ranking_is_stable_within_a_rank():
    """Ties break on `ollama list` order rather than arbitrarily, so the choice
    is reproducible between runs."""
    installed = ["unknown-a:1b", "unknown-b:1b", "unknown-c:1b"]
    assert md.rank_models(installed) == installed


def test_a_base_model_is_not_mistaken_for_a_derived_one():
    """REGRESSION. The first implementation asked "does ANY preferred entry
    appear in this name?", so `phi4-mini` matched the separate entry `phi4`
    and declared itself a derived build of itself. The tie-break then did
    nothing and the custom model still won."""
    assert md.is_derived("phi4-mini:latest") is False
    assert md.is_derived("phi4-mini") is False
    assert md.is_derived("llama3.1:8b") is False


def test_a_custom_modelfile_build_is_recognised_as_derived():
    assert md.is_derived("rishi-phi4-mini-latest:latest") is True


def test_the_exact_model_beats_a_custom_build_of_it():
    """THE bug this tie-break exists for, found on the dev machine:
    `rishi-phi4-mini-latest` ranks as a phi4-mini, but its Modelfile system
    prompt says "Reply in the same language he uses (English or Hindi)".
    Vaani must ALWAYS emit English (AC-06.3), so that model produces
    Devanagari the gate then suppresses - the user sees utterances vanishing
    and concludes Vaani is broken.

    The measured evidence in PREFERRED_MODELS was gathered on the BASE model,
    so when both are installed the base is what the evidence covers."""
    installed = ["rishi-phi4-mini-latest:latest", "phi4-mini:latest"]
    assert md.rank_models(installed)[0] == "phi4-mini:latest"


def test_a_derived_model_is_still_used_when_it_is_all_there_is():
    """Preferring the base must not mean refusing the variant."""
    installed = ["rishi-phi4-mini-latest:latest"]
    assert md.rank_models(installed)[0] == "rishi-phi4-mini-latest:latest"


def test_this_machines_real_model_list_picks_phi4_mini():
    """The exact set installed on the dev laptop. Guards the end-to-end
    ordering decision, not just the individual rank function."""
    installed = ["rishi-phi4-mini-latest:latest", "rishi-qwen2.5-coder-3b:latest",
                 "qwen2.5-coder:3b", "phi4-mini:latest"]
    best = md.rank_models(installed)[0]
    # The plain measured model, not the persona-carrying custom build.
    assert best == "phi4-mini:latest"
    assert md.is_discouraged(best) is False


# ------------------------------------------------- list_models is forgiving

def test_ollama_not_running_gives_an_empty_list_not_an_exception(monkeypatch):
    """"Ollama is not running" is an ordinary state for this product -- it
    falls back to NLLB -- so it must not raise."""
    def boom(*_a, **_kw):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(md.urllib.request, "urlopen", boom)
    assert md.list_models() == []


def test_malformed_json_from_ollama_is_survived(monkeypatch):
    class FakeResponse:
        status = 200

        def read(self):
            return b"not json at all"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(md.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    assert md.list_models() == []


def test_entries_without_a_name_are_skipped(monkeypatch):
    class FakeResponse:
        status = 200

        def read(self):
            return json.dumps({"models": [{"name": "phi4-mini:latest"},
                                          {"size": 123},
                                          {"name": ""}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(md.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    assert md.list_models() == ["phi4-mini:latest"]


# ------------------------------------------------------------- candidates

def test_a_pinned_model_is_returned_alone(monkeypatch):
    """An explicit --llm-model must never silently fall back to something
    else."""
    monkeypatch.setattr(md, "list_models", lambda *a, **k: ["phi4-mini:latest"])
    assert md.candidate_models(pinned="llama3.1:8b") == ["llama3.1:8b"]


def test_candidates_are_capped(monkeypatch):
    """Start-up stays bounded for someone with twenty models installed."""
    many = [f"model-{i}:1b" for i in range(20)]
    monkeypatch.setattr(md, "list_models", lambda *a, **k: many)
    assert len(md.candidate_models(max_candidates=4)) == 4


def test_candidates_do_not_probe(monkeypatch):
    """The cheap half: ordering must not load any model into VRAM."""
    monkeypatch.setattr(md, "list_models", lambda *a, **k: ["phi4-mini:latest"])
    monkeypatch.setattr(md, "probe_model",
                        lambda *a, **k: pytest.fail("probe must not run here"))
    assert md.candidate_models() == ["phi4-mini:latest"]


# --------------------------------------------------------------- resolution

def test_resolution_picks_the_best_installed(monkeypatch):
    monkeypatch.setattr(md, "list_models",
                        lambda *a, **k: ["qwen2.5-coder:3b", "phi4-mini:latest"])
    choice = md.resolve_llm_model()
    assert choice.ok is True
    assert choice.name == "phi4-mini:latest"
    assert choice.discouraged is False


def test_no_models_installed_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(md, "list_models", lambda *a, **k: [])
    choice = md.resolve_llm_model()
    assert choice.ok is False
    assert "ollama pull" in choice.describe()


def test_a_pinned_model_that_is_missing_is_an_error_not_a_substitution(monkeypatch):
    """If the user's two settings contradict each other they should learn
    that, not quietly get a different model."""
    monkeypatch.setattr(md, "list_models", lambda *a, **k: ["phi4-mini:latest"])
    choice = md.resolve_llm_model(pinned="llama3.1:8b")
    assert choice.ok is False
    assert choice.rejected["llama3.1:8b"] == "not installed"
    assert "no usable local LLM" in choice.describe()


def test_a_pinned_model_matches_despite_a_missing_tag(monkeypatch):
    """`--llm-model phi4-mini` must find `phi4-mini:latest`: that is a spelling
    detail, not a different model."""
    monkeypatch.setattr(md, "list_models", lambda *a, **k: ["phi4-mini:latest"])
    choice = md.resolve_llm_model(pinned="phi4-mini")
    assert choice.ok is True
    assert choice.name == "phi4-mini:latest"


def test_a_failing_candidate_is_skipped_not_fatal(monkeypatch):
    """AC-22.3: listed is not the same as runnable. A model can be corrupt or
    too big for a 4 GiB GPU."""
    monkeypatch.setattr(md, "list_models",
                        lambda *a, **k: ["phi4-mini:latest", "llama3.1:8b"])
    monkeypatch.setattr(md, "probe_model",
                        lambda m, *a, **k: "out of memory"
                        if m == "phi4-mini:latest" else None)
    choice = md.resolve_llm_model(probe=True)
    assert choice.name == "llama3.1:8b"
    assert choice.rejected["phi4-mini:latest"] == "out of memory"


def test_every_candidate_failing_reports_all_the_reasons(monkeypatch):
    monkeypatch.setattr(md, "list_models",
                        lambda *a, **k: ["phi4-mini:latest", "llama3.1:8b"])
    monkeypatch.setattr(md, "probe_model", lambda m, *a, **k: "HTTP 500")
    choice = md.resolve_llm_model(probe=True)
    assert choice.ok is False
    described = choice.describe()
    assert "phi4-mini:latest (HTTP 500)" in described
    assert "llama3.1:8b (HTTP 500)" in described


def test_resolution_without_probing_does_not_probe(monkeypatch):
    """The default path runs at meeting start-up; a probe there would cost a
    full model load that warmup() then repeats."""
    monkeypatch.setattr(md, "list_models", lambda *a, **k: ["phi4-mini:latest"])
    monkeypatch.setattr(md, "probe_model",
                        lambda *a, **k: pytest.fail("must not probe by default"))
    assert md.resolve_llm_model().ok is True


def test_only_a_coder_installed_is_used_but_warned_about(monkeypatch):
    """A flawed translation the gate can inspect beats no translation - but the
    user must be told which model they are on."""
    monkeypatch.setattr(md, "list_models", lambda *a, **k: ["qwen2.5-coder:3b"])
    choice = md.resolve_llm_model()
    assert choice.ok is True
    assert choice.discouraged is True
    assert "WARNING" in choice.describe()
    assert "phi4-mini" in choice.describe()     # names the tested alternative


def test_describe_is_a_single_line(monkeypatch):
    """It is printed into a terminal status block and a GUI log line."""
    monkeypatch.setattr(md, "list_models", lambda *a, **k: ["phi4-mini:latest"])
    assert "\n" not in md.resolve_llm_model().describe()


def test_elapsed_is_recorded_for_the_startup_budget(monkeypatch):
    monkeypatch.setattr(md, "list_models", lambda *a, **k: ["phi4-mini:latest"])
    assert md.resolve_llm_model().elapsed_ms >= 0.0


# --------------------------------------------------------------- the probe

def test_probe_reports_http_errors_as_reasons(monkeypatch):
    def raise_http(*_a, **_kw):
        raise urllib.error.HTTPError("u", 500, "boom", {}, None)

    monkeypatch.setattr(md.urllib.request, "urlopen", raise_http)
    assert md.probe_model("phi4-mini:latest") == "HTTP 500"


def test_probe_reports_timeouts_as_reasons(monkeypatch):
    def raise_timeout(*_a, **_kw):
        raise TimeoutError("too slow")

    monkeypatch.setattr(md.urllib.request, "urlopen", raise_timeout)
    reason = md.probe_model("phi4-mini:latest")
    assert reason is not None and "timeout" in reason


def test_probe_returns_none_on_success(monkeypatch):
    class FakeResponse:
        status = 200

        def read(self):
            return json.dumps({"response": "hi"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(md.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    assert md.probe_model("phi4-mini:latest") is None
