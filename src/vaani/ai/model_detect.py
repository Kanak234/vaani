"""Detect which local LLMs actually exist, and pick the best one that works.

WHY THIS MODULE EXISTS
----------------------
Before this, three entry points each hardcoded a different Ollama model:

    cli.py `meeting` / `run`          --llm-model qwen3-coder:latest
    session/meeting_takeover_app.py   --llm-model qwen3:8b
    ai/translate/ollama.py            _DEFAULT_MODEL = phi4-mini:latest

On the development machine `ollama list` returns phi4-mini, qwen2.5-coder:3b and
two `rishi-*` derivatives -- i.e. **two of those three defaults are not
installed**, so the CLI's own default made translation fail. Worse, the repo's
own measurement (see `ai/translate/ollama.py` module docstring) records the
coder-tuned builds as *wrong* on date preservation, which is the single error
class the product cares most about. So the hardcoded default was both absent and
the one we had measured as bad.

This module replaces "hope the hardcoded name is installed" with "ask, rank,
probe, fall back". It is the implementation of F-22 / AC-22.1..22.7 in
`docs/15-MEETING-UNIFICATION.md`.

WHERE IT IS USED
----------------
    cli.py:cmd_meeting / cmd_run   -> resolve_llm_model() when --llm-model is
                                      not given, to choose the translator's LLM
    ai/translate/ollama.py         -> OllamaTranslator(model=None) calls
                                      resolve_llm_model() during warmup()
    session/meeting_takeover_app.py-> same, for the takeover answer LLM

FLOW
----
    list_models()      GET /api/tags        what is installed
      -> rank_models()                      order by measured quality
        -> probe_model() each in turn       does it actually generate?
          -> ModelChoice                    the first one that works

Everything here talks to loopback only, so it does not weaken the local-only
guarantee (AC-17.2) -- exactly the same argument `ollama.py` already makes.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

#: Default Ollama endpoint. Loopback, therefore not network egress.
DEFAULT_HOST = "http://127.0.0.1:11434"

#: Preference order by BASE NAME, best first. Ranked on evidence, not vibes:
#:
#:   phi4-mini   measured CORRECT on the date-preservation fixture
#:               ("kal tak finish kar sakte hain" -> "by tomorrow")
#:   qwen3 / llama3.1 / mistral / gemma
#:               general instruction-tuned models; untested on this machine but
#:               ranked above coder builds because they are tuned for language
#:   *coder*     measured WRONG -- qwen2.5-coder:3b dropped the date entirely.
#:               Still usable as a last resort: a flawed translation the gate can
#:               inspect beats no translation at all.
#:
#: Keep this list short and evidence-led. Adding a model here is a claim that it
#: was tried, so record what was measured when you add one.
PREFERRED_MODELS: tuple[str, ...] = (
    "phi4-mini",
    "phi4",
    "qwen3",
    "llama3.1",
    "llama3",
    "mistral",
    "gemma3",
    "gemma2",
    "qwen2.5",
)

#: Base names we will use but warn about -- see the `coder` note above.
DISCOURAGED_SUBSTRINGS: tuple[str, ...] = ("coder", "code", "embed")

#: An unknown-but-installed model sorts here: after every known-good name, but
#: still ahead of the discouraged ones. Rationale: an unknown model that works is
#: strictly better than no LLM, which is the only alternative.
_UNKNOWN_RANK = len(PREFERRED_MODELS)
_DISCOURAGED_RANK = _UNKNOWN_RANK + 1


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """The outcome of resolution. Always returned; `name` is None on failure."""

    name: str | None
    host: str
    #: Every candidate that was considered, in the order they were tried.
    considered: tuple[str, ...] = ()
    #: candidate -> why it was rejected. Empty when the first pick worked.
    rejected: dict[str, str] = field(default_factory=dict)
    #: True when the chosen model is a coder/embedding build (see AC-22 E-13).
    discouraged: bool = False
    #: Wall-clock cost of the whole resolution, for AC-22.7 (<= 1 s budget).
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.name is not None

    def describe(self) -> str:
        """One line fit for printing to the terminal or the GUI log."""
        if not self.ok:
            if not self.considered:
                return ("no local LLM found -- start Ollama (`ollama serve`) and "
                        "pull one (`ollama pull phi4-mini`)")
            tried = ", ".join(f"{m} ({self.rejected.get(m, 'failed')})"
                              for m in self.considered)
            return f"no usable local LLM; tried {tried}"
        note = ""
        if self.discouraged:
            # Say it out loud: the user should know the model they have is the
            # one we measured getting dates wrong.
            note = ("  WARNING: coder-tuned models measured WRONG on dates; "
                    "`ollama pull phi4-mini` is the tested one")
        return f"{self.name} on {self.host} ({self.elapsed_ms:.0f} ms){note}"


def base_name(model: str) -> str:
    """Strip Ollama's ``:tag`` so ``phi4-mini:latest`` matches ``phi4-mini``."""
    return model.split(":", 1)[0].strip().lower()


def rank_of(model: str) -> int:
    """Lower is better. Used to sort what Ollama reports as installed.

    Matching is by SUBSTRING on the base name, not equality, because users
    routinely have derived models: this machine has
    ``rishi-phi4-mini-latest:latest``, which is a phi4-mini and should rank as
    one (AC-22.2). Equality matching would have ranked it "unknown".
    """
    name = base_name(model)
    if any(bad in name for bad in DISCOURAGED_SUBSTRINGS):
        return _DISCOURAGED_RANK
    for index, preferred in enumerate(PREFERRED_MODELS):
        if preferred in name:
            return index
    return _UNKNOWN_RANK


def is_discouraged(model: str) -> bool:
    return rank_of(model) == _DISCOURAGED_RANK


def list_models(host: str = DEFAULT_HOST, *, timeout: float = 3.0) -> list[str]:
    """Names of every model Ollama has, or [] if Ollama is not reachable.

    Deliberately returns [] instead of raising: "Ollama is not running" is an
    ordinary, recoverable state for this product (translation falls back to
    NLLB), not an exception. The caller distinguishes the cases by the empty
    list plus `ModelChoice.describe()`.
    """
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags",
                                    timeout=timeout) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError,
            json.JSONDecodeError, ValueError):
        return []
    return [entry.get("name", "") for entry in payload.get("models", [])
            if entry.get("name")]


def rank_models(models: list[str]) -> list[str]:
    """Best-first ordering. Stable within a rank, so `ollama list` order breaks ties."""
    return sorted(models, key=lambda m: (rank_of(m), models.index(m)))


def probe_model(model: str, host: str = DEFAULT_HOST, *,
                timeout: float = 20.0) -> str | None:
    """Generate two tokens. Returns None on success, else the reason it failed.

    Listing a model is not evidence it runs: it can be corrupt, too big for the
    GPU, or served by a daemon that is up but wedged. On a 4 GiB laptop GPU that
    difference is routine, so the probe is what AC-22.3 actually rests on.

    `keep_alive: 0` unloads immediately -- the same reasoning as
    `OllamaTranslator.keep_alive`: holding an LLM resident starves XTTS and
    pushes voice synthesis to the CPU.
    """
    body = json.dumps({
        "model": model,
        "prompt": "hi",
        "stream": False,
        "keep_alive": "0",
        "options": {"num_predict": 2, "temperature": 0.0},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return f"HTTP {response.status}"
            json.loads(response.read())    # must be parseable, not just 200
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except (TimeoutError, urllib.error.URLError) as exc:
        return f"timeout/unreachable ({exc})"
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return f"{type(exc).__name__}"
    return None


def candidate_models(*, host: str = DEFAULT_HOST, pinned: str | None = None,
                     max_candidates: int = 4) -> list[str]:
    """Best-first models to TRY, without loading any of them.

    This is the cheap half of resolution and the one callers should normally
    use. MEASURED on the development machine: 10 ms, against 8529 ms for
    `resolve_llm_model(probe=True)` -- because probing generates, which loads the
    model into VRAM, which `OllamaTranslator.warmup()` then has to do all over
    again after `keep_alive: 0` unloaded it.

    So the cheap ordering is produced here and the EXPENSIVE step doubles as the
    probe: `OllamaTranslator.warmup()` walks this list and the first model that
    warms up successfully is both the probe result and the loaded model. One
    load, real fallback (AC-22.3).

    A `pinned` model is returned alone, so an explicit `--llm-model` never
    silently falls back to something else (AC-22.5).
    """
    if pinned:
        return [pinned]
    return rank_models(list_models(host))[:max_candidates]


def resolve_llm_model(*, host: str = DEFAULT_HOST, pinned: str | None = None,
                      probe: bool = False, max_candidates: int = 4,
                      probe_timeout: float = 20.0) -> ModelChoice:
    """Pick the best installed LLM that actually generates.

    `pinned` -- the user passed `--llm-model`. An explicit choice is honoured
    exactly, and its ABSENCE is an error rather than a silent substitution
    (AC-22.5). This mirrors `registry.get()`, which also refuses to quietly
    downgrade an explicit request: if the user's two settings contradict each
    other, they should learn that, not get a different model without being told.

    `probe` defaults to **False** -- detection alone, measured at 10 ms. Turning
    it on generates two tokens per candidate to prove the model really runs,
    which costs a full model load each (measured 8529 ms for the first
    candidate on this machine). Use `probe=True` only where that cost buys
    something: a one-shot diagnostic like `vaani doctor`, never on the meeting
    start-up path, where `OllamaTranslator.warmup()` already performs the same
    load and can fall back on failure by itself.

    `max_candidates` caps how many models are considered so start-up stays
    bounded when someone has twenty models installed (AC-22.7).
    """
    started = time.monotonic()
    installed = list_models(host)

    def elapsed() -> float:
        return (time.monotonic() - started) * 1000.0

    if pinned:
        # Tolerate a missing ":latest" the same way OllamaTranslator.warmup does
        # -- that is a spelling detail, not a different model.
        match = next((m for m in installed
                      if m == pinned or base_name(m) == base_name(pinned)), None)
        if match is None:
            return ModelChoice(name=None, host=host,
                               considered=tuple(installed),
                               rejected={pinned: "not installed"},
                               elapsed_ms=elapsed())
        return ModelChoice(name=match, host=host, considered=(match,),
                           discouraged=is_discouraged(match),
                           elapsed_ms=elapsed())

    if not installed:
        return ModelChoice(name=None, host=host, elapsed_ms=elapsed())

    candidates = rank_models(installed)[:max_candidates]
    rejected: dict[str, str] = {}
    for candidate in candidates:
        if not probe:
            # Detection without probing, for tests and for `vaani doctor`, where
            # spending 20 s per model would make diagnostics unusable.
            return ModelChoice(name=candidate, host=host,
                               considered=tuple(candidates),
                               discouraged=is_discouraged(candidate),
                               elapsed_ms=elapsed())
        reason = probe_model(candidate, host, timeout=probe_timeout)
        if reason is None:
            return ModelChoice(name=candidate, host=host,
                               considered=tuple(candidates), rejected=rejected,
                               discouraged=is_discouraged(candidate),
                               elapsed_ms=elapsed())
        # AC-22.3: a failing candidate is skipped, never fatal.
        rejected[candidate] = reason

    return ModelChoice(name=None, host=host, considered=tuple(candidates),
                       rejected=rejected, elapsed_ms=elapsed())
