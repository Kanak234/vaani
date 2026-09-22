# 15 — Unified Meeting Mode, Takeover, and Model Auto-Detect

**Status:** specification, written before implementation.
**Supersedes:** nothing. **Extends:** [02-PRD](02-PRD.md), [03-TRD](03-TRD.md),
[11-MEETING-TAKEOVER](11-MEETING-TAKEOVER.md).
**Adds features:** F-21 (Meeting Takeover, surfaced), F-22 (Model auto-detect
and fallback).

---

## 0. Why this document exists

Before this change the repository contained **two separate meeting
implementations** that had drifted apart:

| | `vaani meeting` (`cli.py:cmd_meeting`) | `MeetingTakeoverApp` (`session/meeting_takeover_app.py`) |
|---|---|---|
| Reachable from the `vaani` CLI | yes | **no** — separate `argparse` main |
| Translation pipeline | yes | yes |
| Hidden overlay + dual transcript | yes | no |
| Control socket (`vaani hide`, `vaani panic`) | yes | no |
| Auto-routes meeting apps to the virtual mic | yes | no |
| Restores the real mic on exit / signal | yes | no |
| Remote-party capture (hears the other side) | **no** | yes |
| Takeover (answers in the user's voice) | **no** | yes |
| Used by the Windows console | no | yes |

A user therefore had to choose between *"my meeting app is wired up correctly and
I can panic-stop it"* and *"Vaani can speak for me"*. Neither choice is the
product. This document specifies one mode that is both.

It also specifies the model-selection behaviour, because the current defaults are
provably wrong on the development machine (see §4.1).

---

## 1. Verified starting state

Measured on the development machine on 2026-09-19, not assumed.

```
vaani doctor                                         10/11 pass
  [PASS] Virtual microphone      tone recovered at peak 0.496
  [PASS] Speech recognition      CPU; GPU present but unusable (no libcublas.so.12)
  [SKIP] Translation round-trip  NLLB model not downloaded
  [PASS] Text to speech          fallback voice, not the user's
  [PASS] Voice profile           'me': 55 s speech, SNR 22.2 dB
pytest                                               332 passed, 17 skipped
ollama list      phi4-mini:latest, qwen2.5-coder:3b, rishi-phi4-mini-latest,
                 rishi-qwen2.5-coder-3b
```

Four defects were reproduced, not inferred:

- **D-1** `cli.py:564-572` rebinds `translator` to the `RoutingTranslator`
  *inside* the `try`, so the `except` branch's "NLLB only" recovery re-warms the
  object that just failed. Reproduced: the handler raises during handling of the
  first exception and `vaani meeting` dies with a traceback.
- **D-2** `cli.py:653` (`_bail`) closes over `router`, which is not assigned
  until `cli.py:692`. A `SIGINT` during model warm-up raises `UnboundLocalError`
  inside the signal handler, so the microphone is never restored.
- **D-3** Three entry points default to three different Ollama models
  (`qwen3-coder:latest`, `qwen3:8b`, `phi4-mini:latest`). Two of them are not
  installed, and the repo's own measurement ([ollama.py:33-38](../src/vaani/ai/translate/ollama.py))
  records the coder-tuned variants as *wrong* on dates.
- **D-4** The `translate` extra declares `transformers` + `sentencepiece` but
  NLLB requires **torch**, which only the `voice` extra pulls. The README's own
  Debian quickstart therefore produces an install where translation cannot work.

---

## 2. F-21 — Meeting Takeover, surfaced

### 2.1 Behaviour

One command runs the whole thing:

```
vaani meeting --takeover --remote-input <monitor-source>
```

`--takeover` is **opt-in**. Without it `vaani meeting` behaves exactly as it does
today: translation only, no remote capture, no generated speech. This preserves
the existing contract for anyone who just wants translation.

With `--takeover`, in addition to translation:

1. A second capture stream opens on `--remote-input` (the meeting's output, i.e.
   a PipeWire **monitor** source) and transcribes the other participants.
2. Utterances that `looks_like_question()` accepts are handed to the
   `MeetingTakeoverController` as a pending question.
3. The controller grants the floor only under the policy in
   [11-MEETING-TAKEOVER](11-MEETING-TAKEOVER.md): armed, pending question, and
   either repeated hesitation past the silence window or double that window of
   silence — or the explicit phrase *"Vaani, take over"*.
4. When granted, `AnswerAssistant` drafts an answer with the local LLM, it is
   synthesised in the user's enrolled voice, and injected via
   `TranslationSession.inject_audio()` — the same output queue the emergency stop
   flushes.

### 2.2 States

| State | Meaning | Reached by | Overlay |
|---|---|---|---|
| `OFF` | Vaani never generates speech | default; `vaani takeover-stop` | no badge |
| `ARMED` | watching for a pending question | `--takeover --arm`, or `vaani takeover-arm` | `TAKEOVER ARMED` |
| `ACTIVE` | Vaani holds the floor | policy grant, or explicit phrase | `TAKEOVER ACTIVE` |

`--takeover` alone starts **OFF with remote capture running**. Arming is a second,
separate act. Rationale: opening the remote stream is observable and harmless;
authorising generated speech is neither, so it must not be a side effect of a
flag that also does something benign.

### 2.3 Control-socket vocabulary (new)

`ipc/control.py:COMMANDS` gains three entries, wired to the same KDE-shortcut
pattern as the existing ones:

| Command | Effect |
|---|---|
| `takeover-arm` | `OFF → ARMED`. No-op when already armed. |
| `takeover-stop` | `* → OFF`, clears the pending turn. Always succeeds. |
| `takeover-now` | `ARMED → ACTIVE` immediately, without waiting for the policy. No-op (returns `armed: false`) when `OFF`. |

`panic` continues to win over all three: it emergency-stops the session, which
flushes queued audio including a takeover answer mid-sentence.

### 2.4 Edge cases — each must be handled explicitly

| # | Case | Required behaviour |
|---|---|---|
| E-1 | `--takeover` without `--remote-input` | refuse at argument-parse time with a message naming `vaani devices --all` |
| E-2 | `--remote-input` is the same device as `--input` | refuse: it would transcribe the user as the remote party |
| E-3 | remote stream dies mid-meeting | log once, keep translation running; takeover degrades to explicit-phrase only |
| E-4 | no LLM reachable | `--takeover` refuses to arm and says why; translation still runs |
| E-5 | no enrolled voice / no consent | `--takeover` refuses; a takeover answer in the *fallback* voice is never emitted (it would be an unrecognisable robot claiming to be the user) |
| E-6 | user starts speaking while `ACTIVE` | `observe_user_speech` with substantive text returns `USER`; the in-flight answer is **not** retracted, but no further answer is generated until the policy grants again |
| E-7 | question arrives while the user is mid-utterance | queued as pending; the silence window is measured from the later of question time and last user activity (already in `takeover.py:165`) |
| E-8 | `takeover-arm` when the session is `PAUSED` | allowed; nothing is generated while paused because `can_emit_audio` is false |

### 2.5 Acceptance criteria

| ID | Criterion | How it is checked |
|---|---|---|
| AC-21.1 | `vaani meeting --takeover` runs translation **and** takeover in one process with one control socket | integration test asserts both loops start |
| AC-21.2 | Takeover is OFF until explicitly armed | test: `--takeover` alone leaves `controller.state is OFF` |
| AC-21.3 | `takeover-stop` returns to OFF from any state and clears the pending question | unit test over all three states |
| AC-21.4 | `panic` flushes a takeover answer that is already playing | test: inject, panic, assert out-queue drained |
| AC-21.5 | E-1, E-2, E-4, E-5 each refuse with a distinct, actionable message | one test per case |
| AC-21.6 | A takeover answer is never synthesised in the fallback voice | test asserts `is_fallback_voice` path raises |
| AC-21.7 | `vaani takeover` remains available as a standalone entry point | test invokes the parser |

### 2.6 Latency budget for takeover

Takeover is **not** on the translation hot path and has a separate, looser
budget, because the trigger is already several seconds of silence.

```
remote STT        (Whisper, same model as the user path)     ~600 ms
answer draft      (local LLM, ~200 tokens)                  ~2000 ms
synthesis         (XTTS-v2, user's voice)                   ~1200 ms
---------------------------------------------------------------------
takeover response                                           ~3800 ms
budget                                                      6000 ms
```

A breach warns; it does not suppress. Silence after a question is already the
failure state takeover exists to fix, so a late answer still beats none.

### 2.7 The honesty boundary — flagged, not resolved silently

[`ai/assist.py`](../src/vaani/ai/assist.py) states in its module docstring that
the autonomous version of this feature "is not built here", and
`synthesize_if_approved()` refuses to speak an unapproved draft. Takeover
deliberately crosses that line: `takeover_runtime.py:81` sets `draft.approved =
True` without a human in the loop.

**That is intended and is the point of F-21** — the owner asked for it, it is
their own cloned voice, and it is local-only. But the reasoning in `assist.py`
(participants cannot tell a machine's commitment from the user's) does not stop
being true. So this specification requires the mitigations, not the approval:

- takeover must be **armed deliberately**, never on by default (AC-21.2);
- `stop` must be **unconditional and immediate** (AC-21.3);
- the overlay must **show the state** whenever it is not OFF (§2.2);
- every takeover answer is **printed with its trigger reason**, so the user can
  audit afterwards what was said for them and why.

`assist.py`'s docstring must be amended to point here rather than claiming the
autonomous path does not exist, because it now does.

---

## 3. F-22 — Model auto-detect and fallback

### 3.1 Behaviour

At start-up, for the LLM slot, Vaani resolves a model instead of trusting a
hardcoded default:

1. Ask Ollama what is installed (`GET /api/tags`).
2. Score each installed model against a **preference list** ordered by the
   repo's own measurements (§3.2).
3. Pick the highest-scoring installed model; **probe** it with one tiny
   generation.
4. If the probe fails, drop it and try the next candidate.
5. If nothing survives, report `no usable LLM` and let the caller degrade —
   translation falls back to NLLB, and takeover refuses to arm (E-4).

The user can still pin a model with `--llm-model`. An explicitly pinned model
that is absent is an **error, not a silent substitution** — the same principle
`registry.py:84` already applies to local-only violations.

### 3.2 Preference order, and why

| Rank | Model | Evidence |
|---|---|---|
| 1 | `phi4-mini` | [ollama.py:33-38](../src/vaani/ai/translate/ollama.py): correct on the date-preservation case |
| 2 | `qwen3` / `llama3.1` / `mistral` | general instruction-tuned; untested here, ranked above coder builds on priors |
| 3 | `qwen2.5-coder`, `*-coder*` | measured **wrong** (dropped a date); last resort |

Matching is on the model's **base name**, so `rishi-phi4-mini-latest:latest`
matches rank 1. Unknown models are ranked after all known ones rather than
excluded — an unknown model that works beats no translation.

### 3.3 Provider-level fallback

`ProviderRegistry.get()` currently instantiates `usable[0]` and lets a failing
factory propagate ([registry.py:107](../src/vaani/ai/registry.py)). It gains a
`get_with_fallback()` that walks candidates in priority order, collecting
failures, and raises only when every one fails — with all the failures in
`detail`. The strict `get()` is unchanged, because the local-only tests depend on
its exact behaviour.

### 3.4 Edge cases

| # | Case | Required behaviour |
|---|---|---|
| E-9 | Ollama not running | detection returns empty; caller degrades; message names `ollama serve` |
| E-10 | Ollama running, zero models | distinct message naming `ollama pull phi4-mini` |
| E-11 | model listed but the probe times out | dropped, next candidate tried, reason recorded |
| E-12 | `--llm-model` names an absent model | hard error listing what *is* installed |
| E-13 | only coder models installed | used, with a printed warning that they measured wrong on dates |

### 3.5 Acceptance criteria

| ID | Criterion | How it is checked |
|---|---|---|
| AC-22.1 | With several models installed, the measured-best one is chosen | unit test with a stubbed `/api/tags` |
| AC-22.2 | A base-name match wins (`rishi-phi4-mini-latest` → rank 1) | unit test |
| AC-22.3 | A model that fails its probe is skipped, not fatal | unit test |
| AC-22.4 | Zero usable models degrades, never crashes the session | unit test + D-1 regression test |
| AC-22.5 | An explicitly pinned absent model is a hard error | unit test |
| AC-22.6 | `registry.get_with_fallback` tries every candidate before raising | unit test |
| AC-22.7 | Auto-detection adds ≤ 1 s to start-up with 4 models installed | measured, printed by the verify script |

---

## 4. Defect fixes specified

| ID | Fix | Acceptance |
|---|---|---|
| D-1 | Keep the fast engine in its own name; the recovery path warms *that*, never the router | AC-22.4 regression test asserts the function returns instead of raising |
| D-2 | Initialise `router = None` before the signal handlers are installed | test installs handlers, calls `_bail` early, asserts no exception |
| D-3 | One shared default, resolved by F-22, used by all three entry points | test asserts the three parsers agree |
| D-4 | Move `torch`/`torchaudio` into the `translate` extra (NLLB needs them) and leave `coqui-tts` in `voice` | `pip install -e ".[stt,translate]"` then import NLLB in CI |

### 4.1 Risk of the D-4 change

Moving torch into `translate` makes `pip install -e ".[stt,translate]"` download
~2.5 GB where it previously downloaded ~200 MB. That is a real regression in
install size for anyone who wanted a text-only build — but the alternative is the
current state, where that install *silently cannot translate*. Documented in the
README rather than hidden.

---

## 5. What this change does **not** do

- It does not change the translation pipeline, the gate, the state machine, the
  virtual-microphone topology, or the Windows audio path.
- It does not make takeover the default, or arm it automatically.
- It does not add any network dependency. Ollama stays on loopback.
- It does not touch `main`; work lands on `feature/meeting-unification`.

---

## 6. Verification plan

| Claim | How it will be verified | Can it be verified here? |
|---|---|---|
| Unified mode starts both loops | integration test with stub engines | yes |
| Control commands drive takeover | unit test against the handler | yes |
| Auto-detect picks phi4-mini on this box | run against the real Ollama | yes |
| Takeover answer reaches the virtual mic | inject a known tone, capture from the virtual mic | yes |
| End-to-end in a real Zoom/Meet call | — | **no.** Requires a second human on a real call. Will be stated as unverified. |
| XTTS sounds like the owner | — | **no.** Subjective; only the owner can judge. |
