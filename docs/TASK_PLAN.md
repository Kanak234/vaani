# Vaani v1.0 — Task Plan (resume point)

**Branch:** `v1-complete` (from `feature/meeting-takeover` @ b387f75).
**Rule for any fresh session:** read this file, then PRD.md / TRD.md /
UI_UX_BRIEF.md, and continue from the first unchecked box. Don't re-audit.
Mark `[x]` only when verified, and put the evidence in brackets.
`[~]` = code done, real-hardware check pending (UNVERIFIED).

## Phase 0 audit findings (2026-09-22, cloud workspace, CPU, no audio device)

Verified:
- Tests: main 266 pass; feature 317 pass; 1 fail on both = `tests/unit/test_ui.py`
  needs tkinter (missing in the audit environment only).
- ruff: 31 issues; one real bug: `ui/app.py:560` F821, `exc` used in a lambda
  after the except block → NameError on the "test voice" error path.
- `--llm-model` hardcoded default `qwen3-coder:latest` (cli.py `meeting`, `run`);
  `ai/assist.py` default the same; `windows_console.py` hardcodes a model list.
- Missing: push-to-talk, bypass (real voice), offline script mode, mid-session
  fallback, one-step install (`packaging/install.sh` needs a pre-built venv and
  downloads no models), takeover "continue my point".
- The existing takeover (feature branch) = answer the *remote* person's question; it is
  Windows-first (`windows_runtime.py`) and not wired into the Linux `meeting` command.
- The push to GitHub is blocked from the cloud session (repo not in the session's
  authorised sources).

Unverified (from repo docs only): measured latencies, virtual-mic round trip, XTTS
VRAM figures. They have never been run on Kanak's own voice. The 60-min soak has never been run.

## Phase 0 re-audit (2026-09-22, Kanak's HP Victus, local)

The cloud audit above ran on a CPU-only box with no GPU and no audio device.
Re-checked here on the real laptop. Corrections:

Verified on this machine:
- **CUDA works.** `torch.cuda.is_available() == True`, RTX 3050 Laptop 4 GB,
  torch 2.14.0+cu130 on Python 3.14.4. The brief's assumption that the
  proprietary driver might be missing does NOT hold here. CPU fallback is
  therefore still UNVERIFIED - it has not been forced and exercised.
- **Suite: 318 passed, 0 failed.** The cloud run's `test_ui.py` failure was an
  environment artifact (no tkinter there); tkinter is present here.
  The `test_control.py` failure was caused by the uncommitted takeover verbs
  and is now fixed.
- Deps import cleanly: faster_whisper 1.2.1, TTS 0.27.5, transformers 4.57.6.
- Ollama installed models here: `phi4-mini:latest`, `qwen2.5-coder:3b`,
  `rishi-phi4-mini-latest`, `rishi-qwen2.5-coder-3b`.
- **`ai/assist.py:123` still hardcodes `model="qwen3-coder:latest"`, which is
  NOT installed on this machine.** cli.py's hardcoded default was removed by the
  recovered WIP; assist.py's was missed. Live bug - Phase 3.
- ~3 GB of untracked build output (`ci_artifacts/`, `release/windows/`) was
  sitting in the tree. Now gitignored, never committed.

Could NOT verify here:
- The cloud audit's "ruff: 31 issues, F821 at ui/app.py:560". **ruff is not
  installed in this venv**, so that claim is neither confirmed nor refuted.
  Install ruff before ticking the Phase 1 lint box.
- `silero_vad` and `sounddevice` do not import as top-level modules. Believed
  intentional (torch.hub VAD; ctypes libpulse backend) but NOT confirmed.

## Phase 0 — Audit & docs
- [x] Inspect repo, both branches, tests, lint
- [x] PRD.md, TRD.md, UI_UX_BRIEF.md, TASK_PLAN.md
- [x] Decisions D1–D4 recorded (PRD §10)
- [ ] Kanak confirms the proposed defaults (PRD V-7 thresholds, TRD O1–O3, hotkeys)

## Phase 1 — Core pipeline hardening
- [x] ruff installed; F821 in ui/app.py fixed (deferred lambda lost `exc` after the
      except block → NameError swallowed the voice-test error). Bug class swept:
      B023 clean, other 8 deferred lambdas safe. Lint gate added to pyproject
      (select F/E9/B), green across src, tests and scripts.
- [x] `core/supervisor.py`: exception + OOM containment, sliding CrashWindow,
      latched degraded mode. 23 tests, injected clock. **Scope correction:** hard
      stage *timeouts* are NOT implemented and cannot be in-process — Whisper and
      XTTS sit in native calls CPython cannot preempt. run() reports duration for
      post-hoc budget checks; real preemption needs a subprocess (see backlog).
- [x] Bounded queues + drop counters in session/engine.py + 17 tests.
      `_out_queue` was unbounded; now bounded drop-oldest with a counter.
- [x] Worker thread supervision. **Scope correction:** pipeline.process() already
      caught stage exceptions, so the real gap was the five steps AFTER it (db
      write, audio injection, latency record, result callback, state transition)
      which ran bare and killed the worker thread → session alive but deaf.
      All five now run inside the supervisor. 3 crashes/60 s → degraded mode.
- [x] **Memory leak found and fixed:** `session.results` was an unbounded list and
      every UtteranceResult holds its synthesised audio. Now a bounded deque
      (default 200) + lifetime counters. Also made latency_percentiles() genuinely
      rolling, which AC-13.2 specified and the unbounded list never satisfied.
      Consumers updated: cli.py counters, and meeting_takeover_app's index slicing
      replaced with eviction-safe `new_results_since(seq)`.
- [x] Rotating log file. **It existed but was dead code** — setup_logging() was
      never called from anywhere, and the codebase had exactly ONE logging call.
      Now wired into cli.main() behind a global --log-level, with session
      start/stop counters and stage-failure tracebacks. Verified: structured JSON
      lands in ~/.local/share/vaani/logs/vaani.log.
- [x] Soak harness: `scripts/soak_test.py` (RSS sampling, median-window pass rule,
      per-sample checkpointing because the desktop crashes under sustained load).
      **Measured here:** 360 utterances, RSS flat 49.9 → 49.5 MiB, **-0.88%**
      growth, report in release/soak_capped.json. The same harness before the cap
      showed linear growth of ~0.36 MiB per utterance — that was the leak.
- [~] verify_phase1.sh: real 30-min soak with REAL providers on the laptop
      (Kanak runs). The run above used stub providers on purpose: it proves the
      session machinery does not leak, not that Whisper/XTTS do not.

## Phase 2 — Meeting mode
- [x] `session/input_mode.py`: pure state machine, no audio/threads. 31 tests,
      including an exhaustive parametrised sweep proving translation and
      passthrough are NEVER both enabled (two overlapping voices in the meeting).
      Bypass is a flag, not a fourth mode, so releasing it restores the previous
      state exactly.
- [x] Bypass path in engine: raw frames → virtual mic, VAD/STT/MT/TTS skipped
      entirely, and entering bypass flushes queued TTS so a half-spoken AI
      sentence cannot play under the user's real voice. 14 routing tests driving
      _capture_loop with a fake capture stream. Mute correctly outranks both
      bypass and a held talk key.
- [x] CLI + IPC: `bypass`, `bypass-on/off`, `talk`, `mode-always-on|ptt-toggle|
      ptt-hold`; `--input-mode` at startup; closed COMMANDS vocabulary updated.
      **Verified against a live session**, effects confirmed in the log.
- [x] Overlay: `set_input_mode()` badge (LIVE / BYPASS / MUTED), BYPASS in red
      because the room is hearing untranslated Hindi. `status` now prints the
      input mode with an explicit warning line.
- [x] Virtual mic cleanup — **VERIFIED ON REAL HARDWARE, not assumed.**
      Live meeting session: 2 PipeWire modules → 0 after SIGTERM, process exited
      in 4 s, and 2 Microsoft Edge streams were restored to the real microphone.
      Create/reuse/destroy separately verified: second create reused rather than
      stacking, 0 modules left behind.
- [x] `system/evdev_ptt.py` optional hold-to-talk. 17 tests against fake events,
      so the press/release/autorepeat rules are covered without evdev, without
      /dev/input and without root. Wired into `vaani meeting --input-mode
      ptt_hold --ptt-key KEY_F8`.
      **Cannot be exercised on this machine:** `evdev` is not installed AND the
      user is not in the `input` group (a privileged `usermod` + re-login).
      What IS verified live: the degradation path names the missing prerequisite,
      prints the remedy, and falls back to ptt_toggle rather than leaving the
      user in a mode whose key nothing reads.
      Privacy note in the module: reading /dev/input means seeing every keystroke
      on the machine, so it matches one key and never stores or logs a key code.
- [x] **Live end-to-end verification on real hardware** (not a unit test):
      MUTED → `vaani talk` → LIVE → `vaani bypass-on` → BYPASS, each confirmed
      via `vaani status` and the session log; SIGTERM left 0 PipeWire modules and
      restored 2 Microsoft Edge streams to the real mic.
- [~] verify_phase2.sh + manual Meet call with a human on the other end (Kanak).
      Edge was already consuming the virtual mic during testing, which is
      encouraging but is NOT the same as a real Meet call being understood.

## Phase 3 — Auto model detection
- [x] CUDA "really works" probe (already existed: verifies by real inference,
      not by device count) + **VAANI_FORCE_CPU added**. The CPU-fallback branch
      was unverifiable on this laptop because CUDA works; the env override makes
      it reachable. Verified it cascades: forcing CPU steps Whisper medium→small.
- [x] Ranking + reasons in `ai/model_detect.py`. **35 tests added** — it shipped
      with none, despite deciding which LLM translates the user's speech.
      `bench_cache.py` still missing (ranking is static + evidence-led, so a
      benchmark cache has not been needed yet).
- [x] Hardcoded model defaults removed. `ai/assist.py` defaulted to
      `qwen3-coder:latest` (not installed on either machine, and a coder build —
      the category measured getting dates wrong); `OllamaTranslator` defaulted to
      `phi4-mini:latest`. Both now None = auto-detect.
- [x] **REAL BUG found by `vaani models`:** auto-detection picked
      `rishi-phi4-mini-latest` over `phi4-mini`. `ollama show --system` reveals
      that custom build carries "Reply in the same language he uses (English or
      Hindi)" and temperature 0.6. Vaani must ALWAYS emit English (AC-06.3), so
      it would produce Devanagari, the gate would suppress it, and the user would
      watch utterances vanish. Fixed with an `is_derived()` tie-break: same rank,
      base model first. (The first attempt at is_derived was itself wrong —
      `phi4-mini` matched the separate entry `phi4` and flagged itself derived —
      pinned by a regression test.)
- [x] **Mid-session fallback implemented** (the acceptance criterion). The
      subtlety: a broken translation engine never RAISES — pipeline.process()
      catches stage exceptions and returns a suppressed result — so the
      supervisor never saw it and the meeting would degrade into silence with
      nothing swapping the dead provider out. Per-stage CrashWindows now watch
      SUPPRESSED RESULTS, and `session.swap_translator()` performs the swap.
      Ladder: routing (NLLB+LLM) → NLLB alone. No further rung on purpose —
      passthrough would put untranslated Hindi into an English meeting.
      **Verified live:** RoutingTranslator → Nllb after 3 translate failures.
      (Regression test pins the direction: the router is built
      `RoutingTranslator(fast=nllb, accurate=llm)`, and the first version of the
      ladder reached for `accurate` — i.e. degraded TO the engine that had just
      failed three times.)
- [x] `ai/registry.get_with_fallback()` now has 7 tests. Its docstring claimed
      "Used by: cli.py meeting/run start-up" but **nothing called it** —
      `_build_translator` solves start-up degradation differently and better
      (it composes a router rather than picking one provider). Left in place and
      tested rather than deleted; the false docstring claim is the thing to fix
      if it stays unused.
- [x] `vaani models`: GPU status, Whisper model per performance mode with
      download state, ranked Ollama list with DISCOURAGED flags, and the chosen
      model with its reasoning. Auto-detection is only trustworthy if its
      reasoning is inspectable — and this command is what exposed the derived-
      model bug above within a minute of existing.
- [~] verify_phase3.sh (GPU run + FORCE_CPU run) on the laptop. NOTE: a real
      FORCE_CPU meeting run will first download Whisper `small` (~460 MB), which
      is not yet cached here.

## Phase 4 — Takeover
- [x] `session/continuation.py` (Takeover A: finish the USER'S OWN point).
      36 tests. Most of them assert it REFUSING to speak, because the risk here
      is being wrong about what the user THINKS, in their own cloned voice, to
      their colleagues. Refuses when: not armed, context < 40 chars, the
      sentence already landed (terminal punctuation), the other participant
      spoke last, before the 4 s stall threshold, or the model returns PASS.
      The unfinished-ending word list is deliberately conservative — a MISSING
      word means Vaani stays quiet, which is the safe direction. It initially
      missed the most ordinary English stall there is ("...is going to be");
      auxiliaries, modals and prepositions were added after the tests caught it.
- [~] Takeover B: `session/remote_listener.py` + cli `takeover` + ipc verbs +
      `overlay.set_takeover()` landed; end-to-end Linux meeting wiring unproven
- [x] `cancel` ≤200 ms — **MEASURED**, `tests/manual/measure_takeover_cancel.py`:
      | what | p50 | worst | budget |
      |---|---|---|---|
      | cancel() returns | 0.001 ms | 0.006 ms | 200 ms |
      | cancel during a live 2 s draft | — | 0.017 ms | 200 ms |
      | queued AI audio dropped | 0.099 ms | 0.132 ms | 200 ms |
      And the case that matters most, asserted rather than timed: a draft in
      flight when cancel fires is NEVER spoken. A generation counter handles the
      nastier variant — re-arming before a cancelled draft returns must not let
      the stale text through.
- [x] Overlay AI-SPEAKING indicator that **cannot be dismissed while true**.
      Found while implementing: `hide()` withdrew the window entirely, so
      `vaani hide` during an active takeover left the user with their own cloned
      voice in a meeting and NOTHING on screen saying so. Now `set_takeover
      ("active")` force-shows a hidden overlay and `hide()` refuses while
      active. ARMED deliberately does NOT force-show — it means it COULD speak,
      and forcing the overlay up for that would make the indicator meaningless
      through overuse. 4 UI tests.
- [x] CLI/IPC: `continue`, `continue-arm`, `continue-off`, `cancel`, wired into
      `vaani meeting` with a dedicated 500 ms stall poll (the existing 3 s tick
      would fire 4–7 s after the user trailed off, by which point Vaani would be
      interrupting whatever came next). Drafting runs off the Tk thread, because
      freezing the overlay would freeze the AI-SPEAKING badge and the cancel
      path — exactly when both are needed.
      **Verified live:** `continue` → "declined: not enough context to continue
      a point"; `continue-arm` → "continuations ARMED"; `cancel` → "cancelled".
      Commands that printed nothing were fixed: silence is indistinguishable
      from a broken feature, and for `continue` the honest outcome is usually a
      refusal.
- [~] Manual: speaks in HIS voice — **NOT VERIFIED and not verifiable here.**
      Needs (a) Kanak speaking Hindi/Hinglish into the mic and (b) an enrolled
      XTTS voice profile. What IS verified is the decision logic, the command
      surface, the cancel budget and the indicator. What is NOT is whether the
      continuation sounds like him or says something he would actually say —
      which is the whole point of the feature.

## Phase 5 — Offline script mode
- [x] `session/script_mode.py` + chunk checkpoints. 33 tests. Offline inverts
      every live trade-off on purpose: nothing is dropped, low-confidence text is
      KEPT and marked `[?]` rather than suppressed (the user is going to read and
      edit it anyway), and progress is checkpointed after EVERY chunk because the
      target desktop crashes under exactly this kind of long sustained load.
      Chunks cut at the quietest point near the target rather than on a fixed
      window, which would corrupt one word per window with nothing able to tell.
- [x] `vaani script FILE | --record [--audio] [--out] [--source-out]`.
      Handles 44.1 kHz stereo input (resample + mono-by-averaging), because a
      phone recording passed through unchanged does not error — it just
      transcribes badly, which is far harder to diagnose than a crash.
- [x] Fixture check — 2 new integration tests run the fixtures through
      ScriptJob's own path with the REAL NLLB translator, and assert no
      Devanagari survives into the script (AC-06.3 applies to the written script
      too). 19 integration tests pass.
- [x] **Verified end to end on a real file**, which is what caught both bugs
      below: 13 s → 4 chunks → script written, resume confirmed
      ("resumed from chunk 18", 0 s of work instead of ~30 s), and the checkpoint
      correctly retained after a failed run so it stays resumable.
- [~] Kanak's own recording — NOT DONE and not doable here. There is no TTS on
      this machine (no espeak/festival/flite) that can produce intelligible
      Hindi, so the test file was formant tones. Whisper HALLUCINATED
      "Thanks for watching!" on it — its known failure on unintelligible audio.
      Worth knowing: all four chunks were marked `[?]`, so the low-confidence
      marker caught the hallucination. STT quality on real Hindi speech remains
      entirely unverified.

## Phase 6 — UI + packaging
- [x] Root `install.sh`, idempotent. **VERIFIED by actually doing it:** cloned
      the repo to a clean directory with no venv, ran `./install.sh --no-voice`,
      and `vaani models` worked from the fresh install with no manual steps —
      acceptance criterion A1 met. Re-running took **11 s** (reused the venv,
      skipped both already-downloaded models, exit 0).
      It calls `packaging/install.sh` rather than replacing it: that script does
      the narrower desktop-integration job and refuses to run without a venv,
      which is correct for what it is.
      NOTE: running install.sh from a second checkout repoints
      `~/.local/bin/vaani` at that checkout. Correct behaviour, but it caught me
      out during testing and had to be restored by hand.
- [x] `vaani fetch-models` — downloads Whisper (all sizes the modes use), NLLB
      and optionally XTTS up front. This is the real fix for the silent
      first-run problem found in Phase 2; the notice added then was a mitigation.
      Idempotent and resumable.
- [x] Dashboard input-mode switch (always_on / ptt_toggle / ptt_hold, chosen
      BEFORE Start and applied to the session it creates; also pushed to a
      running session, which accepts a mid-session switch).
- [x] Model status: the Models readiness card showed only device placement
      ("stt=cuda"), which cannot answer the question the user actually has —
      "what is it using, and is that the good one?". Now reads e.g.
      `stt=cuda, translation=cuda, tts=cuda · whisper medium · llm
      phi4-mini:latest · 3684 MiB VRAM free`, and the card asks for attention
      when the chosen LLM is a DISCOURAGED coder/embed build.
- [x] Script screen: offline script mode had a CLI and no GUI at all. File
      picker, progress per chunk, the script shown inline and saved next to the
      recording, with `[?]` passages called out. Runs off the Tk thread —
      transcription takes minutes, and freezing the window would freeze the
      progress label that is the only sign it has not died.
      6 new UI tests; the existing "all five screens" test is now an exact
      six-screen check of BOTH `_screens` and `_nav_buttons`, because a screen
      missing from the rail is unreachable and a rail entry with no screen
      crashes on click.
- [x] README rewritten for setup + troubleshooting. Every troubleshooting entry
      is a failure that ACTUALLY happened during development, not a guess:
      the silent model download, the meeting app staying on the real mic, MUTED
      vs BYPASS, the custom Ollama model whose system prompt defeats translation,
      `fix-audio` after a kill -9, the evdev/input-group requirement, Whisper's
      "Thanks for watching!" hallucination, and CPU fallback.
      The Use section previously told users to type
      `PYTHONPATH=src .venv/bin/python -m vaani.cli ...`, which is stale once
      install.sh puts `vaani` on PATH. Rewritten, and **all 19 documented
      commands were verified to exist** by invoking each one.
- [x] Comment pass MEASURED rather than claimed. Files touched in v1 run a
      comment:code ratio of 0.18–0.32 (`core/gate.py` 0.32, `session/engine.py`
      0.21, `session/continuation.py` 0.21, `ipc/control.py` 0.19,
      `ai/translate/ollama.py` 0.18) against a repo median of 0.07.
- [ ] Remaining rule-4 backlog, named not glossed: the Windows modules (out of
      v1 scope per D1) plus `system/platform.py`, `session/takeover.py`,
      `system/ollama_manager.py`, `system/hardware.py`, `ui/components.py`,
      `devices/routing.py`, `core/model_budget.py`, `core/state_machine.py`
      (all 0.00–0.03).
- [x] Fresh-clone install verified on the laptop (see above). A dedicated
      verify_phase6.sh is still not written; the manual run is recorded here
      with its results instead.

## Phase 7 — Final verification
- [x] Full suite + lint: **588 passed**, ruff clean across src/tests/scripts,
      install.sh syntax clean.
- [x] Checklist A1–A8 with evidence → **docs/16-V1-ACCEPTANCE.md**.
      4 PASS (A3 with a memory caveat), 4 PARTIAL. Every PARTIAL is partial
      because it needs Kanak's voice, a second human, or the other machine —
      none because of unfinished code.
- [x] "What could NOT be verified" collected in one place at the end of
      docs/16-V1-ACCEPTANCE.md (6 items).
- [x] **The 30-minute soak found two real bugs that 583 passing tests had not:**
      (1) `STT_EMPTY` from an idle microphone was counted as a provider failure,
      so a SILENT ROOM looked like a broken speech engine — in a real meeting
      with background noise Vaani would have declared STT degraded within the
      first minute; (2) "requesting fallback" fired ~20 times in 4 minutes for a
      stage with no fallback configured. Both fixed with regression tests.
      Memory: +12.03% before the fixes (FAIL vs the ≤10% rule), +2.93% after.
- [~] A clean 30-minute run AFTER the fixes. Run 2 was 20 minutes, so the
      +2.93% is not a like-for-like replacement for run 1's number.

## Backlog raised during Phase 5
- **Whisper hallucinates confident boilerplate on unintelligible audio**
  ("Thanks for watching!", "please subscribe"). Observed directly. The `[?]`
  low-confidence marker caught it here, but a dedicated hallucination check
  against that known phrase set would be stronger. Not implemented.
- Script mode's two bugs both passed every unit test and were caught only by
  running the command for real. The stub translator accepted any
  `source_language` string; the real one rejects `auto`. Lesson recorded:
  provider-contract bugs need an integration test with the real provider.

## Backlog raised during Phase 2
- README must document the two manual steps for ptt_hold: `pip install evdev`
  and `sudo usermod -aG input $USER` + re-login. Phase 6.
- **First run is silent for minutes.** `vaani meeting` prints "Starting Vaani in
  meeting mode..." then downloads the Whisper model (~1.5 GB for `medium`) with no
  progress reaching the terminal. It was mistaken for a hang during testing —
  a first-time user would kill it. Mitigated with an explicit "first run:
  downloading..." notice (`FasterWhisperRecognizer.is_cached()`); the real fix is
  pre-fetching models in install.sh (Phase 6).
- Measured on this machine once warm: STT resolves to **medium on cuda**, and the
  virtual mic appears ~18 s after launch.

## Backlog raised during Phase 1
- Hard stage timeouts need a subprocess worker; in-process preemption of a native
  Whisper/XTTS call is not possible. Revisit if a hang is ever observed in practice.
- `ai/assist.py:123` still hardcodes `qwen3-coder:latest`, which is NOT installed
  on this machine. Phase 3.
- Style lint families (E4/E7, I, UP, SIM) are ~385 findings, concentrated in the
  out-of-v1-scope Windows console. Not gated; not pretended to be clean.
- `tests/unit/test_ui.py` intermittently emits a tkinter `Variable.__del__`
  teardown warning ("main thread is not in main loop"). GC-timing artifact, not a
  failure; appears in some runs only.

## Comment backlog (rule 4)
Existing files not yet brought to the rule-4 comment standard: all of `src/vaani/**`
(to be ticked off as each is touched).

## Session log
- 2026-09-22 Phase 0 done in a cloud session. Commits are local; delivered as a git bundle.
- 2026-09-22 Local session on the Victus. Recovered the cloud session's bundle
  (it had never reached the repo) into branch `v1-complete`, then committed
  ~1430 lines of Phase 3/4 work that was sitting uncommitted and unlost.
  Suite green at 318. Phase 0 still blocked on Kanak confirming defaults.
- 2026-09-22 Phase 1 complete on the Victus. Found and fixed a real memory leak
  (unbounded results list) and a real crash path (unguarded post-pipeline steps),
  plus the F821 NameError. Rotating logs were dead code; now wired. Soak measured
  at -0.88% RSS growth over 360 utterances. Suite 372 green, ruff clean.
  NOT done: hard stage timeouts (impossible in-process, see Phase 1 notes) and the
  real-provider 30-min soak, which needs Kanak's hardware and voice.
- 2026-09-22 Phase 2 core done. Input modes + bypass implemented and wired end to
  end; verified against a live meeting session on real PipeWire, including SIGTERM
  cleanup and app restoration. Suite 417 green, ruff clean. Outstanding: evdev
  hold-to-talk, and the manual Meet call which needs Kanak.
- 2026-09-22 Phase 2 COMPLETE. evdev hold-to-talk implemented and tested against
  fake events; its degradation path verified live because the real path needs a
  privileged group change. Suite 434 green, ruff clean.
- 2026-09-23 Phase 3 essentially complete. Found and fixed a real model-selection
  bug (custom Modelfile telling the LLM to reply in Hindi), removed the last two
  hardcoded model defaults, implemented mid-session fallback, and added
  VAANI_FORCE_CPU so the CPU branch is testable at all. Suite 486 green, ruff
  clean. Remaining in Phase 3: bench_cache.py (may not be needed) and
  verify_phase3.sh.
- 2026-09-23 Phase 4 done except the voice check. Takeover A (continuation)
  implemented with a deliberately refusal-heavy design; cancel measured at
  0.017 ms worst case against a 200 ms budget; AI-speaking indicator made
  undismissable after finding `vaani hide` could conceal an active takeover.
  Suite 526 green, ruff clean.
- 2026-09-23 Phase 5 complete except Kanak's own recording. Two real bugs found
  by end-to-end runs that every unit test had passed: source_language="auto"
  (rejected by real NLLB, every chunk failed) and a chunker that produced 147
  chunks from 13 s of audio because the quiet-point search could reach back to
  start+20 ms. Both fixed with regression tests. Suite 578 green, ruff clean.
- 2026-09-23 Phase 6 packaging done. Root install.sh written and verified by a
  real fresh-clone install (works, and idempotent at 11 s). README install and
  troubleshooting rewritten from actual observed failures; all 19 documented
  commands verified to exist. Suite 578 green, ruff clean.
  Outstanding in Phase 6: the GUI dashboard work (input-mode switch, model
  status, Script screen) and the rule-4 comment backlog.
- 2026-09-23 Phase 6 GUI work done. Dashboard input-mode switch, real model
  status, and a Script screen. Suite 583 green, ruff clean. Phase 6 now complete
  except the rule-4 comment backlog on pre-existing files.
- 2026-09-23 Phase 7 done. docs/16-V1-ACCEPTANCE.md written with A1-A8 evidence.
  The 30-min soak was the highest-value test of the whole effort: two real bugs
  that the unit suite could not have found, because both required a live
  microphone in a quiet room. Suite 588 green, ruff clean.
  Remaining: the rule-4 backlog on pre-existing files, and everything that needs
  Kanak's voice / a real Meet call / the desktop machine.
