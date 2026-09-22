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
- [ ] `session/input_mode.py` (always_on / ptt_toggle / ptt_hold / bypass) + tests
- [ ] Bypass path in engine (flush TTS queue, raw mic → vmic) + tests
- [ ] `system/evdev_ptt.py` optional hold-to-talk + tests (fake device)
- [ ] CLI + IPC: ptt, bypass (+ COMMANDS set)
- [ ] Overlay: states from UI brief, live latency, hold-to-talk button
- [ ] Virtual mic cleanup on SIGTERM/SIGINT/panic/crash-orphan + tests
- [~] verify_phase2.sh + manual Meet check (Kanak)

## Phase 3 — Auto model detection
- [ ] CUDA "really works" probe + VAANI_FORCE_CPU
- [~] Ranking + reasons landed as `ai/model_detect.py` (rank_models / probe_model /
      resolve_llm_model), not `system/model_select.py`. `bench_cache.py` still missing.
- [~] Remove hardcoded model defaults → "auto" (cli.py done via ai/model_detect.py;
      `ai/assist.py:123` STILL hardcodes qwen3-coder:latest - not installed here)
- [~] `ai/registry.get_with_fallback()` landed; per-stage supervisor wiring + overlay toast pending
- [ ] `vaani models` command (shows choices + reasons)
- [~] verify_phase3.sh (GPU run + FORCE_CPU run) on the laptop

## Phase 4 — Takeover
- [ ] `session/continuation.py` (Takeover A) + tests
- [~] Takeover B: `session/remote_listener.py` + cli `takeover` + ipc verbs +
      `overlay.set_takeover()` landed; end-to-end Linux meeting wiring unproven
- [ ] AI-tagged output + `cancel` ≤200 ms + tests (measure like measure_emergency_stop)
- [ ] Overlay AI-SPEAKING banner (can't be hidden)
- [ ] CLI/IPC: takeover, cancel
- [~] Manual: speaks in his voice (needs enrolled voice)

## Phase 5 — Offline script mode
- [ ] `session/script_mode.py` + chunk checkpoints + tests
- [ ] `vaani script FILE | --record [--audio] [--out]`
- [ ] Fixture check: Hinglish cases → correct English
- [~] Kanak's own recording

## Phase 6 — UI + packaging
- [ ] Root `install.sh` (TRD §6), idempotent
- [ ] Dashboard input-mode switch, model status, Script screen
- [ ] Config defaults + README (setup + troubleshooting)
- [ ] Comment pass: every file touched in v1 has rule-4 comments; remaining files
      listed in the "comment backlog" below and done file by file
- [~] verify_phase6.sh fresh-clone install on the laptop

## Phase 7 — Final verification
- [ ] Full suite + lint
- [ ] Checklist A1–A8 with PASS / UNVERIFIED and evidence
- [ ] List of what couldn't be verified (desktop machine, etc.)

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
