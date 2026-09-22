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
- [ ] Install ruff in the venv (missing), then fix F821 in ui/app.py; ruff clean on touched files
- [ ] `core/supervisor.py`: stage timeout/exception/OOM wrapper + tests
- [ ] Bounded queues + drop-oldest counters in session/engine.py + tests
- [ ] Worker thread supervision (restart stage; 3 crashes/60 s → fallback)
- [ ] Rotating log file
- [ ] Soak test: RSS sampling + pass rule (≤ +10 % from 5→30 min); fake-provider run here
- [~] verify_phase1.sh: real 30-min soak on the laptop (Kanak runs)

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

## Comment backlog (rule 4)
Existing files not yet brought to the rule-4 comment standard: all of `src/vaani/**`
(to be ticked off as each is touched).

## Session log
- 2026-09-22 Phase 0 done in a cloud session. Commits are local; delivered as a git bundle.
- 2026-09-22 Local session on the Victus. Recovered the cloud session's bundle
  (it had never reached the repo) into branch `v1-complete`, then committed
  ~1430 lines of Phase 3/4 work that was sitting uncommitted and unlost.
  Suite green at 318. Phase 0 still blocked on Kanak confirming defaults.
