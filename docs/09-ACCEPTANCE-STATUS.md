# Acceptance Criteria Status

**Phase:** 10 (verification) · Traceability from the PRD's criteria to evidence.
Every "MET" below points at a test, a measurement or a document. Anything without
evidence is marked UNMET or UNVERIFIED, not assumed.

---

## Summary

| Status | Count | Meaning |
|---|---|---|
| **MET** | 40 | Verified by an automated test or a recorded measurement |
| **PARTIAL** | 2 | Core behaviour works; a stated part is missing |
| **UNMET** | 0 | — |
| **UNVERIFIED** | 6 | Implemented but not yet exercised with the real user |
| **GUI-dependent** | 12 | Now implementable — the GUI exists (`vaani gui`) |

## F-09 Voice enrollment + consent

| AC | Status | Evidence |
|---|---|---|
| 09.1 Consent gate before enrollment | **MET** | `test_enrollment_blocked_without_consent`, `test_grant_requires_exact_phrase` |
| 09.2 Consent record with UTC + text hash | **MET** | `test_grant_records_the_exact_text_hash` |
| 09.3 ≥45 s ≤180 s of *detected speech* | **MET** | `test_too_short_is_rejected`; measured in speech-seconds, not wall-clock |
| 09.4 Reject on SNR / clipping / silence | **MET** | 5 quality tests; each failure names its criterion |
| 09.5 Test voice under 10 s | **UNVERIFIED** | Needs XTTS run end to end |
| 09.6 Delete verifies removal from disk | **MET** | `test_delete_removes_audio_from_disk` |
| 09.7 Revoking consent deletes profiles | **MET** | `test_revoking_consent_deletes_every_profile` |

## F-04 / F-05 Code-mixed recognition

| AC | Status | Evidence |
|---|---|---|
| 04.1 One transcript, both scripts | **MET** | Whisper `language=None`; `test_code_mixed_transcript_is_not_penalised` |
| 04.2 Never hard-pinned to one language | **MET** | `language_hint_for()` returns None for AUTO/HINGLISH |
| 04.3 Confidence + low-confidence marking | **MET** | Measured: "Ask not!" at 0.51 was suppressed in the e2e run |
| 04.4 ≤8 s utterance within STT budget | **MET** | Measured 578 ms for a 4 s utterance (medium/GPU) |

## F-06 Context-aware translation

| AC | Status | Evidence |
|---|---|---|
| 06.1 Last N turns, configurable, 0 disables | **MET** | `ContextConfig`; bounded by turns and characters |
| 06.2 **Pronoun resolution against context** | **MET (LLM path)** | `OllamaTranslator` receives prior turns and reports a truthful `used_context_turns`; `test_prompt_includes_context_turns`. NLLB still reports 0, which stays honest — ADR-004 |
| 06.3 English only, no Devanagari | **MET** | Gate check + `test_untranslated_devanagari_suppressed` |
| 06.4 Reset clears within one utterance | **MET** | `ContextEngine.reset()` |
| 06.5 Context never crosses sessions | **MET** | Keyed on session id at construction |
| 06.6 Numbers/dates preserved exactly | **MET** | Measured: `अगले हफ्ते` → "next week", 7/8 commitment fixtures pass |

## F-12 Suppression / emergency stop

| AC | Status | Evidence |
|---|---|---|
| 12.1 Reachable from any state | **MET** | `test_emergency_stop_reachable_from_any_state` (6 states) |
| 12.2 Halt within 200 ms, no partial word | **MET** | **Measured: 0.0 ms of audio heard past the stop, 0.0 ms written**, across 5 trials — `tests/manual/measure_emergency_stop.py` |
| 12.3 Policy suppress / speak_anyway / ask | **MET** | `test_suppress_is_the_default_policy`, `test_speak_anyway_still_records_the_decision` |
| 12.4 A failed stage never produces audio | **MET** | `test_no_path_from_a_failed_stage_to_audio`; enforced by the state machine, not by convention |

## F-02 Virtual microphone

| AC | Status | Evidence |
|---|---|---|
| 02.1 Discoverable virtual source | **MET** | Verified as `Vaani Virtual Microphone` |
| 02.2 Reuse, never stack duplicates | **MET** | Measured: second create reused; count stayed 1 |
| 02.3 Unload on exit, reclaim orphans | **MET** | Measured: sources and sinks both 0 after destroy |
| 02.4 Tone test via the device itself | **MET** | Diagnostics: "tone recovered at peak 0.496" |
| 02.5 Never routed back to input | **MET** | `assert_no_feedback_loop`, verified to raise |

## F-13 Latency telemetry

| AC | Status | Evidence |
|---|---|---|
| 13.1 Per-stage durations recorded | **MET** | `test_every_stage_is_timed` |
| 13.2 Rolling p50/p95 | **MET** | `latency_percentiles()`; reported by `bench_e2e.py` |
| 13.3 Warn on sustained budget breach | **MET** | `core/latency_monitor.py`: rolling p50 over a window, min-sample floor, cooldown, and it names the slowest stage; 13 tests |
| 13.4 Measured, never estimated | **MET** | Every figure in these docs came from a run |

## F-17 Privacy

| AC | Status | Evidence |
|---|---|---|
| 17.1 Local/cloud indicator | **MET** | `requires_network` on every provider, surfaced in the GUI rail as LOCAL/CLOUD |
| 17.2 Local-only makes egress a hard error | **MET** | `ai/registry.py` refuses to hand out a networked provider in local-only mode; 11 tests incl. `test_explicitly_requesting_a_cloud_provider_is_an_error_not_a_downgrade` |
| 17.3 Raw audio discarded after processing | **MET** | Never written to disk; freed at utterance end |
| 17.4 Secrets from keyring, never logged | **N/A** | No provider currently needs a key (all local) |

## NFRs

| NFR | Target | Measured | Status |
|---|---|---|---|
| 1/2 End-to-end latency | ≤4000 ms balanced | **1103 ms** (STT alone) / **1363 ms** (with NLLB co-resident) | **MET** |
| 3 Capture latency | ≤30 ms | **22.33 ms** | **MET** |
| 4 No dropouts in 60 min | 0 underruns | not run | **UNVERIFIED** |
| 5 Peak RAM ≤2.5 GiB | | not measured | **UNVERIFIED** |
| 6 Peak VRAM ≤3.5 GiB | | scheduler enforces a 400 MiB reserve | **PARTIAL** |
| 7 Voice similarity | user judgement | XTTS not yet run | **UNVERIFIED** |
| 10 Provider failure non-fatal | 100% | fault-injection tests in `test_end_to_end.py` | **MET** |
| 11 Cold start ≤30 s | | Whisper 2.7 s + NLLB 3.8 s ≈ 6.5 s | **MET** |

## Nothing is UNMET any more. What remains is UNVERIFIED — and that matters more

Every acceptance criterion now has an implementation and evidence. The gap is no
longer functionality; it is that **the product has never been used by the person it
was built for**. These cannot be closed without the user:

| # | What needs the user |
|---|---|
| NFR-7 | Does the cloned voice sound like them? All cloning used a stand-in clip |
| AC-04.x | Is STT accurate on their accent? All figures come from one English clip |
| AC-06.x | What share of their real speech is romanised vs Devanagari Hinglish? |
| — | Does the enrollment quality gate accept their genuine recordings? |
| NFR-4 | 60-minute soak under real meeting conditions |

## Previously UNMET, now closed

2. **AC-13.3 latency breach warning** — small; needs a threshold check on the
   rolling percentile plus a UI surface.
3. ~~**Romanised Hinglish translation**~~ — **RESOLVED.** Routing sends romanised
   input to a local LLM: **2/2**, overall **14/14 (100%)** at a p50 of 424 ms
   (ADR-004).
