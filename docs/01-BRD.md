# Business Requirements Document — Vaani

**Product:** Vaani — Real-Time Personal Voice Translation & Meeting Assistant
**Version:** 1.0 (v1 scope)
**Phase:** 1

---

## 1. Product vision

A person who thinks in Hindi should not have to sound less capable in English.

Vaani captures a user speaking naturally — Hindi, English, or the Hinglish blend
most Indian professionals actually speak — and emits fluent English speech **in that
user's own voice**, routed into a virtual microphone so any meeting application
receives it as ordinary mic input.

The product is explicitly *not* a subtitle generator and *not* a note-taker. Those
exist. The differentiator is that the output is **speech, in the speaker's own
voice, fast enough to hold a conversation.**

## 2. Problem statement

Fluency in English is used as a proxy for competence in global professional
settings. Millions of skilled engineers, founders, analysts and consultants in India
carry an ability penalty in English-language meetings that has nothing to do with
their actual ability. The penalty shows up as:

- Slower speech while mentally translating, read by others as hesitancy
- Reduced participation in fast-moving discussion
- Loss of nuance — the speaker says the simple thing they can say, not the precise
  thing they mean
- Accent-related comprehension friction on the listener's side

Existing tools miss in specific ways:

| Existing approach | Why it fails this problem |
|---|---|
| Live captions (Meet/Teams) | Text, not speech. Reader must look away. Speaker still sounds hesitant |
| Generic TTS translators | Robotic third-party voice. Destroys speaker identity and rapport |
| Interpreter services | Cost, scheduling, a third person in a private conversation |
| Post-hoc transcription tools | Not real time. Solves recall, not participation |
| Phone translation apps | Turn-based, hold-to-talk. Unusable in a live meeting |

The gap: **real-time, voice-preserving, code-mixed-aware, meeting-routable.**

## 3. Target users

**Primary (v1):** A single technical professional in India who works in
English-language meetings and speaks Hinglish natively. Self-hosted, single-user,
on their own laptop. This is deliberately narrow — it is the author's own use case,
which means requirements can be validated against a real user rather than a
hypothesis.

**Secondary (v2+):** Distributed teams, consultants, sales engineers, remote
contractors, students in foreign-language programs.

**Explicitly out of scope:** simultaneous multi-speaker interpretation, conference
interpretation, broadcast use, or anything where a person other than the enrolled
user's voice is synthesised.

## 4. Personas

### P1 — Kanak, Senior Engineer (primary persona)
- Thinks and reasons in Hinglish; writes English fluently; speaks it slower than he thinks
- 4–6 video calls/week with non-Hindi speakers
- Owns a Linux laptop with a mid-range GPU
- Technically capable of running local models; privacy-conscious about voice data
- **Success for Kanak:** he stops rehearsing sentences before speaking in standups
- **Failure for Kanak:** the tool mistranslates a commitment ("next week" vs "next
  month") and he has to publicly correct it. He would rather it emit nothing.

### P2 — Priya, Consultant
- Client-facing; tone and rapport matter more than raw accuracy
- Would abandon the product instantly if it sounded like a robot on a client call
- **Success:** clients don't notice a tool is involved
- **Failure:** obvious synthetic voice, or lag that makes her step on others' speech

### P3 — Arjun, Graduate Student
- Non-native English, academic seminars, low budget, older hardware
- **Success:** works acceptably without a paid API subscription
- **Failure:** requires a GPU he does not have

## 5. Business use cases

| ID | Use case | Priority |
|---|---|---|
| BUC-1 | Speak Hinglish in a video meeting; participants hear English in the user's voice | **P0** |
| BUC-2 | Practice/preview mode — hear own translated output through headphones | P0 |
| BUC-3 | Enroll and manage a consented personal voice profile | **P0** |
| BUC-4 | Diagnose why audio is not reaching the meeting app | P0 |
| BUC-5 | Review a session transcript after the fact | P1 |
| BUC-6 | Run fully offline with no data leaving the machine | P1 |
| BUC-7 | Swap AI providers without reinstalling | P1 |
| BUC-8 | Additional target languages beyond English | P2 |

## 6. Core workflows

**W1 — First run.** Install → grant mic access → select input device → create virtual
mic → record consent → enroll voice (~60 s of speech) → test → ready.

**W2 — Meeting.** Open Vaani → select input mic + virtual output → Start → select
"Vaani Virtual Microphone" inside Zoom/Meet → speak → English in own voice reaches
participants → Stop.

**W3 — Correction.** User sees a wrong transcript on screen → presses Suppress before
synthesis completes → nothing is emitted. (Directly addresses P1's stated failure mode.)

**W4 — Degradation.** Cloud provider times out → the app switches to the local
provider and shows the change → session continues at lower quality rather than dying.

## 7. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Capture audio from a user-selected input device | P0 |
| FR-2 | Detect speech/silence boundaries to segment utterances | P0 |
| FR-3 | Transcribe Hindi, English and code-mixed Hinglish | P0 |
| FR-4 | Identify language mix per utterance without assuming a single language | P0 |
| FR-5 | Translate to natural English using conversational context | P0 |
| FR-6 | Synthesise English speech in the user's enrolled voice | P0 |
| FR-7 | Create and manage a virtual microphone device | P0 |
| FR-8 | Route synthesised audio to the virtual microphone | P0 |
| FR-9 | Record explicit, revocable voice-cloning consent before enrollment | P0 |
| FR-10 | Provide Start/Stop/Pause/Mute and an emergency stop | P0 |
| FR-11 | Show live original transcript and English translation | P0 |
| FR-12 | Maintain a bounded conversational context window; allow reset | P0 |
| FR-13 | Abstract STT/translation/TTS behind swappable provider interfaces | P0 |
| FR-14 | Report per-stage and end-to-end latency | P0 |
| FR-15 | Suppress output rather than emit a known-bad translation | P0 |
| FR-16 | Diagnostics for every device and provider in the chain | P0 |
| FR-17 | Delete a voice profile and all derived data on request | P0 |
| FR-18 | Indicate unambiguously when data is leaving the machine | P0 |
| FR-19 | Optional, off-by-default transcript persistence | P1 |
| FR-20 | Local-only operation mode | P1 |

## 8. Non-functional requirements

| ID | Requirement | Target | Verification |
|---|---|---|---|
| NFR-1 | End-to-end latency, low-latency mode | ≤ 2.5 s p50 speech-end → audio-start | Measured, Phase 10 |
| NFR-2 | End-to-end latency, balanced mode | ≤ 4.0 s p50 | Measured |
| NFR-3 | Capture latency | ≤ 30 ms | **Measured: 22.33 ms** ✅ |
| NFR-4 | No audio dropouts in a 60-min session | 0 underruns | Soak test |
| NFR-5 | Peak RAM | ≤ 2.5 GiB | Constrained by 2.9 GiB free |
| NFR-6 | Peak VRAM | ≤ 3.5 GiB | 4 GiB card, leave headroom |
| NFR-7 | Voice similarity to enrolled speaker | Subjective pass by the user | User eval |
| NFR-8 | API keys never written to disk in plaintext | 0 violations | Code review + test |
| NFR-9 | Raw audio never persisted unless explicitly enabled | 0 violations | Test |
| NFR-10 | Any single provider failure is non-fatal to the session | 100% | Fault injection |
| NFR-11 | Cold start to ready | ≤ 30 s | Measured |

## 9. Success criteria

**v1 ships when:**
1. The primary user completes a real 30-minute meeting using it, and participants
   understood him throughout.
2. p50 end-to-end latency in low-latency mode is at or under 2.5 s, measured.
3. Zero incidents of confidently emitting a wrong translation of a commitment
   (date, number, or yes/no). Suppression is the correct behaviour.
4. Voice output is judged by the user to sound like himself.
5. A full offline session completes with the network physically disabled — and this
   has actually been tested, not assumed.

**Anti-goals — the product fails if it:** sounds robotic; adds so much lag the user
talks over people; silently uploads audio; or requires sudo/kernel modules to install.

## 10. Constraints

- **C1** 4 GiB VRAM — one mid-size GPU model resident at a time
- **C2** ~2.9 GiB free RAM at baseline — the true binding constraint
- **C3** Linux/PipeWire only in v1; OS audio layer must sit behind an interface
- **C4** No sudo may be required for normal operation (verified achievable)
- **C5** Python 3.14 — dependency availability must be verified, never assumed
- **C6** Single user, single voice profile in v1
- **C7** Personal-scale budget: local-first, cloud strictly optional

## 11. Risks

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R-1 | Hinglish STT accuracy is poor on code-mixed speech | **High** | Medium | Benchmark on real user audio early; allow larger model in Quality mode |
| R-2 | Voice-cloning TTS too slow or too large for 4 GiB | High | **High** | Phase 7 evaluates; fallback to a fast non-cloned voice with a clear UI indication |
| R-3 | Cumulative latency makes conversation unnatural | **High** | Medium | Per-stage budget; streaming wherever supported; measure continuously |
| R-4 | Meeting app rejects the virtual source | High | Low | **Already mitigated — virtual source verified on this machine** |
| R-5 | Mistranslation causes real professional harm | **High** | Medium | Confidence gating + suppression + visible transcript (FR-15) |
| R-6 | Voice profile misused to impersonate someone | **High** | Low | Consent record required; single enrolled user; local-only profile storage |
| R-7 | Python 3.14 dependency gaps | Medium | Medium | Verify each dependency before adopting; two already eliminated this way |
| R-8 | Memory exhaustion mid-meeting | High | Medium | Model scheduling; hard memory ceiling; graceful degradation |
| R-9 | Echo — translated output re-enters the mic | Medium | Medium | Gate capture during playback; never route the virtual sink back to input |

## 12. Assumptions

- A1 The user is authorised to clone the voice they enroll (they are the speaker)
- A2 A headset is used; open-speaker echo is a v2 concern
- A3 The user can select the input device inside their meeting application
- A4 Meetings are the dominant use case; broadcast is out of scope
- A5 Occasional imperfect translation is acceptable **provided errors are visible**
- A6 The user tolerates a short delay for correctness over instant wrong output

## 13. Future opportunities

Bidirectional translation (understanding other participants) · additional target
languages · Windows/macOS support · mobile companion · team deployment ·
glossary/jargon pinning for domain vocabulary · speaker-adaptive fine-tuning ·
meeting summary from the transcript already produced.
