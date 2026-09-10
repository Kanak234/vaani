# Product Requirements Document — Vaani

**Phase:** 2 · **Depends on:** BRD v1.0

---

## 1. Product overview

A single-user Linux desktop application. The user speaks Hindi/English/Hinglish into
a real microphone; the application emits fluent English speech in the user's own
enrolled voice into a virtual microphone that meeting software can select as input.

Five surfaces: **Dashboard**, **Meeting Mode**, **Voice Profile**, **Settings**,
**Diagnostics**.

## 2. Feature list

| ID | Feature | Priority | Surface |
|---|---|---|---|
| F-01 | Audio device selection & management | P0 | Dashboard/Settings |
| F-02 | Virtual microphone lifecycle | P0 | Meeting/Diagnostics |
| F-03 | Voice activity detection & utterance segmentation | P0 | Engine |
| F-04 | Streaming speech recognition (Hi/En/Hinglish) | P0 | Engine |
| F-05 | Code-mix language identification | P0 | Engine |
| F-06 | Context-aware translation | P0 | Engine |
| F-07 | English text normalization | P0 | Engine |
| F-08 | Personal-voice TTS | P0 | Engine |
| F-09 | Voice enrollment + consent | P0 | Voice Profile |
| F-10 | Meeting Mode transport controls | P0 | Meeting |
| F-11 | Live dual transcript | P0 | Meeting |
| F-12 | Output suppression / emergency stop | P0 | Meeting |
| F-13 | Latency telemetry | P0 | All |
| F-14 | Provider abstraction & switching | P0 | Settings |
| F-15 | Diagnostics suite | P0 | Diagnostics |
| F-16 | Conversation mode (monitor to headphones) | P0 | Dashboard |
| F-17 | Privacy controls & cloud indicator | P0 | Global |
| F-18 | Transcript persistence (opt-in) | P1 | Settings |
| F-19 | Local-only mode | P1 | Settings |
| F-20 | Session history | P1 | Dashboard |

## 3. User stories & acceptance criteria

Every criterion below is written to be **mechanically checkable**.

### F-09 Voice enrollment + consent
> As the user, I want to enroll my own voice so translated speech sounds like me.

**AC-09.1** Enrollment is unreachable until a consent dialog is affirmatively
accepted. The dialog states: whose voice, that a clone will be produced, where the
data is stored, and how to delete it. A checkbox that must be ticked plus a typed
confirmation of the word `I CONSENT`. Closing the dialog any other way aborts.
**AC-09.2** A consent record is persisted with UTC timestamp, consent text hash,
and app version before any audio is captured for enrollment.
**AC-09.3** Enrollment requires ≥ 45 s and ≤ 180 s of detected *speech* (silence
excluded). Progress shows speech-seconds captured, not wall-clock.
**AC-09.4** Enrollment audio is rejected with a specific reason if: mean SNR < 15 dB,
clipping on > 1 % of samples, or > 40 % silence.
**AC-09.5** "Test Voice" synthesises a fixed English sentence in under 10 s and plays
it to the monitor output.
**AC-09.6** "Delete Voice Profile" requires typed confirmation, removes the profile
row, the embedding file, and all reference audio, then verifies removal from disk.
Post-deletion, TTS reports `VOICE_PROFILE_MISSING` rather than falling back silently.
**AC-09.7** Revoking consent triggers AC-09.6 automatically.

### F-04 / F-05 Recognition of code-mixed speech
> As the user, I want "Actually मेरा point ये है कि deadline थोड़ी tight है" understood
> as one sentence, not an error.

**AC-04.1** A mixed-script utterance produces one transcript containing both scripts;
it is never split into two utterances on script change alone.
**AC-04.2** The recogniser is never hard-pinned to a single language when the mode is
Hinglish; it reports a per-utterance language distribution (e.g. `hi:0.6, en:0.4`).
**AC-04.3** Transcripts carry a confidence score. Below the configured threshold
(default 0.55) the utterance is marked low-confidence and F-12 gating applies.
**AC-04.4** An utterance of ≤ 8 s produces a transcript within the mode's STT budget.

### F-06 Context-aware translation
> As the user, I want "उसमें authentication वाला part बाकी है" to translate using what
> was already discussed.

**AC-06.1** Translation input includes the last N turns (default 6, configurable
0–20). N = 0 disables context.
**AC-06.2** Pronouns and elided subjects resolve against context where the
antecedent is present; regression fixtures cover this.
**AC-06.3** Output is English only — no Devanagari, no untranslated Hindi words
except proper nouns.
**AC-06.4** "Reset Context" clears the window within one utterance boundary.
**AC-06.5** Context never crosses sessions unless explicitly restored.
**AC-06.6** Numbers, dates and durations are preserved exactly. A regression suite
asserts `अगले हफ्ते` → "next week", never "next month".

### F-12 Suppression / emergency stop
> As the user, I would rather emit nothing than emit a confident mistranslation.

**AC-12.1** Emergency stop is reachable in one action from Meeting Mode and by a
global hotkey.
**AC-12.2** Emergency stop halts synthesis, flushes the output buffer, and writes
silence to the virtual mic within 200 ms. It never leaves a partial word playing.
**AC-12.3** When STT confidence < threshold **or** translation reports low
confidence, behaviour follows the configured policy: `suppress` (default),
`speak_anyway`, or `ask`. In `suppress`, the UI shows what was withheld and why.
**AC-12.4** A failed stage never produces audio. Silence plus a visible error is the
only acceptable failure output.

### F-02 Virtual microphone
**AC-02.1** On start, the app creates a PipeWire virtual source named
`Vaani Virtual Microphone`, discoverable by any Pulse/PipeWire client.
**AC-02.2** If a Vaani virtual source already exists (crash leftover), it is reused
or cleanly replaced — duplicates are never stacked.
**AC-02.3** On graceful exit the module is unloaded. On crash, the next start
detects and reclaims the orphan.
**AC-02.4** Diagnostics can write a 1 kHz tone to the virtual mic and confirm it via
the node's own monitor.
**AC-02.5** The virtual source is never routed back into the capture input (feedback
loop prevention), and this is asserted at startup.

### F-10 / F-11 Meeting Mode
**AC-10.1** Start is disabled with a stated reason unless: input device selected,
virtual mic created, voice profile ready, all providers healthy.
**AC-10.2** Pause stops synthesis but keeps devices and models warm; resume is < 500 ms.
**AC-10.3** Mute stops capture at the source — no audio enters the pipeline at all.
**AC-10.4** The live transcript shows original and translation side by side, with
per-utterance state (`transcribing`/`translating`/`speaking`/`suppressed`/`error`).
**AC-10.5** Device disconnect mid-session pauses and surfaces a recovery action; it
does not crash.

### F-13 Latency telemetry
**AC-13.1** Every utterance records per-stage durations: capture, VAD, STT,
translate, TTS, buffer, output.
**AC-13.2** The UI shows a rolling p50/p95 end-to-end figure.
**AC-13.3** Sustained breach of the mode's budget raises a visible warning with the
offending stage named.
**AC-13.4** Latency figures are measured, never estimated or hard-coded.

### F-17 Privacy
**AC-17.1** A persistent indicator shows LOCAL or CLOUD, and in cloud mode names the
provider and what is sent (text/audio).
**AC-17.2** Local-only mode makes any outbound network call from the AI layer a hard
error rather than a silent fallback.
**AC-17.3** Raw audio is discarded after processing unless persistence is explicitly on.
**AC-17.4** Secrets are read from the OS keyring or environment, never from the repo
and never logged.

## 4. User flows

**Enrollment:** Voice Profile → Enroll → consent gate (AC-09.1) → consent recorded →
prompts shown → record until ≥ 45 s speech → quality check → embedding built →
test voice → active. *Failure at quality check returns to recording with the reason.*

**Translation (per utterance):** capture frames → VAD start → buffer → VAD end →
finalize STT → language ID → context assembly → translate → confidence gate →
normalize → TTS → post-process → buffer → virtual mic. *A gate failure exits to
"suppressed" and emits nothing.*

**Meeting:** launch → select devices → create virtual mic → preflight → Start → user
selects the virtual mic inside Zoom/Meet → speak → Stop → module unloaded.

## 5. Error states

| Condition | Behaviour | Recovery |
|---|---|---|
| Mic unavailable | Blocked start, device named | Re-scan devices |
| Mic permission denied | Explain PipeWire/portal permission | Retry |
| Virtual mic creation fails | Meeting Mode disabled; conversation mode still allowed | Retry / diagnostics |
| Orphaned virtual mic | Auto-reclaim, informational | Automatic |
| No network, cloud provider | Fail over to local, banner shown | Automatic |
| No network, local-only | Unaffected | n/a |
| STT failure | Utterance dropped, marked error, no audio | Next utterance |
| Translation failure | Suppressed, marked error, no audio | Next utterance |
| TTS failure | Suppressed; optional fallback voice **with explicit UI marking** | Next utterance |
| Voice profile missing | TTS blocked; prompt to enroll | Enroll |
| Provider timeout | Cancel, mark, continue | Automatic |
| Rate limit | Backoff + switch provider | Automatic |
| Buffer underrun | Insert silence, count it, warn if repeated | Automatic |
| Device disconnect | Pause, surface picker | Reselect |
| OOM risk | Refuse to load larger model, recommend smaller | User choice |
| Crash | Next start reclaims virtual mic and marks session crashed | Automatic |

## 6. Edge cases

Utterance > 30 s → force-segment at a prosodic boundary · pure English input →
detected, passed through lightly normalized, still spoken in user's voice ·
no speech for 5 min → idle, models may unload · rapid short utterances ("हाँ",
"ok") → coalesced within 400 ms · overlapping user speech during playback → capture
gated to prevent echo · unsupported language detected → suppressed with reason ·
profanity/sensitive content → translated faithfully, never editorialised ·
numbers/dates → preserved exactly (AC-06.6) · silence-only utterance → discarded
before STT.

## 7. Permissions

Microphone access via PipeWire/portal · PipeWire module load (no root — verified) ·
keyring for secrets · filesystem confined to the app data directory · network only
when a cloud provider is enabled.

## 8. Settings

**Audio:** input device, monitor output, virtual mic name, sample rate, frame size,
input gain.
**Language:** source mode (auto/hi/en/hinglish), target (English v1), formality.
**Providers:** STT / translation / TTS provider + model, per-stage timeouts.
**Performance:** mode (low-latency/balanced/quality), context window size, VAD
sensitivity, max utterance length.
**Privacy:** local-only toggle, transcript persistence, retention days, log level,
telemetry (off, no remote telemetry in v1).

## 9. Diagnostics

Microphone test (level meter + 3 s echo) · speaker test · virtual mic test
(AC-02.4) · network reachability · STT round-trip on a fixture · translation
round-trip on a fixture · TTS round-trip · **end-to-end latency test producing the
per-stage table** · model/memory report · full system report export (secrets redacted).
