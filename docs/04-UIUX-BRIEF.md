# UI/UX Design Brief — Vaani

**Phase:** 4 · Defined *before* implementation. Components are not to be invented ad hoc.

---

## 1. Design philosophy

**"Instrument, not dashboard."**

This application runs while its user is in a live meeting, being watched by other
people. Every design decision follows from that single fact:

1. **Glanceable, not readable.** The user has roughly half a second of attention.
   State must be legible from posture, colour and position — never from reading a
   paragraph.
2. **One unmistakable truth per screen.** The dominant question in Meeting Mode is
   *"is my translated voice going out right now?"* That answer gets the most visual
   weight on the screen. Everything else is subordinate.
3. **Calm by default, loud only on failure.** No animation, no gradients, no
   decorative motion. Colour is a signal, not a style — if everything is coloured,
   nothing is.
4. **Destructive actions are deliberately slower.** Emergency stop is instant;
   deleting a voice profile requires typing.
5. **Honesty over polish.** If the system does not know, it says so. No fake progress
   bars, no spinner standing in for an unknown state, no invented latency numbers.

**Explicit anti-pattern:** the generic dark "AI dashboard" — purple gradients, glowing
cards, sparkline decoration, an emoji in every heading. Rejected. This is a tool that
sits beside a video call, not a product screenshot.

## 2. Layout system

Fixed left rail (72 px, icon + label) → content region → optional right inspector
(320 px). Minimum window 1024×680; usable at 1280×800.

8 px base grid. Spacing scale: `4, 8, 12, 16, 24, 32, 48, 64`. Content max width
960 px so text never runs edge to edge.

Meeting Mode overrides the pattern: a compact mode (480×220, always-on-top) shows
only status, latency, transport and emergency stop — sized to sit next to a meeting
window without covering faces.

## 3. Typography

One family: **Inter**, fallback `system-ui, sans-serif`. Devanagari falls back to
**Noto Sans Devanagari** — required, since transcripts render mixed script and a
missing glyph box on screen is a failure.

| Token | Size/Line | Weight | Use |
|---|---|---|---|
| `display` | 32/40 | 600 | Session state word |
| `title` | 20/28 | 600 | Screen titles |
| `heading` | 16/24 | 600 | Section headers |
| `body` | 14/20 | 400 | Default |
| `transcript` | 16/26 | 400 | Transcript text — larger, for glancing |
| `label` | 12/16 | 500 | Field labels, uppercase, 0.04em tracking |
| `mono` | 13/20 | 400 | Latency, device ids, logs (`JetBrains Mono`) |

Numbers in the latency readout use tabular figures so digits do not jitter as values
change — a jittering number attracts the eye during a meeting.

## 4. Colour system

Semantic tokens only. No raw hex in components.

**Dark theme (default — sits beside a video call without glare):**

| Token | Value | Use |
|---|---|---|
| `bg-base` | `#0F1115` | App background |
| `bg-surface` | `#171A21` | Cards, rail |
| `bg-raised` | `#1F232C` | Inputs, hover |
| `border` | `#2A2F3A` | Dividers |
| `text-primary` | `#E8EAED` | Body |
| `text-secondary` | `#9AA1AE` | Labels |
| `text-muted` | `#6B7280` | Disabled |
| `accent` | `#4F8DF7` | Primary action, focus |
| `state-live` | `#22C55E` | Actively translating |
| `state-warn` | `#F5A623` | Degraded, latency breach |
| `state-error` | `#EF4444` | Failure, emergency |
| `state-idle` | `#6B7280` | Stopped |
| `cloud` | `#A855F7` | Data leaving the machine |

Light theme mirrors the same token names.

**Colour rules.** `state-live` green appears in exactly one place at a time — the
session state. `cloud` purple is reserved solely for the privacy indicator; it is
never used decoratively, so purple on screen always means one thing. All pairs meet
WCAG AA (4.5:1 body, 3:1 large). No state is signalled by colour alone — each carries
an icon and a text label, for colour-blind users and for glanceability.

## 5. Component system

Closed set. Anything not here is not built without extending this document.

- **Button** — `primary` / `secondary` / `ghost` / `danger`; sizes `sm|md|lg`;
  states default/hover/active/disabled/loading. Disabled buttons always carry a
  tooltip naming the blocking condition (AC-10.1).
- **StatusPill** — dot + label. `idle|initializing|live|paused|degraded|error`.
- **DeviceSelect** — labelled select with a live level meter inline, a refresh
  action, and an explicit empty state.
- **Card** — `surface` background, 1 px border, 4 px radius, 16 px padding.
  Optional header with title + action. No shadow (shadows read as "web page").
- **MetricReadout** — mono value, unit, label, optional trend arrow. Shows `—` when
  unknown. **Never shows a placeholder number.**
- **TranscriptRow** — two columns (original / English), per-row state chip,
  timestamp, suppress action while pending.
- **LevelMeter** — horizontal, −60→0 dBFS, peak hold, red above −3 dBFS.
- **Toggle**, **TextField**, **Slider** (value always shown numerically),
  **Modal** (focus-trapped, Esc closes non-destructive only),
  **Banner** (`info|warn|error`, dismissible unless blocking),
  **ConsentDialog** (single-purpose, cannot be dismissed by clicking outside),
  **LatencyBar** (stacked per-stage segments, the one visualisation in the product),
  **EmptyState**, **DiagnosticRow** (name, status, duration, detail, run action).

## 6. Interaction states

- **Loading:** skeletons only where the shape is known; otherwise an indeterminate
  bar with the operation named ("Loading STT model…"). Any operation > 3 s shows
  elapsed time. Model downloads show bytes and percent — real values only.
- **Empty:** icon + one sentence + one primary action. Used for no devices, no
  voice profile, no transcript, no sessions.
- **Error:** what failed, why, what to do next, and a retry. Never a bare code.
  Technical detail sits behind a "Details" disclosure for bug reports.
- **Focus:** 2 px `accent` ring, 2 px offset, never removed.

## 7. Screens

**Dashboard.** Readiness at a glance: four status cards (Microphone / Voice Profile /
Virtual Mic / Providers), a language row (source → target), a latency readout from
the last session, and one primary `Start Translation`. If anything blocks start, the
blocking card is outlined in `state-warn` and the button explains why.

**Meeting Mode.** Top: session state (`display` type) + latency + privacy indicator.
Middle: the dual live transcript, newest at the bottom, auto-scrolling with a pause
on manual scroll. Bottom: transport bar — Pause, Mute, and **Emergency Stop, visually
separated, `danger`, always enabled, never behind a confirmation.** Right inspector
(collapsible): device selectors, mode, per-stage LatencyBar.

**Voice Profile.** Before enrollment: a plain-language explanation of what a voice
clone is, then Begin Enrollment. During: prompts to read, speech-seconds progress,
live meter, quality warnings in real time. After: status, created date, test voice,
re-enroll, delete (destructive, typed confirmation), and the consent record with its
timestamp.

**Settings.** Left sub-nav (Audio, Language, Providers, Performance, Privacy, Logs).
Changes apply immediately; those requiring a restart are marked. Privacy is a
first-class section, not a footnote.

**Diagnostics.** A list of DiagnosticRows and `Run All`. Results are pass/fail with
measured durations. The end-to-end test renders the per-stage LatencyBar and can
export a redacted report.

## 8. Accessibility

Full keyboard operation; visible focus everywhere; a global emergency-stop hotkey
that works unfocused. All controls labelled for screen readers; transcript is an ARIA
live region (`polite`) so updates are announced without interrupting. State never
conveyed by colour alone. Respects `prefers-reduced-motion` and OS light/dark.
Minimum 14 px body text; UI scales to 200 % without loss of function. Targets ≥ 32 px.

## 9. Responsive behaviour

Desktop-first, three widths: compact overlay (480×220), standard (1024–1440, right
inspector collapses below 1180), wide (> 1440, inspector pinned). Not a mobile
product; the layout must simply not break when the window is resized.
