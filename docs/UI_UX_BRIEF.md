# Vaani v1.0 — UI/UX Brief (source of truth)

**Status:** Phase 0 draft, 2026-09-22. This is the v1 delta on top of `04-UIUX-BRIEF.md`,
whose design system (dark default palette, tokens in `ui/theme.py`, components in
`ui/components.py`) is kept unchanged.

## 1. Direction

In a meeting, Vaani is **felt, not watched**: a small always-on-top overlay in a
screen corner, driven by hotkeys. The full app (`vaani gui`) is for setup:
enrollment, models, diagnostics, script mode. No new toolkit: tkinter.

## 2. Screens

| Screen | Purpose |
|---|---|
| Overlay (meeting) | state, latency, model status, takeover indicator |
| App → Dashboard | start/stop meeting, input-mode switch, model status + reasons |
| App → Voice | consent, enrollment, test voice (existing) |
| App → Script | pick file / record → progress → open the resulting script |
| App → Diagnostics | `doctor` checks + auto-detection report (existing + new) |

## 3. Overlay layout (about 320×110 px, corner configurable)

```
┌──────────────────────────────────────────┐
│ ● LISTENING   always-on         1.9s p50 │  row 1: state pill · input mode · latency
│ "we can finish it by tomorrow"           │  row 2: last English line (1 line, ellipsised)
│ STT medium·GPU  LLM phi4-mini  TTS me·GPU│  row 3: model status (tooltip = reason)
└──────────────────────────────────────────┘
```

State pill values and colours (tokens from `theme.py`):

| State | Label | Colour |
|---|---|---|
| idle / ptt closed | `PTT — press to talk` | muted |
| capturing | `LISTENING` | accent |
| processing | `TRANSLATING` | accent, pulsing |
| speaking (his translation) | `SPEAKING` | success |
| bypass | `BYPASS — your real voice` | warning (amber), whole border amber |
| **AI speaking (takeover)** | `AI SPEAKING — Ctrl+Alt+C to stop` | **danger red, full-width banner, border red** |
| fallback happened | `fallback: x → y` toast for 5 s | warning |
| error | message + fix hint | danger |

The AI-SPEAKING state must be readable at a glance from across the screen, so it
takes over row 1 and row 2 completely. It can't be hidden while active: `hide` is
ignored during AI speech, and the overlay re-shows itself.

## 4. Interactions

- Every action is available as a hotkey (desktop shortcut → `vaani <cmd>`), as a
  CLI command, and as an overlay right-click menu item.
- The overlay has a **hold-to-talk button** (mouse down = talk) in PTT modes.
- Clicking the latency opens a small popover: per-stage p50/p95.
- Model status tooltip: the one-line reason from auto-detection.

## 5. Responsive / accessibility

Fixed-size overlay. The app window has a 720×480 minimum. Every state is conveyed by
**text plus colour**, never colour alone. Font size follows `theme.py`, with a minimum of
11 pt. All actions are keyboard reachable.

## 6. States per screen

- Loading: "Detecting hardware…" / "Benchmarking models (≤20 s)…" with a
  checklist that fills in (reuses `diagnostics/startup.py` callbacks).
- Empty: no enrolled voice → Dashboard shows "Enroll your voice to speak as
  yourself" and takeover controls are disabled with that reason.
- Error: a banner with the fix (for example "Ollama not running: `ollama serve`").
- Success: script mode finishes with a "Open script" and "Open folder" button.
- Disabled: controls greyed out, with the tooltip giving the reason.

## 7. Script mode flow

Pick file / Record → (options: English audio in my voice ☐, output folder) → a
progress bar per chunk (resumable) → the result pane shows the English script, editable,
with Save.
