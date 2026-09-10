"""Design tokens from docs/04-UIUX-BRIEF.md.

Semantic tokens only. No component may use a raw hex value; if a colour is needed
that is not here, the brief gets extended first. That rule is what keeps a UI from
drifting into the "generic AI dashboard" the brief explicitly rejects.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Palette:
    bg_base: str
    bg_surface: str
    bg_raised: str
    border: str
    text_primary: str
    text_secondary: str
    text_muted: str
    accent: str
    state_live: str
    state_warn: str
    state_error: str
    state_idle: str
    cloud: str


#: Dark is the default: this window sits beside a video call, where a bright
#: panel is glare in the user's eyes for an hour.
DARK = Palette(
    bg_base="#0F1115", bg_surface="#171A21", bg_raised="#1F232C",
    border="#2A2F3A", text_primary="#E8EAED", text_secondary="#9AA1AE",
    text_muted="#6B7280", accent="#4F8DF7", state_live="#22C55E",
    state_warn="#F5A623", state_error="#EF4444", state_idle="#6B7280",
    cloud="#A855F7",
)

LIGHT = Palette(
    bg_base="#FFFFFF", bg_surface="#F6F7F9", bg_raised="#EDEFF3",
    border="#D6DAE1", text_primary="#12141A", text_secondary="#4B5361",
    text_muted="#8A93A2", accent="#2563EB", state_live="#15803D",
    state_warn="#B45309", state_error="#B91C1C", state_idle="#6B7280",
    cloud="#7E22CE",
)

#: Typography. Devanagari fallback is REQUIRED -- transcripts render mixed script
#: and a missing-glyph box on screen is a failure, not a cosmetic issue.
FONT_STACK = ("Inter", "Noto Sans", "DejaVu Sans", "TkDefaultFont")
MONO_STACK = ("JetBrains Mono", "DejaVu Sans Mono", "TkFixedFont")

TYPE = {
    "display":    (32, "bold"),
    "title":      (20, "bold"),
    "heading":    (16, "bold"),
    "body":       (14, "normal"),
    "transcript": (16, "normal"),   # larger: read at a glance, mid-meeting
    "label":      (12, "bold"),
    "mono":       (13, "normal"),
}

#: 8 px base grid.
SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "2xl": 32, "3xl": 48}

RADIUS = 4


def state_colour(palette: Palette, state: str) -> str:
    """Map a session state to its token.

    Note `state_live` is returned for exactly one state. The brief requires green
    to appear in one place at a time -- if everything is coloured, nothing reads.
    """
    return {
        "idle": palette.state_idle,
        "initializing": palette.accent,
        "listening": palette.state_live,
        "transcribing": palette.accent,
        "translating": palette.accent,
        "gating": palette.accent,
        "synthesizing": palette.accent,
        "outputting": palette.state_live,
        "suppressed": palette.state_warn,
        "paused": palette.state_warn,
        "muted": palette.state_warn,
        "degraded": palette.state_warn,
        "recovering": palette.state_warn,
        "stopping": palette.state_idle,
        "failed": palette.state_error,
    }.get(state, palette.state_idle)


#: Every state carries an icon and a label as well as a colour, so state is never
#: signalled by colour alone (accessibility, and glanceability).
STATE_GLYPH = {
    "idle": "○", "initializing": "◔", "listening": "●",
    "transcribing": "◔", "translating": "◔", "gating": "◔",
    "synthesizing": "◔", "outputting": "▶", "suppressed": "⊘",
    "paused": "⏸", "muted": "✕", "degraded": "⚠",
    "recovering": "↻", "stopping": "■", "failed": "✕",
}

STATE_LABEL = {
    "idle": "Idle", "initializing": "Starting", "listening": "Listening",
    "transcribing": "Recognising", "translating": "Translating",
    "gating": "Checking", "synthesizing": "Speaking", "outputting": "Speaking",
    "suppressed": "Withheld", "paused": "Paused", "muted": "Muted",
    "degraded": "Degraded", "recovering": "Recovering", "stopping": "Stopping",
    "failed": "Failed",
}
