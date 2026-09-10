"""The closed component set from docs/04-UIUX-BRIEF.md §5.

Nothing outside this module builds raw widgets. The brief's rule is that
components are not invented during implementation, and this file is where that
rule is enforced.
"""
from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

from .theme import (
    FONT_STACK,
    MONO_STACK,
    RADIUS,
    SPACE,
    STATE_GLYPH,
    STATE_LABEL,
    TYPE,
    Palette,
    state_colour,
)


def font(kind: str = "body", mono: bool = False) -> tuple:
    size, weight = TYPE[kind]
    family = MONO_STACK[0] if mono else FONT_STACK[0]
    return (family, size, weight)


class Card(tk.Frame):
    """Surface panel. No shadow -- shadows read as 'web page', not 'instrument'."""

    def __init__(self, parent, palette: Palette, *, title: str | None = None,
                 accent: str | None = None, **kw):
        super().__init__(parent, bg=palette.bg_surface,
                         highlightbackground=accent or palette.border,
                         highlightthickness=2 if accent else 1, bd=0, **kw)
        self.palette = palette
        self.body = self
        if title:
            header = tk.Label(self, text=title.upper(), bg=palette.bg_surface,
                              fg=palette.text_secondary, font=font("label"),
                              anchor="w")
            header.pack(fill="x", padx=SPACE["lg"], pady=(SPACE["md"], SPACE["xs"]))
            self.body = tk.Frame(self, bg=palette.bg_surface)
            self.body.pack(fill="both", expand=True,
                           padx=SPACE["lg"], pady=(0, SPACE["md"]))

    def set_accent(self, colour: str | None) -> None:
        self.configure(highlightbackground=colour or self.palette.border,
                       highlightthickness=2 if colour else 1)


class StatusPill(tk.Frame):
    """Dot + glyph + label. Never colour alone."""

    def __init__(self, parent, palette: Palette, state: str = "idle", **kw):
        super().__init__(parent, bg=palette.bg_surface, **kw)
        self.palette = palette
        self._glyph = tk.Label(self, bg=palette.bg_surface, font=font("body"))
        self._glyph.pack(side="left")
        self._label = tk.Label(self, bg=palette.bg_surface, font=font("body"),
                               fg=palette.text_primary)
        self._label.pack(side="left", padx=(SPACE["xs"], 0))
        self.set_state(state)

    def set_state(self, state: str, detail: str | None = None) -> None:
        colour = state_colour(self.palette, state)
        self._glyph.configure(text=STATE_GLYPH.get(state, "○"), fg=colour)
        self._label.configure(text=detail or STATE_LABEL.get(state, state), fg=colour)


class Button(tk.Button):
    """primary | secondary | ghost | danger.

    A disabled button ALWAYS carries a tooltip naming what blocks it (AC-10.1) --
    a greyed-out control with no explanation is the most common way a UI wastes
    someone's time.
    """

    def __init__(self, parent, palette: Palette, text: str,
                 command: Callable | None = None, *, variant: str = "secondary",
                 size: str = "md", **kw):
        self.palette, self.variant = palette, variant
        bg, fg, active = self._colours(palette, variant)
        pad_x = {"sm": SPACE["md"], "md": SPACE["lg"], "lg": SPACE["xl"]}[size]
        pad_y = {"sm": SPACE["xs"], "md": SPACE["sm"], "lg": SPACE["md"]}[size]
        super().__init__(parent, text=text, command=command, bg=bg, fg=fg,
                         activebackground=active, activeforeground=fg,
                         disabledforeground=palette.text_muted,
                         font=font("body"), relief="flat", bd=0,
                         padx=pad_x, pady=pad_y, cursor="hand2",
                         highlightthickness=0, **kw)
        self._tooltip: Tooltip | None = None

    @staticmethod
    def _colours(p: Palette, variant: str) -> tuple[str, str, str]:
        return {
            "primary": (p.accent, "#FFFFFF", p.accent),
            "secondary": (p.bg_raised, p.text_primary, p.border),
            "ghost": (p.bg_surface, p.text_secondary, p.bg_raised),
            "danger": (p.state_error, "#FFFFFF", p.state_error),
        }[variant]

    def set_enabled(self, enabled: bool, *, reason: str | None = None) -> None:
        self.configure(state="normal" if enabled else "disabled")
        if not enabled and reason:
            if self._tooltip is None:
                self._tooltip = Tooltip(self, self.palette, reason)
            else:
                self._tooltip.set_text(reason)
        elif self._tooltip is not None:
            self._tooltip.set_text("")


class Tooltip:
    def __init__(self, widget, palette: Palette, text: str):
        self.widget, self.palette, self.text = widget, palette, text
        self._window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def set_text(self, text: str) -> None:
        self.text = text

    def _show(self, _event=None) -> None:
        if not self.text or self._window is not None:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._window = tk.Toplevel(self.widget)
        self._window.wm_overrideredirect(True)
        self._window.wm_geometry(f"+{x}+{y}")
        tk.Label(self._window, text=self.text, bg=self.palette.bg_raised,
                 fg=self.palette.text_primary, font=font("body"),
                 padx=SPACE["sm"], pady=SPACE["xs"],
                 highlightbackground=self.palette.border,
                 highlightthickness=1).pack()

    def _hide(self, _event=None) -> None:
        if self._window is not None:
            self._window.destroy()
            self._window = None


class MetricReadout(tk.Frame):
    """Mono value + unit + label.

    Shows an em dash when the value is unknown. It NEVER shows a placeholder
    number -- the whole product's credibility rests on not inventing figures.
    Tabular-style mono font so digits do not jitter as values update, because a
    jittering number pulls the eye during a meeting.
    """

    def __init__(self, parent, palette: Palette, label: str, unit: str = "", **kw):
        super().__init__(parent, bg=palette.bg_surface, **kw)
        self.palette, self.unit = palette, unit
        self._value = tk.Label(self, text="—", bg=palette.bg_surface,
                               fg=palette.text_primary, font=font("title", mono=True),
                               anchor="w")
        self._value.pack(anchor="w")
        tk.Label(self, text=label.upper(), bg=palette.bg_surface,
                 fg=palette.text_muted, font=font("label"), anchor="w").pack(anchor="w")

    def set_value(self, value: float | int | str | None, *,
                  colour: str | None = None) -> None:
        if value is None:
            self._value.configure(text="—", fg=self.palette.text_muted)
            return
        text = f"{value:.0f}{self.unit}" if isinstance(value, (int, float)) else str(value)
        self._value.configure(text=text, fg=colour or self.palette.text_primary)


class LevelMeter(tk.Canvas):
    """Horizontal meter, −60→0 dBFS, with peak hold and a red zone above −3."""

    def __init__(self, parent, palette: Palette, width: int = 200, height: int = 8, **kw):
        super().__init__(parent, width=width, height=height, bg=palette.bg_raised,
                         highlightthickness=0, bd=0, **kw)
        # NOTE: never name these _w/_h -- Tkinter uses self._w internally for
        # the widget path, and overwriting it breaks the widget.
        self.palette, self._width, self._height = palette, width, height
        self._peak = 0.0
        self._bar = self.create_rectangle(0, 0, 0, height, fill=palette.state_live,
                                          width=0)
        self._hold = self.create_rectangle(0, 0, 0, height,
                                           fill=palette.text_secondary, width=0)

    def set_level(self, rms: float) -> None:
        import math
        db = 20 * math.log10(max(rms, 1e-9))
        frac = max(0.0, min(1.0, (db + 60) / 60))
        self._peak = max(frac, self._peak * 0.92)
        colour = (self.palette.state_error if db > -3
                  else self.palette.state_warn if db > -12
                  else self.palette.state_live)
        self.coords(self._bar, 0, 0, self._width * frac, self._height)
        self.itemconfigure(self._bar, fill=colour)
        x = self._width * self._peak
        self.coords(self._hold, x - 2, 0, x, self._height)


class Banner(tk.Frame):
    """info | warn | error. Dismissible unless it blocks."""

    def __init__(self, parent, palette: Palette, **kw):
        super().__init__(parent, bg=palette.bg_raised, **kw)
        self.palette = palette
        self._label = tk.Label(self, text="", bg=palette.bg_raised,
                               fg=palette.text_primary, font=font("body"),
                               anchor="w", justify="left", wraplength=760)
        self._label.pack(side="left", fill="x", expand=True,
                         padx=SPACE["md"], pady=SPACE["sm"])
        self._close = tk.Label(self, text="✕", bg=palette.bg_raised,
                               fg=palette.text_muted, font=font("body"),
                               cursor="hand2")
        self._close.pack(side="right", padx=SPACE["md"])
        self._close.bind("<Button-1>", lambda _e: self.hide())

    def show(self, text: str, level: str = "info") -> None:
        colour = {"info": self.palette.accent, "warn": self.palette.state_warn,
                  "error": self.palette.state_error}[level]
        self._label.configure(text=text, fg=colour)
        self.pack(fill="x", padx=SPACE["xl"], pady=(0, SPACE["sm"]))

    def hide(self) -> None:
        self.pack_forget()


class LatencyBar(tk.Canvas):
    """Stacked per-stage segments. The single visualisation in the product."""

    STAGE_ORDER = ["capture", "vad", "stt", "context", "translate", "tts",
                   "buffer", "output"]

    def __init__(self, parent, palette: Palette, width: int = 320, height: int = 18, **kw):
        super().__init__(parent, width=width, height=height, bg=palette.bg_raised,
                         highlightthickness=0, bd=0, **kw)
        # NOTE: never name these _w/_h -- Tkinter uses self._w internally for
        # the widget path, and overwriting it breaks the widget.
        self.palette, self._width, self._height = palette, width, height

    def set_stages(self, stages: dict[str, float], budget_ms: float) -> None:
        self.delete("all")
        total = sum(stages.values())
        if total <= 0:
            return
        shades = [self.palette.accent, self.palette.state_live,
                  self.palette.text_secondary, self.palette.cloud]
        x = 0.0
        scale = self._width / max(total, budget_ms)
        for i, name in enumerate([s for s in self.STAGE_ORDER if s in stages]):
            w = stages[name] * scale
            self.create_rectangle(x, 0, x + w, self._height,
                                  fill=shades[i % len(shades)], width=0)
            x += w
        # Budget marker: shows at a glance whether the bar has room left.
        bx = budget_ms * scale
        if bx < self._width:
            self.create_line(bx, 0, bx, self._height,
                             fill=self.palette.state_error, width=2)


class EmptyState(tk.Frame):
    """Icon + one sentence + one action. Never a bare blank panel."""

    def __init__(self, parent, palette: Palette, glyph: str, message: str,
                 action: tuple[str, Callable] | None = None, **kw):
        super().__init__(parent, bg=palette.bg_surface, **kw)
        tk.Label(self, text=glyph, bg=palette.bg_surface, fg=palette.text_muted,
                 font=(FONT_STACK[0], 28)).pack(pady=(SPACE["xl"], SPACE["sm"]))
        tk.Label(self, text=message, bg=palette.bg_surface,
                 fg=palette.text_secondary, font=font("body"),
                 wraplength=380, justify="center").pack()
        if action:
            Button(self, palette, action[0], action[1], variant="primary").pack(
                pady=SPACE["lg"])


class DiagnosticRow(tk.Frame):
    def __init__(self, parent, palette: Palette, name: str, **kw):
        super().__init__(parent, bg=palette.bg_surface, **kw)
        self.palette = palette
        self._pill = StatusPill(self, palette, "idle")
        self._pill.pack(side="left")
        tk.Label(self, text=name, bg=palette.bg_surface, fg=palette.text_primary,
                 font=font("body"), anchor="w", width=26).pack(side="left",
                                                               padx=SPACE["md"])
        self._detail = tk.Label(self, text="", bg=palette.bg_surface,
                                fg=palette.text_secondary, font=font("body"),
                                anchor="w")
        self._detail.pack(side="left", fill="x", expand=True)
        self._time = tk.Label(self, text="", bg=palette.bg_surface,
                              fg=palette.text_muted, font=font("mono", mono=True))
        self._time.pack(side="right")

    def set_result(self, status: str, detail: str, duration_ms: float) -> None:
        mapped = {"pass": "listening", "fail": "failed", "skip": "idle"}[status]
        label = {"pass": "Pass", "fail": "Fail", "skip": "Skip"}[status]
        self._pill.set_state(mapped, label)
        self._detail.configure(text=detail)
        self._time.configure(text=f"{duration_ms:.0f} ms")
