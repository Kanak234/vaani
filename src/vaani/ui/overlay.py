"""Compact meeting overlay (docs/04-UIUX-BRIEF.md §2, "compact mode").

Sized and positioned to sit beside a meeting window without covering faces, and
hideable entirely so nothing about the tool is visible on screen.

WAYLAND REALITY, stated because it changes how this must be driven:
An application cannot grab a global hotkey on Wayland. Tkinter's key bindings
only fire while the window has focus, and during a meeting focus is on Zoom. So
hide/show is driven from OUTSIDE, through the control socket in `ipc/control.py`:

    KDE shortcut  ->  `vaani hide`  ->  socket  ->  this window

`vaani meeting --hidden` starts with no window at all, which is the normal way to
use it: start before the call, and nothing appears unless you ask for it.
"""
from __future__ import annotations

import tkinter as tk

from .components import StatusPill, font
from .theme import DARK, SPACE, Palette, state_colour


class MeetingOverlay:
    """A small always-on-top status window that can vanish completely."""

    WIDTH, HEIGHT = 460, 132

    def __init__(self, root: tk.Tk, palette: Palette = DARK, *,
                 corner: str = "bottom-right", start_hidden: bool = True,
                 on_panic=None, on_toggle_pause=None) -> None:
        self.palette = palette
        self.root = root
        self._on_panic = on_panic
        self._on_toggle_pause = on_toggle_pause
        self._visible = False
        self._drag = (0, 0)

        p = palette
        self.win = tk.Toplevel(root)
        self.win.withdraw()
        self.win.title("Vaani")
        # No decorations: a title bar is the most recognisable part of an app,
        # and it is also what makes the window appear in the taskbar.
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg=p.bg_surface, highlightbackground=p.border,
                           highlightthickness=1)
        self.win.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self._place(corner)

        header = tk.Frame(self.win, bg=p.bg_surface)
        header.pack(fill="x", padx=SPACE["md"], pady=(SPACE["sm"], 0))
        self.state_pill = StatusPill(header, p, "idle")
        self.state_pill.pack(side="left")
        self.latency = tk.Label(header, text="—", bg=p.bg_surface,
                                fg=p.text_muted, font=font("mono", mono=True))
        self.latency.pack(side="right")

        self.line_src = tk.Label(self.win, text="", bg=p.bg_surface,
                                 fg=p.text_secondary, font=font("body"),
                                 anchor="w", wraplength=self.WIDTH - 24,
                                 justify="left")
        self.line_src.pack(fill="x", padx=SPACE["md"], pady=(SPACE["xs"], 0))
        self.line_out = tk.Label(self.win, text="Ready", bg=p.bg_surface,
                                 fg=p.text_primary, font=font("body"),
                                 anchor="w", wraplength=self.WIDTH - 24,
                                 justify="left")
        self.line_out.pack(fill="x", padx=SPACE["md"])

        controls = tk.Frame(self.win, bg=p.bg_surface)
        controls.pack(fill="x", padx=SPACE["md"], pady=SPACE["sm"])
        self._mini(controls, "Hide", self.hide).pack(side="left")
        self._mini(controls, "Pause", self._pause).pack(side="left", padx=SPACE["xs"])
        self._mini(controls, "STOP", self._panic, danger=True).pack(side="right")

        # Draggable by its body, since there is no title bar to grab.
        for widget in (self.win, header, self.line_src, self.line_out):
            widget.bind("<Button-1>", self._grab)
            widget.bind("<B1-Motion>", self._move)

        if not start_hidden:
            self.show()

    def _mini(self, parent, text, command, *, danger=False) -> tk.Button:
        p = self.palette
        return tk.Button(parent, text=text, command=command,
                         bg=p.state_error if danger else p.bg_raised,
                         fg="#FFFFFF" if danger else p.text_secondary,
                         font=font("label"), relief="flat", bd=0,
                         padx=SPACE["sm"], pady=2, cursor="hand2",
                         highlightthickness=0, activebackground=p.border)

    def _place(self, corner: str) -> None:
        self.win.update_idletasks()
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        margin = 24
        x = margin if "left" in corner else sw - self.WIDTH - margin
        y = margin if "top" in corner else sh - self.HEIGHT - margin - 48
        self.win.geometry(f"{self.WIDTH}x{self.HEIGHT}+{x}+{y}")

    def _grab(self, event) -> None:
        self._drag = (event.x_root - self.win.winfo_x(),
                      event.y_root - self.win.winfo_y())

    def _move(self, event) -> None:
        self.win.geometry(f"+{event.x_root - self._drag[0]}"
                          f"+{event.y_root - self._drag[1]}")

    # ------------------------------------------------------------ visibility

    @property
    def visible(self) -> bool:
        return self._visible

    def show(self) -> None:
        self.win.deiconify()
        self.win.attributes("-topmost", True)
        self._visible = True

    def hide(self) -> None:
        """Vanish completely — no window, no taskbar entry.

        Translation keeps running. This hides the interface, not the engine.
        """
        self.win.withdraw()
        self._visible = False

    def toggle(self) -> bool:
        self.hide() if self._visible else self.show()
        return self._visible

    # --------------------------------------------------------------- updates

    def set_state(self, state: str, latency_ms: float | None = None) -> None:
        self.state_pill.set_state(state)
        self.latency.configure(
            text=f"{latency_ms:.0f} ms" if latency_ms else "—",
            fg=state_colour(self.palette, state) if latency_ms else
            self.palette.text_muted)

    def set_lines(self, source: str, output: str, *, suppressed: bool = False) -> None:
        p = self.palette
        self.line_src.configure(text=source[:110])
        self.line_out.configure(
            text=(f"⊘ withheld — {output}" if suppressed else output)[:110],
            fg=p.state_warn if suppressed else p.text_primary)

    def _pause(self) -> None:
        if self._on_toggle_pause:
            self._on_toggle_pause()

    def _panic(self) -> None:
        if self._on_panic:
            self._on_panic()
