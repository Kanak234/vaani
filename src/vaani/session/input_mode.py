"""Who is allowed to speak into the meeting, and in whose voice.

WHY THIS IS A SEPARATE MODULE
-----------------------------
Phase 2 adds three things that all answer the same question -- "should this
frame of microphone audio become translated speech, raw speech, or nothing?" --
but that arrive from three different places: a CLI flag, a hotkey, and an IPC
command from the overlay. Putting that logic in the capture loop would mean
testing it through a real PulseAudio stream, which is exactly the kind of thing
that is only ever tested by hand and therefore only ever tested once.

So the decision is a pure state machine here, with no audio and no threads, and
``session/engine.py`` asks it per frame. Every combination below is covered by
a unit test that runs in microseconds.

THE TWO OUTPUTS
---------------
``translating_enabled``  -> capture feeds VAD/STT and the meeting hears the
                            English clone of what the user said.
``passthrough_enabled``  -> raw microphone frames go straight to the virtual
                            mic and the meeting hears the user's ACTUAL voice,
                            untranslated.

They are mutually exclusive by construction. Both being true would put two
overlapping voices into the meeting -- the user's Hindi and its English clone
at the same time -- which is the single worst thing this component could do.
:meth:`InputModeController.state` enforces that in one place rather than
leaving each caller to remember it.

WHY BYPASS IS ORTHOGONAL TO THE MODE
------------------------------------
Bypass is not a fourth mode. It is a live override that has to work instantly
from whatever mode is active, because of what it is FOR: the user is in a
meeting, the translation is wrong or the moment is urgent, and they need their
own voice in the room right now. Making it a mode would mean remembering and
restoring the previous mode on the way out; making it a flag means releasing it
returns to exactly what was there before, with no state to lose.

WHERE THE DATA FLOWS NEXT
-------------------------
``engine._capture_loop`` consults :meth:`state` for every frame. On a
transition into bypass the engine also flushes queued TTS audio -- see
``TranslationSession.set_bypass`` -- because a half-spoken AI sentence
continuing underneath the user's real voice is the same two-voices failure.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class InputMode(Enum):
    """How the user signals that they want to be translated."""

    #: Everything the VAD detects is translated. The default: in a normal
    #: meeting the user simply talks, and having to hold a key to be understood
    #: defeats the point of the product.
    ALWAYS_ON = "always_on"

    #: Press once to start being translated, press again to stop. WHY THIS IS
    #: THE DEFAULT PTT rather than hold: it needs no special permissions. A
    #: desktop shortcut running `vaani ptt` is enough, whereas true hold-to-talk
    #: needs raw device access (the `input` group) that a fresh install does not
    #: have.
    PTT_TOGGLE = "ptt_toggle"

    #: Translated only while the key is physically held. Closest to a radio,
    #: and the least likely to leave the user accidentally live -- but it
    #: requires evdev access, so it is opt-in.
    PTT_HOLD = "ptt_hold"


@dataclass(frozen=True, slots=True)
class InputState:
    """The resolved answer for one frame.

    Frozen because it is read from the capture thread while the control thread
    may be mutating the controller: handing out an immutable snapshot means a
    hotkey pressed mid-frame can never produce a half-applied state where both
    translation and passthrough look enabled.
    """
    mode: InputMode
    translating_enabled: bool
    passthrough_enabled: bool
    bypass: bool
    talking: bool

    @property
    def label(self) -> str:
        """Short text for the overlay. The UI brief asks for a single word the
        user can read at a glance without parsing three booleans."""
        if self.bypass:
            return "BYPASS"
        if self.translating_enabled:
            return "LIVE"
        return "MUTED"


class InputModeController:
    """The state machine. No audio, no threads, no I/O.

    Thread-safety note: the individual attribute writes here are single
    assignments of immutable values, which CPython performs atomically, and
    :meth:`state` reads them into one snapshot. There is deliberately no lock:
    a lock on the per-frame path would be contended 50 times a second for no
    benefit, and the worst possible race is one frame resolved against the
    previous state -- 20 ms, inaudible.
    """

    def __init__(self, mode: InputMode = InputMode.ALWAYS_ON, *,
                 on_change: Callable[[InputState], None] | None = None) -> None:
        self._mode = mode
        # Only meaningful in the two PTT modes. Starting False means a session
        # launched in PTT mode is silent until the user asks to be heard, which
        # is the safe direction: the failure mode of starting True is that the
        # user is translated into a meeting without realising it.
        self._talking = False
        self._bypass = False
        self._on_change = on_change

    # ------------------------------------------------------------- queries

    @property
    def mode(self) -> InputMode:
        return self._mode

    @property
    def bypass(self) -> bool:
        return self._bypass

    @property
    def talking(self) -> bool:
        return self._talking

    def state(self) -> InputState:
        """Resolve mode + talking + bypass into one consistent snapshot.

        THE INVARIANT lives here and nowhere else: translation and passthrough
        are never both enabled.
        """
        if self._bypass:
            # Bypass wins over everything, including an active PTT press. The
            # user asked for their own voice in the room; giving them a
            # translation as well would be actively harmful.
            return InputState(mode=self._mode, translating_enabled=False,
                              passthrough_enabled=True, bypass=True,
                              talking=self._talking)

        if self._mode is InputMode.ALWAYS_ON:
            translating = True
        else:
            # Both PTT modes gate on the same flag; what differs is only how
            # that flag is driven (toggle() vs press()/release()), which is the
            # caller's business, not this method's.
            translating = self._talking

        return InputState(mode=self._mode, translating_enabled=translating,
                          passthrough_enabled=False, bypass=False,
                          talking=self._talking)

    # -------------------------------------------------------------- control

    def set_mode(self, mode: InputMode) -> InputState:
        """Switch modes mid-session.

        Resets ``talking`` on every switch. WHY: leaving it set would mean
        switching from ALWAYS_ON to PTT_TOGGLE silently leaves the user live,
        which is the opposite of what someone reaching for push-to-talk wants.
        """
        self._mode = mode
        self._talking = False
        return self._changed()

    def toggle(self) -> InputState:
        """PTT_TOGGLE: flip between live and muted.

        Deliberately works in ALWAYS_ON too, where it acts as a plain mute --
        the user pressing their talk key expects *something* to happen, and
        doing nothing would read as a broken hotkey.
        """
        self._talking = not self._talking
        return self._changed()

    def press(self) -> InputState:
        """PTT_HOLD: key down."""
        self._talking = True
        return self._changed()

    def release(self) -> InputState:
        """PTT_HOLD: key up."""
        self._talking = False
        return self._changed()

    def set_bypass(self, enabled: bool) -> InputState:
        self._bypass = bool(enabled)
        return self._changed()

    def toggle_bypass(self) -> InputState:
        self._bypass = not self._bypass
        return self._changed()

    # -------------------------------------------------------------- internal

    def _changed(self) -> InputState:
        state = self.state()
        if self._on_change is not None:
            # A broken UI callback must never break the control path -- the
            # same containment rule the supervisor applies to the worker.
            try:
                self._on_change(state)
            except Exception:
                pass
        return state
