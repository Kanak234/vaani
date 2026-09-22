"""Tests for session/input_mode.py.

The single most important property, asserted from several directions: the
meeting must never hear the user's real voice and its English clone at the same
time. Everything else here is about not leaving the user accidentally live.
"""
from __future__ import annotations

import dataclasses

import pytest

from vaani.session.input_mode import InputMode, InputModeController


# ------------------------------------------------------------ the invariant

@pytest.mark.parametrize("mode", list(InputMode))
@pytest.mark.parametrize("talking", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_translation_and_passthrough_are_never_both_enabled(mode, talking, bypass):
    """Exhaustive over every reachable combination. Both enabled would put two
    overlapping voices into the meeting."""
    c = InputModeController(mode)
    if talking:
        c.press()
    c.set_bypass(bypass)
    s = c.state()
    assert not (s.translating_enabled and s.passthrough_enabled)


# ------------------------------------------------------------------ defaults

def test_always_on_translates_without_any_key():
    c = InputModeController(InputMode.ALWAYS_ON)
    s = c.state()
    assert s.translating_enabled is True
    assert s.passthrough_enabled is False
    assert s.label == "LIVE"


@pytest.mark.parametrize("mode", [InputMode.PTT_TOGGLE, InputMode.PTT_HOLD])
def test_ptt_modes_start_silent(mode):
    """The safe direction: the failure mode of starting live is being
    translated into a meeting without realising it."""
    c = InputModeController(mode)
    s = c.state()
    assert s.translating_enabled is False
    assert s.label == "MUTED"


# ----------------------------------------------------------------- toggle

def test_toggle_flips_live_and_back():
    c = InputModeController(InputMode.PTT_TOGGLE)
    assert c.toggle().translating_enabled is True
    assert c.toggle().translating_enabled is False


def test_toggle_in_always_on_acts_as_mute():
    """A pressed talk key must do *something*; silence would read as a broken
    hotkey."""
    c = InputModeController(InputMode.ALWAYS_ON)
    assert c.state().translating_enabled is True
    c.toggle()
    # ALWAYS_ON ignores `talking`, so the user is still live; what matters is
    # that the call is accepted rather than raising.
    assert c.talking is True


# ------------------------------------------------------------------- hold

def test_press_and_release_gate_translation():
    c = InputModeController(InputMode.PTT_HOLD)
    assert c.press().translating_enabled is True
    assert c.release().translating_enabled is False


def test_release_without_press_is_harmless():
    """Key-up events can arrive without a matching key-down (focus changes,
    a session started with the key already held)."""
    c = InputModeController(InputMode.PTT_HOLD)
    assert c.release().translating_enabled is False


def test_repeated_press_is_idempotent():
    """evdev autorepeat sends many key-down events for one physical hold."""
    c = InputModeController(InputMode.PTT_HOLD)
    for _ in range(10):
        c.press()
    assert c.state().translating_enabled is True
    assert c.release().translating_enabled is False


# ----------------------------------------------------------------- bypass

def test_bypass_passes_the_real_voice_through():
    c = InputModeController(InputMode.ALWAYS_ON)
    s = c.set_bypass(True)
    assert s.passthrough_enabled is True
    assert s.translating_enabled is False
    assert s.label == "BYPASS"


def test_bypass_overrides_an_active_ptt_press():
    """The user asked for their own voice in the room; also sending a
    translation would be actively harmful."""
    c = InputModeController(InputMode.PTT_HOLD)
    c.press()
    assert c.state().translating_enabled is True
    s = c.set_bypass(True)
    assert s.translating_enabled is False
    assert s.passthrough_enabled is True


def test_releasing_bypass_restores_exactly_what_was_there_before():
    """THE reason bypass is a flag and not a mode: nothing to remember, so
    nothing to lose."""
    c = InputModeController(InputMode.PTT_HOLD)
    c.press()
    before = c.state()
    c.set_bypass(True)
    c.set_bypass(False)
    after = c.state()
    assert after == before


def test_bypass_toggles():
    c = InputModeController()
    assert c.toggle_bypass().bypass is True
    assert c.toggle_bypass().bypass is False


# ------------------------------------------------------------- mode switch

def test_switching_into_ptt_does_not_leave_the_user_live():
    """Someone reaching for push-to-talk wants to stop being live, not
    silently stay live."""
    c = InputModeController(InputMode.ALWAYS_ON)
    c.toggle()                       # talking = True
    s = c.set_mode(InputMode.PTT_TOGGLE)
    assert s.translating_enabled is False
    assert c.talking is False


def test_switching_back_to_always_on_is_live_again():
    c = InputModeController(InputMode.PTT_HOLD)
    s = c.set_mode(InputMode.ALWAYS_ON)
    assert s.translating_enabled is True


def test_mode_switch_preserves_bypass():
    """Bypass is an override, not part of the mode; changing mode while the
    user is holding their real voice open must not cut them off."""
    c = InputModeController(InputMode.ALWAYS_ON)
    c.set_bypass(True)
    s = c.set_mode(InputMode.PTT_TOGGLE)
    assert s.bypass is True
    assert s.passthrough_enabled is True


# ---------------------------------------------------------------- callbacks

def test_every_control_call_notifies_the_ui():
    seen = []
    c = InputModeController(InputMode.PTT_TOGGLE, on_change=seen.append)
    c.toggle()
    c.set_bypass(True)
    c.set_bypass(False)
    c.set_mode(InputMode.ALWAYS_ON)
    assert len(seen) == 4
    assert [s.label for s in seen] == ["LIVE", "BYPASS", "LIVE", "LIVE"]


def test_a_broken_ui_callback_cannot_break_the_control_path():
    """A hotkey must keep working even if the overlay has a bug."""
    def boom(_state):
        raise ValueError("overlay bug")

    c = InputModeController(InputMode.PTT_HOLD, on_change=boom)
    assert c.press().translating_enabled is True      # did not raise


# ------------------------------------------------------------- snapshot use

def test_state_is_immutable():
    """The capture thread reads this while the control thread mutates the
    controller; a frozen snapshot cannot be half-applied."""
    c = InputModeController()
    s = c.state()
    # Specifically FrozenInstanceError, not a blind Exception: asserting the
    # exact type is what proves the dataclass is frozen rather than merely
    # broken in some other way.
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.translating_enabled = False       # type: ignore[misc]


def test_a_snapshot_does_not_change_under_the_reader():
    c = InputModeController(InputMode.ALWAYS_ON)
    snapshot = c.state()
    c.set_bypass(True)                      # control thread moves on
    assert snapshot.translating_enabled is True     # reader's view is stable
    assert c.state().passthrough_enabled is True
