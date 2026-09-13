"""UI design-system and screen tests.

Rendering tests are skipped without a display. The theme rules are enforced
unconditionally, because the brief's whole point is that the design system is
decided up front rather than drifting during implementation.
"""
from __future__ import annotations

import os

import pytest

from vaani.ui.theme import (
    DARK,
    LIGHT,
    SPACE,
    STATE_GLYPH,
    STATE_LABEL,
    TYPE,
    state_colour,
)

def _check_display() -> bool:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        import tkinter as _tk
        _root = _tk.Tk()
        _root.destroy()
        return True
    except Exception:
        return False

HAS_DISPLAY = _check_display()
needs_display = pytest.mark.skipif(not HAS_DISPLAY, reason="no usable display")


# --- design tokens ----------------------------------------------------------

def test_both_themes_define_every_token():
    assert set(DARK.__dataclass_fields__) == set(LIGHT.__dataclass_fields__)


def test_no_token_is_empty():
    for palette in (DARK, LIGHT):
        for name in palette.__dataclass_fields__:
            value = getattr(palette, name)
            assert value.startswith("#") and len(value) == 7, name


def test_every_state_has_a_colour_glyph_and_label():
    """State must never be signalled by colour alone (accessibility + glance)."""
    states = ["idle", "initializing", "listening", "transcribing", "translating",
              "gating", "synthesizing", "outputting", "suppressed", "paused",
              "muted", "degraded", "recovering", "stopping", "failed"]
    for s in states:
        assert state_colour(DARK, s).startswith("#")
        assert s in STATE_GLYPH and STATE_GLYPH[s]
        assert s in STATE_LABEL and STATE_LABEL[s]


def test_live_green_is_reserved_for_active_speech():
    """The brief: green appears in one place at a time, or it stops meaning anything."""
    live = [s for s in STATE_LABEL if state_colour(DARK, s) == DARK.state_live]
    assert set(live) == {"listening", "outputting"}


def test_cloud_purple_is_never_a_decorative_colour():
    """Purple must mean exactly one thing: data leaving the machine."""
    for palette in (DARK, LIGHT):
        others = [getattr(palette, f) for f in palette.__dataclass_fields__
                  if f != "cloud"]
        assert palette.cloud not in others


def test_spacing_follows_the_8px_grid():
    for name, value in SPACE.items():
        assert value % 4 == 0, f"{name}={value} breaks the grid"


def test_transcript_type_is_larger_than_body():
    """Transcripts are read at a glance mid-meeting."""
    assert TYPE["transcript"][0] > TYPE["body"][0]


def test_body_text_is_at_least_14px():
    assert TYPE["body"][0] >= 14


def test_dark_is_the_default_palette():
    from vaani.ui import app as app_module
    import inspect
    sig = inspect.signature(app_module.VaaniApp.__init__)
    assert sig.parameters["palette"].default is DARK


# --- rendering --------------------------------------------------------------

@needs_display
def test_app_builds_all_five_screens():
    from vaani.ui.app import VaaniApp
    app = VaaniApp()
    try:
        assert set(app._screens) == {"dashboard", "meeting", "voice", "settings",
                                     "diagnostics"}
    finally:
        app.root.destroy()


@needs_display
def test_every_screen_renders():
    from vaani.ui.app import VaaniApp
    app = VaaniApp()
    try:
        for name in app._screens:
            app.show(name)
            app.root.update()
    finally:
        app.root.destroy()


@needs_display
def test_readiness_reports_every_precondition():
    from vaani.ui.app import VaaniApp
    app = VaaniApp()
    try:
        checks = app.readiness()
        assert {"Microphone", "Voice profile", "Audio system", "Models"} <= set(checks)
        for ok, reason in checks.values():
            assert isinstance(ok, bool) and reason
    finally:
        app.root.destroy()


@needs_display
def test_disabled_start_button_states_a_reason():
    """AC-10.1: a greyed-out control with no explanation wastes the user's time."""
    from vaani.ui.app import VaaniApp
    app = VaaniApp()
    try:
        dash = app._screens["dashboard"]
        dash.update_readiness({
            "Microphone": (False, "no capture device found"),
            "Voice profile": (True, "ok"),
            "Audio system": (True, "ok"),
            "Models": (True, "ok"),
        })
        assert str(dash.start["state"]) == "disabled"
        assert dash.start._tooltip is not None
        assert "capture device" in dash.start._tooltip.text
    finally:
        app.root.destroy()


@needs_display
def test_missing_voice_profile_does_not_block_start():
    """The fallback voice works and identifies itself; blocking would be wrong."""
    from vaani.ui.app import VaaniApp
    app = VaaniApp()
    try:
        dash = app._screens["dashboard"]
        dash.update_readiness({
            "Microphone": (True, "ok"), "Voice profile": (False, "not enrolled"),
            "Audio system": (True, "ok"), "Models": (True, "ok"),
        })
        assert str(dash.start["state"]) == "normal"
    finally:
        app.root.destroy()


@needs_display
def test_metric_readout_shows_a_dash_not_a_fake_number():
    from vaani.ui.app import VaaniApp
    from vaani.ui.components import MetricReadout
    app = VaaniApp()
    try:
        m = MetricReadout(app.root, DARK, "latency", " ms")
        assert m._value["text"] == "—"
        m.set_value(2411)
        assert m._value["text"] == "2411 ms"
        m.set_value(None)
        assert m._value["text"] == "—"
    finally:
        app.root.destroy()


@needs_display
def test_suppressed_utterance_is_shown_as_withheld():
    import time

    import numpy as np

    from vaani.core.types import (
        SuppressionReason,
        Transcript,
        Utterance,
        UtteranceResult,
    )
    from vaani.ui.app import VaaniApp
    app = VaaniApp()
    try:
        u = Utterance(session_id="s", seq=1, audio=np.zeros(4, dtype=np.float32),
                      sample_rate=16000, speech_end_time=time.monotonic(),
                      speech_ms=500)
        r = UtteranceResult(utterance=u)
        r.transcript = Transcript(text="कुछ", language_distribution={"hi": 1.0},
                                  confidence=0.3)
        r.suppressed = True
        r.suppression_reason = SuppressionReason.LOW_STT_CONFIDENCE
        screen = app._screens["meeting"]
        screen.add_result(r)
        text = screen.transcript.get("1.0", "end")
        assert "withheld" in text and "low_stt_confidence" in text
    finally:
        app.root.destroy()


# --- consent and enrollment in the GUI --------------------------------------

@needs_display
def test_voice_screen_offers_consent_when_none_is_recorded(tmp_path, monkeypatch):
    """Before this, the screen told the user to go and run a CLI command."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    from vaani.ui.app import VaaniApp
    app = VaaniApp()
    try:
        voice = app._screens["voice"]
        voice.render()
        assert hasattr(voice, "_consent") and hasattr(voice, "_enroll")
    finally:
        app.root.destroy()


@needs_display
def test_consent_requires_a_name_and_an_explicit_tick():
    """AC-09.1: consent must be an affirmative act, not a default.

    This replaced a required typed phrase. The phrase was silently blocking a
    user who HAD consented -- the <KeyRelease> binding behind it missed pasted
    and input-method text, so the field looked correct while the button stayed
    disabled with no explanation.
    """
    from vaani.ui.app import VaaniApp
    from vaani.ui.enrollment_flow import ConsentDialog
    app = VaaniApp()
    try:
        d = ConsentDialog(app.root, app.palette, "0.1.0", lambda s: None)
        assert str(d.confirm["state"]) == "disabled"
        assert d.agreed.get() is False          # never pre-ticked

        d.subject_var.set("Kanak")
        assert str(d.confirm["state"]) == "disabled"   # name alone is not consent

        d.agreed.set(True)
        d._validate()
        assert str(d.confirm["state"]) == "normal"
        d._cancel()
    finally:
        app.root.destroy()


@needs_display
def test_consent_tick_alone_is_not_enough_without_a_name():
    from vaani.ui.app import VaaniApp
    from vaani.ui.enrollment_flow import ConsentDialog
    app = VaaniApp()
    try:
        d = ConsentDialog(app.root, app.palette, "0.1.0", lambda s: None)
        d.agreed.set(True)
        d._validate()
        assert str(d.confirm["state"]) == "disabled"
        d._cancel()
    finally:
        app.root.destroy()


@needs_display
def test_consent_dialog_says_what_is_missing():
    """The previous version just stayed disabled and explained nothing."""
    from vaani.ui.app import VaaniApp
    from vaani.ui.enrollment_flow import ConsentDialog
    app = VaaniApp()
    try:
        d = ConsentDialog(app.root, app.palette, "0.1.0", lambda s: None)
        assert "tick" in d.error["text"].lower()
        d.subject_var.set("Kanak")
        assert "tick" in d.error["text"].lower()
        d.agreed.set(True)
        d._validate()
        assert d.error["text"] == ""
        d._cancel()
    finally:
        app.root.destroy()


@needs_display
def test_typed_input_updates_state_without_key_events():
    """Regression: paste and IME input never fire <KeyRelease>."""
    from vaani.ui.app import VaaniApp
    from vaani.ui.enrollment_flow import ConsentDialog
    app = VaaniApp()
    try:
        d = ConsentDialog(app.root, app.palette, "0.1.0", lambda s: None)
        d.subject.insert(0, "Kanak")        # no key event generated
        app.root.update()
        assert "tick" in d.error["text"].lower()   # trace fired anyway
        d._cancel()
    finally:
        app.root.destroy()


@needs_display
def test_enrollment_finish_is_blocked_until_enough_speech():
    """AC-09.3: progress is speech-seconds, not wall-clock."""
    from vaani.ui.app import VaaniApp
    from vaani.ui.enrollment_flow import EnrollmentDialog
    app = VaaniApp()
    try:
        d = EnrollmentDialog(app.root, app.palette, lambda p: None)
        assert str(d.finish["state"]) == "disabled"
        d._speech_s = d.TARGET_SPEECH_S
        d._tick(0.1)
        assert str(d.finish["state"]) == "normal"
        d._cancel()
    finally:
        app.root.destroy()


@needs_display
def test_enrollment_progress_counts_speech_not_wall_clock():
    from vaani.ui.app import VaaniApp
    from vaani.ui.enrollment_flow import EnrollmentDialog
    app = VaaniApp()
    try:
        d = EnrollmentDialog(app.root, app.palette, lambda p: None)
        d._speech_s = 12.0
        d._tick(0.05)
        assert "12 s of speech" in d.progress["text"]
        d._cancel()
    finally:
        app.root.destroy()
