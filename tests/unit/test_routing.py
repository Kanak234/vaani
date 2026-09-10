"""Automatic microphone routing.

Built after a real failure: the user made a live WhatsApp call, Vaani reported
healthy, and the other side heard untranslated Hindi -- because WhatsApp was
still on the real microphone. Nothing surfaced that.
"""
from __future__ import annotations

import pytest

from vaani.core.errors import ErrorCode, VaaniError
from vaani.devices.routing import (
    AutoRouter,
    CaptureStream,
    _is_vaani_stream,
    move_to_virtual_mic,
)


def stream(app="Zoom", *, index="10", source="5", own=False, on_virtual=False):
    return CaptureStream(index=index, source_index=source, app_name=app,
                         media_name="", is_own=own, on_virtual_mic=on_virtual)


# --- the feedback-loop guard ------------------------------------------------

def test_vaanis_own_stream_is_never_movable():
    """Moving our own capture onto our own virtual mic would loop output back
    into recognition and compound without limit."""
    assert stream("Vaani", own=True).movable is False


def test_moving_our_own_stream_raises():
    with pytest.raises(VaaniError) as e:
        move_to_virtual_mic(stream("Vaani", own=True))
    assert e.value.code is ErrorCode.FEEDBACK_LOOP_DETECTED
    assert not e.value.retryable


@pytest.mark.parametrize("props", [
    {"application.name": "Vaani"},
    {"node.name": "input.VaaniVirtualMic"},
    {"node.group": "remap-source-536870916"},
    {"node.link-group": "loopback-2285-13"},
    {"device.description": "Vaani_Virtual_Microphone"},
])
def test_own_plumbing_is_recognised(props):
    assert _is_vaani_stream(props) is True


@pytest.mark.parametrize("props", [
    {"application.name": "Zoom"},
    {"application.name": "WhatsApp"},
    {"application.name": "Chromium"},
])
def test_other_applications_are_not_ours(props):
    assert _is_vaani_stream(props) is False


# --- what may be moved ------------------------------------------------------

def test_a_meeting_app_on_a_real_mic_is_movable():
    assert stream("Zoom").movable is True


def test_an_app_already_on_the_virtual_mic_is_left_alone():
    assert stream("Zoom", on_virtual=True).movable is False


@pytest.mark.parametrize("app", ["pavucontrol", "Audacity", "parec"])
def test_non_meeting_capture_tools_are_never_moved(app):
    """Hijacking a recording app's input would be surprising and wrong."""
    assert stream(app).movable is False


# --- the router -------------------------------------------------------------

def test_router_moves_new_streams_once(monkeypatch):
    calls = []
    streams = [stream("Zoom", index="10")]
    monkeypatch.setattr("vaani.devices.routing.list_capture_streams",
                        lambda: streams)
    monkeypatch.setattr("vaani.devices.routing.move_to_virtual_mic",
                        lambda s: calls.append(s.index))
    r = AutoRouter()
    assert [s.app_name for s in r.sweep()] == ["Zoom"]
    assert r.sweep() == []          # already moved; not moved twice
    assert calls == ["10"]


def test_router_catches_an_app_that_starts_later(monkeypatch):
    """A meeting client opens its stream when the CALL starts, after Vaani is
    already running -- a one-shot sweep at startup would miss it."""
    streams: list = []
    monkeypatch.setattr("vaani.devices.routing.list_capture_streams",
                        lambda: streams)
    monkeypatch.setattr("vaani.devices.routing.move_to_virtual_mic", lambda s: None)
    r = AutoRouter()
    assert r.sweep() == []
    streams.append(stream("WhatsApp", index="20"))
    assert [s.app_name for s in r.sweep()] == ["WhatsApp"]


def test_router_records_the_original_source_for_restore(monkeypatch):
    monkeypatch.setattr("vaani.devices.routing.list_capture_streams",
                        lambda: [stream("Zoom", index="10", source="7")])
    monkeypatch.setattr("vaani.devices.routing.move_to_virtual_mic", lambda s: None)
    r = AutoRouter()
    r.sweep()
    assert r.moved == {"10": "7"}


def test_restore_puts_applications_back(monkeypatch):
    moved_back = []
    monkeypatch.setattr("vaani.devices.routing._pactl",
                        lambda *a: moved_back.append(a) or "")
    r = AutoRouter(real_source="alsa_input.real")
    r.moved = {"10": "7"}
    assert r.restore_all() == 1
    assert r.moved == {}
    assert moved_back[0][1:] == ("10", "alsa_input.real")


def test_router_survives_a_failed_move(monkeypatch):
    """One stubborn app must not stop the others being switched."""
    def boom(s):
        if s.app_name == "Bad":
            raise VaaniError(code=ErrorCode.VIRTUAL_MIC_UNAVAILABLE,
                             message="nope")
    monkeypatch.setattr("vaani.devices.routing.list_capture_streams",
                        lambda: [stream("Bad", index="1"), stream("Zoom", index="2")])
    monkeypatch.setattr("vaani.devices.routing.move_to_virtual_mic", boom)
    r = AutoRouter()
    assert [s.app_name for s in r.sweep()] == ["Zoom"]
    assert r.last_error == "nope"


# --- crash recovery ---------------------------------------------------------
# From a real incident: Vaani was killed rather than stopped, apps were left on
# a virtual mic that no longer existed, and the user's microphone appeared
# broken system-wide with nothing on screen linking it to this program.

def test_moves_are_persisted_immediately(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr("vaani.devices.routing.list_capture_streams",
                        lambda: [stream("Zoom", index="10", source="7")])
    monkeypatch.setattr("vaani.devices.routing.move_to_virtual_mic", lambda s: None)
    monkeypatch.setattr("vaani.devices.routing._default_source", lambda: "real")
    from vaani.devices.routing import AutoRouter, _state_path
    AutoRouter().sweep()
    import json
    assert json.loads(_state_path().read_text())["moved"] == {"10": "7"}


def test_startup_reclaims_a_crashed_sessions_routing(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr("vaani.devices.routing._pactl",
                        lambda *a: calls.append(a) or "")
    from vaani.devices.routing import _state_path, reclaim_orphaned_routing
    import json
    _state_path().parent.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps({"real_source": "real", "moved": {"10": "7"}}))

    assert reclaim_orphaned_routing() == 1
    assert calls[0] == ("move-source-output", "10", "real")
    assert not _state_path().exists()      # not reclaimed twice


def test_reclaim_is_safe_with_no_state(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    from vaani.devices.routing import reclaim_orphaned_routing
    assert reclaim_orphaned_routing() == 0


def test_reclaim_survives_a_corrupt_state_file(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    from vaani.devices.routing import _state_path, reclaim_orphaned_routing
    _state_path().parent.mkdir(parents=True, exist_ok=True)
    _state_path().write_text("{ not json")
    assert reclaim_orphaned_routing() == 0
    assert not _state_path().exists()


def test_new_router_reclaims_on_construction(monkeypatch, tmp_path):
    """A fresh start must clean up after a previous crash without being asked."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr("vaani.devices.routing._pactl", lambda *a: "")
    from vaani.devices.routing import AutoRouter, _state_path
    import json
    _state_path().parent.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps({"real_source": "real", "moved": {"10": "7"}}))
    AutoRouter(real_source="real")
    assert not _state_path().exists()


def test_restore_clears_the_state_file(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr("vaani.devices.routing._pactl", lambda *a: "")
    from vaani.devices.routing import AutoRouter, _state_path
    r = AutoRouter(real_source="real")
    r.moved = {"10": "7"}
    r._persist()
    assert _state_path().exists()
    r.restore_all()
    assert not _state_path().exists()
