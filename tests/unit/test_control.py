"""Control socket: how the app is driven while hidden during a meeting."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from vaani.ipc import control


@pytest.fixture
def sock(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    yield tmp_path / "vaani.sock"


@pytest.fixture
def server(sock):
    received = []

    def handler(cmd):
        received.append(cmd)
        return {"state": "listening", "visible": False}

    s = control.ControlServer(handler)
    s.start()
    s.received = received
    yield s
    s.stop()


def test_socket_is_created_in_the_runtime_dir(server, sock):
    assert sock.exists()


def test_socket_is_private_to_this_user(server, sock):
    """Anyone who can write to it can drive the microphone."""
    assert oct(sock.stat().st_mode)[-3:] == "600"


def test_known_commands_are_dispatched(server):
    for cmd in ("show", "hide", "toggle", "pause", "mute", "status"):
        reply = control.send(cmd)
        assert reply["ok"] is True
    assert server.received[:3] == ["show", "hide", "toggle"]


def test_unknown_commands_are_rejected(server):
    reply = control.send("rm -rf /")
    assert reply["ok"] is False
    assert "unknown" in reply["error"]
    assert server.received == []      # never reached the handler


def test_command_vocabulary_is_closed():
    assert "eval" not in control.COMMANDS
    assert control.COMMANDS == {"show", "hide", "toggle", "pause", "resume",
                                "mute", "unmute", "panic", "status", "quit"}


def test_handler_exception_does_not_kill_the_server(sock):
    def boom(_cmd):
        raise RuntimeError("handler bug")
    s = control.ControlServer(boom)
    s.start()
    try:
        reply = control.send("show")
        assert reply["ok"] is False
        assert control.send("hide")["ok"] is False      # still serving
    finally:
        s.stop()


def test_send_without_a_server_raises(sock):
    with pytest.raises(ConnectionError):
        control.send("show")


def test_is_running_reflects_reality(sock, server):
    assert control.is_running() is True
    server.stop()
    assert control.is_running() is False


def test_stop_removes_the_socket(sock):
    s = control.ControlServer(lambda c: {})
    s.start()
    assert sock.exists()
    s.stop()
    assert not sock.exists()


def test_a_stale_socket_from_a_crash_is_reclaimed(sock):
    """A crash leaves the file behind; the next start must not be blocked."""
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.touch()
    s = control.ControlServer(lambda c: {})
    s.start()          # must not raise
    try:
        assert control.send("status")["ok"] is True
    finally:
        s.stop()


def test_second_instance_is_refused(sock, server):
    """Two instances would fight over the microphone."""
    second = control.ControlServer(lambda c: {})
    with pytest.raises(RuntimeError, match="already running"):
        second.start()
