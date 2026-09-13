"""Control channel for a running Vaani instance.

WHY THIS EXISTS: on Wayland an application cannot grab a global hotkey. Tkinter's
`bind_all` only fires while the window has focus, which is useless during a
meeting — focus is on Zoom, not on us. KDE and GNOME can bind a shortcut to a
COMMAND, though, so the pattern that actually works is:

    [KDE shortcut] -> `vaani hide` -> unix socket -> running instance

The socket lives in the user's runtime directory with 0600 permissions, so only
this user can drive it. Commands are a fixed vocabulary; nothing is eval'd.
"""
from __future__ import annotations

import json
import os
import socket
import threading
from collections.abc import Callable
from pathlib import Path

#: Fixed command vocabulary. Anything else is rejected.
COMMANDS = frozenset({"show", "hide", "toggle", "pause", "resume", "mute",
                      "unmute", "panic", "status", "quit"})


from typing import Union

def socket_path() -> Union[Path, tuple[str, int]]:
    if os.name == 'nt':
        return ('127.0.0.1', 18923)
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/vaani-{os.getuid()}"
    return Path(base) / "vaani.sock"


class ControlServer:
    """Listens for commands from `vaani hide` and friends."""

    def __init__(self, handler: Callable[[str], dict]) -> None:
        self.handler = handler
        self.path = socket_path()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if os.name != 'nt':
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                try:
                    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    probe.settimeout(0.2)
                    probe.connect(str(self.path))
                    probe.close()
                    raise RuntimeError("another Vaani instance is already running")
                except (ConnectionRefusedError, socket.timeout, OSError):
                    self.path.unlink(missing_ok=True)
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.bind(str(self.path))
            os.chmod(self.path, 0o600)
        else:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self._sock.bind(self.path)
            except OSError:
                raise RuntimeError("another Vaani instance is already running")

        self._sock.listen(4)
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="vaani-control")
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                return
            with conn:
                try:
                    raw = conn.recv(256).decode("utf-8", "replace").strip()
                    command = raw.split()[0] if raw else ""
                    if command not in COMMANDS:
                        reply = {"ok": False, "error": f"unknown command {command!r}"}
                    else:
                        reply = {"ok": True, **(self.handler(command) or {})}
                except Exception as exc:
                    reply = {"ok": False, "error": str(exc)}
                try:
                    conn.sendall(json.dumps(reply).encode())
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if os.name != 'nt':
            self.path.unlink(missing_ok=True)


def send(command: str, *, timeout: float = 2.0) -> dict:
    """Send one command to a running instance. Raises if none is listening."""
    path = socket_path()
    if os.name == 'nt':
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    else:
        if not path.exists():
            raise ConnectionError("Vaani is not running")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    sock.settimeout(timeout)
    try:
        if os.name == 'nt':
            sock.connect(path)
        else:
            sock.connect(str(path))
        sock.sendall(command.encode())
        return json.loads(sock.recv(4096).decode() or "{}")
    finally:
        sock.close()


def is_running() -> bool:
    try:
        send("status", timeout=0.5)
        return True
    except (ConnectionError, OSError, ValueError):
        return False
