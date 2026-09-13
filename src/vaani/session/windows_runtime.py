"""Reliable Windows runtime for the desktop meeting console.

The target Windows machine has a verified CPU Ollama path while the default
CUDA runner is currently crashing. This module keeps the desktop application
usable by running a private, loopback-only CPU Ollama server on a separate port.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

CPU_OLLAMA_HOST = "http://127.0.0.1:11435"
CPU_OLLAMA_PORT = 11435


def _ollama_executable() -> str:
    found = shutil.which("ollama")
    if found:
        return found
    candidate = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
    if os.path.isfile(candidate):
        return candidate
    raise RuntimeError("Ollama executable was not found. Install Ollama before starting Vaani.")


def _tags(host: str) -> list[str]:
    import urllib.request
    with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [str(item.get("name", "")) for item in payload.get("models", [])]


def _has_model(host: str, model: str) -> bool:
    try:
        names = _tags(host)
    except Exception:
        return False
    return any(name == model or name.split(":")[0] == model.split(":")[0] for name in names)


def ensure_cpu_ollama(model: str, *, timeout_s: float = 30.0) -> str:
    """Ensure a loopback-only CPU Ollama endpoint is available for Vaani."""
    if _has_model(CPU_OLLAMA_HOST, model):
        return CPU_OLLAMA_HOST

    exe = _ollama_executable()
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{CPU_OLLAMA_PORT}"
    env["OLLAMA_LLM_LIBRARY"] = "cpu_avx2"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [exe, "serve"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if _has_model(CPU_OLLAMA_HOST, model):
                return CPU_OLLAMA_HOST
        except Exception as exc:
            last_error = exc
        time.sleep(0.25)

    raise RuntimeError(
        f"Could not start the private CPU Ollama endpoint at {CPU_OLLAMA_HOST}; "
        f"model {model!r} is not available there."
    ) from last_error


class WindowsReliableMeetingRuntime:
    """MeetingTakeoverApp configured for the verified Windows CPU path."""

    def __init__(self, *, input_device: str, remote_input_device: str,
                 output_device: str, llm_model: str, voice: str = "fallback",
                 performance_mode=None) -> None:
        from ..core.types import PerformanceMode
        from .meeting_takeover_app import MeetingTakeoverApp

        if os.name != "nt":
            raise RuntimeError("WindowsReliableMeetingRuntime is Windows-only")
        host = ensure_cpu_ollama(llm_model)
        self._app = MeetingTakeoverApp(
            input_device=input_device,
            remote_input_device=remote_input_device,
            output_device=output_device,
            model="small",
            llm_model=llm_model,
            voice=voice,
            performance_mode=performance_mode or PerformanceMode.LOW_LATENCY,
            ollama_host=host,
            stt_device="cpu",
            stt_compute_type="int8",
        )

    def start(self) -> None:
        self._app.start()

    def stop(self) -> None:
        self._app.stop()

    def emergency_stop(self) -> None:
        self._app.stop_event.set()
        if self._app.session is not None:
            self._app.session.emergency_stop()

    @property
    def session(self):
        return self._app.session
