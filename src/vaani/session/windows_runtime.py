"""Reliable Windows runtime for the desktop meeting console.

The target Windows machine has a verified CPU Ollama path while the default
CUDA runner is currently crashing. This module keeps the desktop application
usable by running a private, loopback-only CPU Ollama server on a separate port
instead of depending on the unstable default server.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from typing import Any

from ..ai.assist import AnswerAssistant
from ..ai.stt.whisper import FasterWhisperRecognizer
from ..ai.tts.fallback import FallbackSynthesizer
from ..ai.translate.ollama import OllamaTranslator
from ..ai.vad.energy import EnergyVad
from ..core.types import PerformanceMode
from .meeting_takeover_app import MeetingTakeoverApp

CPU_OLLAMA_HOST = "http://127.0.0.1:11435"
CPU_OLLAMA_PORT = 11435


def _ollama_executable() -> str:
    found = shutil.which("ollama")
    if found:
        return found
    candidate = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
    if os.path.isfile(candidate):
        return candidate
    raise RuntimeError(
        "Ollama executable was not found. Install Ollama before starting Vaani."
    )


def _port_open(host: str = "127.0.0.1", port: int = CPU_OLLAMA_PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def _tags(host: str) -> list[str]:
    import urllib.request

    with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [str(item.get("name", "")) for item in payload.get("models", [])]


def ensure_cpu_ollama(model: str, *, timeout_s: float = 30.0) -> str:
    """Ensure a loopback-only CPU Ollama endpoint is available for Vaani."""
    if _port_open():
        names = _tags(CPU_OLLAMA_HOST)
        if any(name == model or name.split(":")[0] == model.split(":")[0] for name in names):
            return CPU_OLLAMA_HOST

    exe = _ollama_executable()
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{CPU_OLLAMA_PORT}"
    env["OLLAMA_LLM_LIBRARY"] = "cpu_avx2"
    env["OLLAMA_ORIGINS"] = "*"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [exe, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            names = _tags(CPU_OLLAMA_HOST)
            if any(name == model or name.split(":")[0] == model.split(":")[0] for name in names):
                return CPU_OLLAMA_HOST
        except Exception as exc:
            last_error = exc
        time.sleep(0.25)

    raise RuntimeError(
        f"Could not start the private CPU Ollama endpoint at {CPU_OLLAMA_HOST}. "
        f"Model {model!r} must already exist in the configured Ollama model store."
    ) from last_error


class WindowsReliableMeetingRuntime:
    """Windows meeting runtime using verified CPU STT + CPU Ollama fallback."""

    def __init__(self, *, input_device: str, remote_input_device: str,
                 output_device: str, llm_model: str,
                 voice: str = "fallback",
                 performance_mode: PerformanceMode = PerformanceMode.LOW_LATENCY) -> None:
        if os.name != "nt":
            raise RuntimeError("WindowsReliableMeetingRuntime is Windows-only")

        host = ensure_cpu_ollama(llm_model)
        recognizer = FasterWhisperRecognizer(
            model_size="small", device="cpu", compute_type="int8", beam_size=1)
        recognizer.warmup()
        translator = OllamaTranslator(host=host, model=llm_model)
        translator.warmup()

        synthesizer: Any = FallbackSynthesizer()
        profile_id = None
        if voice == "personal":
            from ..ai.tts.xtts import XttsSynthesizer
            from ..voice.consent import ConsentLedger
            from ..voice.enrollment import VoiceProfileStore
            store, ledger = VoiceProfileStore(), ConsentLedger()
            profile = store.active()
            if profile is None:
                raise RuntimeError("No active voice profile is available.")
            if not ledger.has_active_consent():
                raise RuntimeError("Voice consent is not active.")
            synthesizer = XttsSynthesizer(profile_store=store, consent_ledger=ledger)
            synthesizer.warmup()
            profile_id = profile.id

        # Reuse the proven takeover implementation, but replace its heavy
        # components with the verified CPU/loopback components below.
        self._app = MeetingTakeoverApp.__new__(MeetingTakeoverApp)
        self._app.stop_event = __import__("threading").Event()
        self._app.session = None
        self._app._remote = None
        self._app._thread = None
        self._app._monitor_thread = None
        self._app._last_result_count = 0
        self._app.output_device = output_device
        assistant = AnswerAssistant(translator=translator)
        from .takeover import TakeoverConfig
        from .takeover_runtime import MeetingTakeoverRuntime
        runtime = MeetingTakeoverRuntime(
            assistant=assistant, synthesizer=synthesizer,
            config=TakeoverConfig(), voice_profile_id=profile_id)
        runtime.arm()
        from .windows_session import WindowsSessionConfig, WindowsTranslationSession
        self._app.session = WindowsTranslationSession(
            recognizer=recognizer, translator=translator,
            synthesizer=synthesizer, vad=EnergyVad(),
            config=WindowsSessionConfig(
                input_device=input_device,
                virtual_output_device=output_device,
                performance_mode=performance_mode,
                voice_profile_id=profile_id,
                use_virtual_mic=True,
            ))
        self._app.runtime = runtime
        self._app.remote_input_device = remote_input_device

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
