"""Diagnostic checks (PRD §9).

Every check answers one question with a measured result. Nothing here estimates or
reports a nominal value: a diagnostic that lies is worse than no diagnostic, because
the user then trusts a broken path.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import VaaniError


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str                       # pass | fail | skip
    duration_ms: float
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def line(self) -> str:
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[self.status]
        return f"[{mark}] {self.name:<32} {self.duration_ms:7.1f}ms  {self.detail}"


def _run(name: str, fn: Callable[[], tuple[str, str, dict]]) -> CheckResult:
    started = time.perf_counter()
    try:
        status, detail, data = fn()
    except VaaniError as exc:
        return CheckResult(name, "fail", (time.perf_counter() - started) * 1000,
                           exc.user_message(), {"code": exc.code.value})
    except Exception as exc:
        return CheckResult(name, "fail", (time.perf_counter() - started) * 1000,
                           f"{type(exc).__name__}: {exc}")
    return CheckResult(name, status, (time.perf_counter() - started) * 1000, detail, data)


# ------------------------------------------------------------------- checks

def check_audio_backend() -> CheckResult:
    def fn():
        from ..audio.backend.pulse_bindings import available
        if not available():
            return "fail", "libpulse-simple not found", {}
        if not shutil.which("pactl"):
            return "fail", "pactl not found", {}
        server = subprocess.run(["pactl", "info"], capture_output=True, text=True,
                                timeout=5).stdout
        name = next((l.split(":", 1)[1].strip() for l in server.splitlines()
                     if l.startswith("Server Name")), "unknown")
        return "pass", name, {"server": name}
    return _run("Audio backend", fn)


def check_input_devices() -> CheckResult:
    def fn():
        from ..devices.manager import list_sources
        devices = [d for d in list_sources() if not d.is_virtual]
        if not devices:
            return "fail", "no capture devices found", {}
        return "pass", f"{len(devices)} device(s)", {
            "devices": [d.display_name for d in devices]}
    return _run("Input devices", fn)


def check_microphone_capture(device: str | None = None,
                             seconds: float = 1.0) -> CheckResult:
    """Open the mic and measure real signal. Reports level, not just success."""
    def fn():
        from ..audio.backend.pulse_backend import PulseCaptureStream
        stream = PulseCaptureStream(device=device, stream_name="diag-mic")
        try:
            frames = []
            for _ in range(int(seconds * 1000 / stream.frame_ms)):
                frames.append(stream.read_frame())
            latency = stream.latency_ms()
        finally:
            stream.close()
        audio = np.concatenate(frames)
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2)))
        db = 20 * np.log10(max(rms, 1e-9))
        data = {"peak": round(peak, 4), "rms_dbfs": round(db, 1),
                "latency_ms": round(latency, 2)}
        if peak < 1e-4:
            # Not a failure: the room may simply be quiet. But the user needs to
            # know the difference between "mic works" and "mic produced signal".
            return "pass", f"opened, but silent (peak {peak:.5f}) - is it muted?", data
        return "pass", f"peak {peak:.3f}, {db:.1f} dBFS, latency {latency:.1f}ms", data
    return _run("Microphone capture", fn)


def check_virtual_microphone() -> CheckResult:
    """Create the virtual mic, write a tone, and capture it back (AC-02.4).

    Deliberately captures from the device rather than trusting that the write
    succeeded -- see ADR-001 for why that distinction is not academic.
    """
    def fn():
        from ..audio.backend.pulse_backend import PulsePlaybackStream
        from ..devices.manager import VirtualMicrophone
        mic = VirtualMicrophone.create()
        try:
            if not mic.exists():
                return "fail", "created but not visible as a source", {}
            proc = subprocess.Popen(
                ["parec", "--device", mic.node_name, "--format=s16le",
                 "--rate=16000", "--channels=1", "--latency-msec=20"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            time.sleep(0.3)
            sr, n = 16000, 16000 // 50
            t = np.arange(sr, dtype=np.float32) / sr
            tone = (np.sin(2 * np.pi * 1000 * t) * 0.5).astype(np.float32)
            sink = PulsePlaybackStream(device=mic.sink_name, sample_rate=sr,
                                       frame_ms=20, stream_name="diag-vmic",
                                       require_device=True)
            try:
                for i in range(0, tone.size, n):
                    sink.write(tone[i:i + n])
            finally:
                sink.close()
            time.sleep(0.3)
            proc.terminate()
            raw = proc.stdout.read()
            proc.wait(timeout=5)
            got = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            peak = float(np.max(np.abs(got))) if got.size else 0.0
            data = {"peak": round(peak, 4), "samples": int(got.size),
                    "source": mic.node_name, "sink": mic.sink_name}
            if peak < 0.05:
                return "fail", f"virtual mic carried silence (peak {peak:.4f})", data
            return "pass", f"tone recovered at peak {peak:.3f}", data
        finally:
            mic.destroy()
    return _run("Virtual microphone", fn)


def check_network() -> CheckResult:
    def fn():
        try:
            with socket.create_connection(("1.1.1.1", 53), timeout=3):
                return "pass", "reachable", {"online": True}
        except OSError:
            # Offline is a legitimate state for a local-first product.
            return "pass", "offline (local-only providers unaffected)", {"online": False}
    return _run("Network", fn)


def check_vad() -> CheckResult:
    def fn():
        from ..ai.vad.energy import EnergyVad
        vad = EnergyVad()
        sr, fn_ = 16000, 320
        silence = np.zeros(fn_, dtype=np.float32)
        t = np.arange(fn_, dtype=np.float32) / sr
        speech = (np.sin(2 * np.pi * 220 * t) * 0.3).astype(np.float32)
        for _ in range(30):
            vad.is_speech(silence)
        p_sil = vad.is_speech(silence)
        p_sp = max(vad.is_speech(speech) for _ in range(5))
        if p_sp <= p_sil:
            return "fail", f"cannot distinguish speech ({p_sp:.2f}) from silence ({p_sil:.2f})", {}
        return "pass", f"speech {p_sp:.2f} vs silence {p_sil:.2f}", {
            "speech": p_sp, "silence": p_sil}
    return _run("Voice activity detection", fn)


def check_tts() -> CheckResult:
    def fn():
        from ..ai.tts.fallback import FallbackSynthesizer
        tts = FallbackSynthesizer()
        tts.warmup()
        started = time.perf_counter()
        out = tts.synthesize("This is a synthesis test.")
        ms = (time.perf_counter() - started) * 1000
        if out.samples.size == 0:
            return "fail", "produced no audio", {}
        dur = out.samples.size / out.sample_rate
        note = " (fallback voice, not the user's)" if out.is_fallback_voice else ""
        return "pass", f"{dur:.2f}s in {ms:.0f}ms{note}", {
            "duration_s": round(dur, 2), "synth_ms": round(ms, 1),
            "is_fallback": out.is_fallback_voice}
    return _run("Text to speech", fn)


def check_stt() -> CheckResult:
    def fn():
        try:
            import ctranslate2
            import faster_whisper  # noqa: F401
        except ImportError:
            return "skip", "faster-whisper not installed", {}
        n = ctranslate2.get_cuda_device_count()
        # Use the same code path the recogniser uses, so the diagnostic cannot
        # disagree with what a real session will do. cuda_available() preloads
        # the pip CUDA libraries and then verifies by running actual inference --
        # model construction alone succeeds even when cuBLAS is missing.
        from ..ai.stt.cuda_setup import cuda_available
        cuda_usable, cuda_detail = cuda_available()
        device = "cuda" if cuda_usable else "cpu"
        types = sorted(ctranslate2.get_supported_compute_types(device))
        detail = f"will use {device.upper()}"
        if n and not cuda_usable:
            detail += f"; {cuda_detail}"
        return "pass", f"{detail}; compute types: {', '.join(types)}", {
            "device": device, "cuda_devices": n, "cuda_usable": cuda_usable,
            "cuda_detail": cuda_detail, "compute_types": types}

    return _run("Speech recognition runtime", fn)


def check_voice_consent() -> CheckResult:
    def fn():
        from ..voice.consent import ConsentLedger
        ledger = ConsentLedger()
        grant = ledger.active_grant()
        if grant is None:
            # Not a failure: a user who has not enrolled has correctly given no
            # consent. The fallback voice still works.
            return "pass", "no consent on record (personal voice unavailable)", {
                "has_consent": False}
        return "pass", f"granted for {grant.subject_label!r} on {grant.granted_at_utc[:10]}", {
            "has_consent": True, "record_id": grant.id}
    return _run("Voice consent", fn)


def check_voice_profile() -> CheckResult:
    def fn():
        from ..voice.enrollment import VoiceProfileStore
        from pathlib import Path
        store = VoiceProfileStore()
        profile = store.active()
        if profile is None:
            return "pass", "no voice profile enrolled (fallback voice will be used)", {
                "has_profile": False}
        if not Path(profile.reference_audio_path).exists():
            return "fail", "profile exists but its reference audio is missing", {
                "profile_id": profile.id}
        return "pass", (f"{profile.name!r}: {profile.speech_seconds:.0f}s speech, "
                        f"SNR {profile.quality_snr_db:.1f} dB"), {
            "has_profile": True, "profile_id": profile.id}
    return _run("Voice profile", fn)


def check_translation() -> CheckResult:
    """Round-trip a fixed Hindi sentence. Verifies the model actually translates."""
    def fn():
        try:
            import ctranslate2  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return "skip", "translation dependencies not installed", {}
        from pathlib import Path
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        if not (hub.exists() and any(hub.glob("*nllb-200*"))):
            return "skip", "NLLB model not downloaded", {}

        from ..ai.translate.nllb import NllbTranslator
        from ..core.model_budget import plan_placement
        engine = NllbTranslator(device=plan_placement(["translation"]).device_for("translation"))
        engine.warmup()
        started = time.perf_counter()
        result = engine.translate("यह काम कल तक हो जाएगा।",
                                  source_language="hi", target_language="en")
        ms = (time.perf_counter() - started) * 1000
        engine.shutdown()
        if "tomorrow" not in result.text.lower():
            return "fail", f"unexpected translation: {result.text[:60]!r}", {
                "output": result.text}
        return "pass", f"{ms:.0f}ms on {engine.device_used}, output correct", {
            "latency_ms": round(ms, 1), "device": engine.device_used}
    return _run("Translation round-trip", fn)


ALL_CHECKS: list[Callable[[], CheckResult]] = [
    check_audio_backend,
    check_input_devices,
    check_microphone_capture,
    check_virtual_microphone,
    check_vad,
    check_stt,
    check_translation,
    check_tts,
    check_voice_consent,
    check_voice_profile,
    check_network,
]


def run_all() -> list[CheckResult]:
    return [check() for check in ALL_CHECKS]
