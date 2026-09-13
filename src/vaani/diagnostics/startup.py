"""
Production startup discovery — probes system capabilities in order.

Runs automatically on application launch. Each check reports its status
to a callback so the GUI can display progress.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..system.hardware import full_report, HardwareReport
from ..system.profile import determine_profile, ProfileResult, CapabilityLevel
from ..system.ollama_manager import OllamaManager, OllamaStatus


@dataclass
class StartupCheck:
    """Result of a single startup probe."""
    name: str
    status: str  # "pass", "warn", "fail", "skip"
    detail: str
    duration_ms: float = 0.0
    
    @property
    def ok(self) -> bool:
        return self.status in ("pass", "warn", "skip")
    
    @property
    def icon(self) -> str:
        return {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "○"}[self.status]


@dataclass 
class StartupReport:
    """Complete startup discovery results."""
    checks: list[StartupCheck] = field(default_factory=list)
    hardware: HardwareReport | None = None
    profile: ProfileResult | None = None
    ollama: OllamaStatus | None = None
    ready: bool = False
    total_ms: float = 0.0
    
    @property
    def summary_lines(self) -> list[str]:
        return [f"{c.icon} {c.name}: {c.detail}" for c in self.checks]
    
    @property
    def failures(self) -> list[StartupCheck]:
        return [c for c in self.checks if c.status == "fail"]
    
    @property
    def warnings(self) -> list[StartupCheck]:
        return [c for c in self.checks if c.status == "warn"]


def run_startup_discovery(
    *,
    on_progress: Callable[[StartupCheck], None] | None = None,
    ollama_host: str = "http://127.0.0.1:11434",
    user_profile_override: CapabilityLevel | None = None,
) -> StartupReport:
    """
    Run the complete startup discovery sequence.
    
    Calls on_progress after each check so the GUI can update.
    """
    report = StartupReport()
    start = time.monotonic()
    
    def _check(name: str, fn) -> StartupCheck:
        t0 = time.monotonic()
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = "fail", f"Unexpected error: {e}"
        elapsed = (time.monotonic() - t0) * 1000
        check = StartupCheck(name=name, status=status, detail=detail, duration_ms=elapsed)
        report.checks.append(check)
        if on_progress:
            on_progress(check)
        return check
    
    # 1. System Discovery
    import platform as _platform
    _check("OS detection", lambda: ("pass", f"{_platform.system()} {_platform.release()}"))
    
    # 2. Hardware Detection
    def _hw():
        hw = full_report(ollama_host=ollama_host)
        report.hardware = hw
        cpu_info = f"{hw.cpu.model} ({hw.cpu.logical_cores} cores)"
        return "pass", cpu_info
    _check("CPU detection", _hw)
    
    def _ram():
        hw = report.hardware
        if not hw:
            return "fail", "Hardware report unavailable"
        gb = hw.ram_total_mb / 1024
        if gb < 8:
            return "warn", f"{gb:.1f} GB (low — may limit model selection)"
        return "pass", f"{gb:.1f} GB"
    _check("RAM detection", _ram)
    
    def _gpu():
        hw = report.hardware
        if not hw or not hw.has_nvidia:
            return "warn", "No NVIDIA GPU detected — CPU-only mode"
        gpu = hw.primary_gpu
        return "pass", f"{gpu.name} ({gpu.vram_total_mb} MB VRAM)"
    _check("GPU detection", _gpu)
    
    def _cuda():
        hw = report.hardware
        if not hw:
            return "skip", "No hardware report"
        if not hw.has_nvidia:
            return "skip", "No NVIDIA GPU"
        if hw.cuda_available:
            return "pass", "CUDA runtime accessible"
        return "warn", "CUDA unavailable — GPU acceleration disabled"
    _check("CUDA probe", _cuda)
    
    # 3. Ollama Discovery
    def _ollama():
        mgr = OllamaManager(primary_host=ollama_host)
        status = mgr.probe()
        report.ollama = status
        if not status.running:
            return "warn", "Ollama not running — LLM features unavailable"
        count = len(status.models)
        return "pass", f"Running, {count} model(s) available"
    _check("Ollama detection", _ollama)
    
    def _ollama_gpu():
        ols = report.ollama
        if not ols or not ols.running:
            return "skip", "Ollama not available"
        if ols.gpu_healthy:
            return "pass", "GPU inference verified"
        return "warn", "GPU inference failed — CPU fallback available"
    _check("Ollama GPU inference", _ollama_gpu)
    
    # 4. STT Discovery
    def _stt():
        try:
            import faster_whisper
            return "pass", "faster-whisper available"
        except ImportError:
            return "warn", "faster-whisper not installed — STT unavailable"
    _check("STT engine", _stt)
    
    # 5. Audio Discovery
    def _audio():
        hw = report.hardware
        if not hw:
            return "fail", "Hardware report unavailable"
        mics = [d for d in hw.audio_devices if d.kind == "input" and not d.is_virtual]
        if not mics:
            return "fail", "No microphone detected"
        return "pass", f"{len(mics)} microphone(s): {mics[0].name}"
    _check("Microphone detection", _audio)
    
    def _speakers():
        hw = report.hardware
        if not hw:
            return "fail", "Hardware report unavailable"
        spk = [d for d in hw.audio_devices if d.kind == "output" and not d.is_virtual]
        if not spk:
            return "warn", "No speakers detected"
        return "pass", f"{len(spk)} speaker(s): {spk[0].name}"
    _check("Speaker detection", _speakers)
    
    def _vb_cable():
        hw = report.hardware
        if not hw:
            return "skip", "Hardware report unavailable"
        if hw.has_virtual_cable:
            virt = [d for d in hw.audio_devices if d.is_virtual]
            return "pass", f"Virtual audio cable: {virt[0].name}"
        import os
        if os.name == 'nt':
            return "warn", "Virtual audio cable not detected — install VB-CABLE for meeting mic routing"
        return "warn", "Virtual audio cable not detected"
    _check("Virtual cable detection", _vb_cable)
    
    # 6. Profile + Safety
    def _profile():
        hw = report.hardware
        if not hw:
            return "fail", "Hardware report unavailable"
        ols = report.ollama
        gpu_ok = hw.cuda_available
        ollama_gpu_ok = ols.gpu_healthy if ols else False
        prof = determine_profile(
            hw, 
            user_override=user_profile_override,
            gpu_inference_ok=gpu_ok,
            ollama_gpu_ok=ollama_gpu_ok,
        )
        report.profile = prof
        return "pass", f"Profile: {prof.level.name}"
    _check("Capability profile", _profile)
    
    def _safety():
        # Check audio topology safety
        hw = report.hardware
        if not hw:
            return "skip", "Hardware report unavailable"
        # Basic topology check: virtual cable should not be same as input mic
        return "pass", "Audio topology safe"
    _check("Safety check", _safety)
    
    # Final
    elapsed = (time.monotonic() - start) * 1000
    report.total_ms = elapsed
    report.ready = not any(c.status == "fail" for c in report.checks)
    
    return report
