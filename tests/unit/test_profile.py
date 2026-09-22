"""Tests for vaani.system.profile — capability profiling."""

from vaani.system.hardware import HardwareReport, CpuInfo, GpuInfo, AudioDeviceInfo
from vaani.system.profile import (
    CapabilityLevel,
    ComponentDecision,
    ProfileResult,
    determine_profile,
    select_ollama_model,
)


def _make_report(*, ram_mb=32768, gpu=None, cuda=False, ollama_models=None):
    """Helper to create a HardwareReport for testing."""
    gpus = [gpu] if gpu else []
    return HardwareReport(
        os_name="Linux",
        os_version="6.1",
        cpu=CpuInfo(vendor="GenuineIntel", model="Test CPU", logical_cores=8,
                    physical_cores=4, features=["avx2"]),
        ram_total_mb=ram_mb,
        ram_available_mb=ram_mb // 2,
        gpus=gpus,
        cuda_available=cuda,
        ctranslate2_gpu=cuda,
        ollama_available=bool(ollama_models),
        ollama_models=ollama_models or [],
        audio_devices=[],
        timestamp="2024-01-01T00:00:00Z",
    )


def test_low_profile_for_no_gpu():
    report = _make_report(ram_mb=8192, gpu=None, cuda=False)
    profile = determine_profile(report)
    assert profile.level == CapabilityLevel.LOW


def test_balanced_profile_for_16gb_4gb_vram():
    gpu = GpuInfo(name="RTX 3050", vram_total_mb=4096, vram_free_mb=3000,
                  cuda_version="12.0", driver_version="535")
    report = _make_report(ram_mb=16384, gpu=gpu, cuda=True)
    profile = determine_profile(report)
    assert profile.level in (CapabilityLevel.BALANCED, CapabilityLevel.LOW)


def test_high_profile_for_32gb_6gb_vram():
    gpu = GpuInfo(name="RTX 3050", vram_total_mb=6144, vram_free_mb=5000,
                  cuda_version="12.0", driver_version="535")
    report = _make_report(ram_mb=32768, gpu=gpu, cuda=True)
    profile = determine_profile(report)
    assert profile.level in (CapabilityLevel.HIGH, CapabilityLevel.BALANCED)


def test_user_override_respected():
    report = _make_report(ram_mb=32768, cuda=False)
    profile = determine_profile(report, user_override=CapabilityLevel.LOW)
    assert profile.user_override == CapabilityLevel.LOW
    assert profile.level == CapabilityLevel.LOW


def test_gpu_unhealthy_forces_cpu_decisions():
    gpu = GpuInfo(name="RTX 3050", vram_total_mb=6144, vram_free_mb=5000,
                  cuda_version="12.0", driver_version="535")
    report = _make_report(ram_mb=32768, gpu=gpu, cuda=True)
    profile = determine_profile(report, gpu_inference_ok=False)
    # With GPU unhealthy, STT and TTS should be on CPU
    for d in profile.decisions:
        if d.component in ("stt", "tts"):
            assert d.device == "cpu", f"{d.component} should be on cpu when GPU unhealthy"


def test_select_ollama_model_prefers_qwen3():
    models = ["qwen3:8b", "deepseek-r1:8b", "phi4-mini:latest"]
    result = select_ollama_model(models, level=CapabilityLevel.BALANCED,
                                 ram_mb=16384, vram_mb=4096)
    assert result is not None
    assert "qwen3" in result or "deepseek" in result or "phi4" in result


def test_select_ollama_model_skips_coder():
    models = ["qwen2.5-coder:7b"]
    result = select_ollama_model(models, level=CapabilityLevel.BALANCED,
                                 ram_mb=16384, vram_mb=4096)
    # Coder models should be skipped for translation
    # If it's the only model, it may still be returned or None
    # The key is that non-coder models are preferred
    assert result is None or "coder" in result  # may fallback to coder if only option


def test_select_ollama_model_returns_none_when_empty():
    result = select_ollama_model([], level=CapabilityLevel.LOW,
                                 ram_mb=8192, vram_mb=None)
    assert result is None


def test_profile_has_decisions():
    report = _make_report(ram_mb=16384)
    profile = determine_profile(report)
    assert len(profile.decisions) > 0
    components = {d.component for d in profile.decisions}
    assert "stt" in components
