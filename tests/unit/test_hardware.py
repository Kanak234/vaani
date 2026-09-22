"""Tests for vaani.system.hardware — hardware detection."""

from unittest.mock import patch, MagicMock
import pytest

from vaani.system.hardware import (
    detect_cpu,
    detect_gpus,
    detect_cuda,
    detect_ollama,
    full_report,
    GpuInfo,
    CpuInfo,
    HardwareReport,
)


def test_detect_cpu_returns_info():
    info = detect_cpu()
    assert isinstance(info, CpuInfo)
    assert info.model  # non-empty string
    assert info.logical_cores > 0


@patch("subprocess.run")
def test_detect_gpus_handles_missing_nvidia_smi(mock_run):
    mock_run.side_effect = FileNotFoundError
    gpus = detect_gpus()
    assert isinstance(gpus, list)
    assert len(gpus) == 0


def test_detect_cuda_returns_tuple():
    ok, detail = detect_cuda()
    assert isinstance(ok, bool)
    assert detail is None or isinstance(detail, str)


@patch("urllib.request.urlopen")
def test_detect_ollama_handles_connection_refused(mock_urlopen):
    import urllib.error
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
    ok, models = detect_ollama()
    assert ok is False
    assert models == []


def test_full_report_never_crashes():
    report = full_report()
    assert isinstance(report, HardwareReport)
    assert report.os_name  # non-empty
    assert report.ram_total_mb > 0


def test_gpu_info_dataclass():
    gpu = GpuInfo(
        name="Test GPU",
        vram_total_mb=4096,
        vram_free_mb=3000,
        cuda_version="12.0",
        driver_version="535.104",
    )
    assert gpu.name == "Test GPU"
    assert gpu.vram_total_mb == 4096


def test_hardware_report_properties():
    report = full_report()
    # has_nvidia is a property
    assert isinstance(report.has_nvidia, bool)
    # primary_gpu is None or GpuInfo
    if report.gpus:
        assert report.primary_gpu is not None
    else:
        assert report.primary_gpu is None
