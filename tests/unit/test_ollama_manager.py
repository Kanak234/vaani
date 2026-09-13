"""Unit tests for vaani.system.ollama_manager — Ollama lifecycle and CPU fallback."""

import json
from unittest.mock import MagicMock, patch
import urllib.error
import pytest

from vaani.system.ollama_manager import OllamaManager, OllamaStatus


def test_ollama_status_dataclass():
    status = OllamaStatus(
        running=True,
        host="http://127.0.0.1:11434",
        gpu_healthy=True,
        cpu_fallback_running=False,
        cpu_fallback_host=None,
        models=["qwen3:8b"],
        selected_model="qwen3:8b",
        error=None,
    )
    assert status.running is True
    assert status.gpu_healthy is True
    assert "qwen3:8b" in status.models


@patch("urllib.request.urlopen")
def test_list_models_returns_model_names(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "models": [{"name": "qwen3:8b"}, {"name": "deepseek-r1:8b"}]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    mgr = OllamaManager()
    models = mgr.list_models("http://127.0.0.1:11434")
    assert models == ["qwen3:8b", "deepseek-r1:8b"]


@patch("urllib.request.urlopen")
def test_probe_handles_offline_ollama(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
    mgr = OllamaManager()
    status = mgr.probe()
    assert status.running is False
    assert status.gpu_healthy is False
    assert status.models == []


@patch("urllib.request.urlopen")
def test_probe_handles_gpu_crash_exit_code(mock_urlopen):
    # Tags succeed, but generate returns 500
    mock_tags = MagicMock()
    mock_tags.read.return_value = json.dumps({
        "models": [{"name": "qwen3:8b"}]
    }).encode("utf-8")

    err = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/generate",
        code=500,
        msg="Internal Server Error: llama-server process has terminated: exit status 0xc0000005",
        hdrs={},
        fp=None,
    )

    def side_effect(req, *args, **kwargs):
        if "api/tags" in req.full_url:
            cm = MagicMock()
            cm.__enter__.return_value = mock_tags
            return cm
        raise err

    mock_urlopen.side_effect = side_effect

    mgr = OllamaManager()
    status = mgr.probe()
    assert status.running is True
    assert status.gpu_healthy is False
    assert status.models == ["qwen3:8b"]


def test_manager_get_host_and_stop():
    mgr = OllamaManager(primary_host="http://127.0.0.1:11434")
    assert mgr.get_host() == "http://127.0.0.1:11434"
    mgr.stop()
    assert mgr.get_host() == "http://127.0.0.1:11434"
