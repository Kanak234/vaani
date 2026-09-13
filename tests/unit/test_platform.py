"""Tests for vaani.system.platform — cross-platform path resolution."""

import os
from pathlib import Path
from unittest.mock import patch

from vaani.system.platform import (
    data_dir,
    config_dir,
    runtime_dir,
    models_dir,
    log_dir,
    cache_dir,
    is_windows,
    is_linux,
)


def test_data_dir_returns_path(tmp_path):
    with patch.dict(os.environ, {"XDG_DATA_HOME": str(tmp_path)}):
        result = data_dir()
        assert isinstance(result, Path)
        assert result.exists()


def test_config_dir_returns_path(tmp_path):
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
        result = config_dir()
        assert isinstance(result, Path)
        assert result.exists()


def test_runtime_dir_returns_path(tmp_path):
    with patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(tmp_path)}):
        result = runtime_dir()
        assert isinstance(result, Path)
        assert result.exists()


def test_log_dir_returns_path(tmp_path):
    with patch.dict(os.environ, {"XDG_DATA_HOME": str(tmp_path)}):
        result = log_dir()
        assert isinstance(result, Path)
        assert result.exists()


def test_cache_dir_returns_path(tmp_path):
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
        result = cache_dir()
        assert isinstance(result, Path)
        assert result.exists()


def test_models_dir_returns_path(tmp_path):
    with patch.dict(os.environ, {"VAANI_MODELS_DIR": str(tmp_path / "models")}):
        result = models_dir()
        assert isinstance(result, Path)
        assert result.exists()


def test_models_dir_respects_env_var(tmp_path):
    custom = tmp_path / "my_models"
    with patch.dict(os.environ, {"VAANI_MODELS_DIR": str(custom)}):
        result = models_dir()
        assert result == custom
        assert result.exists()


def test_is_windows_or_linux():
    w = is_windows()
    l = is_linux()
    assert isinstance(w, bool)
    assert isinstance(l, bool)
    if os.name != "nt":
        assert l is True
        assert w is False


def test_dirs_are_created(tmp_path):
    with patch.dict(os.environ, {"XDG_DATA_HOME": str(tmp_path)}):
        d = data_dir()
        assert d.is_dir()
