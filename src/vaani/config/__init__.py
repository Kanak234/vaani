"""Vaani application configuration."""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..system.platform import data_dir, config_dir, models_dir


@dataclass
class VaaniConfig:
    """Central application configuration with platform-aware defaults."""

    # Performance
    performance_mode: str = "auto"  # "auto", "low_latency", "balanced", "quality"

    # Models
    stt_model: str = "auto"  # "auto", "tiny", "base", "small", "medium"
    stt_device: str = "auto"  # "auto", "cuda", "cpu"
    llm_model: str = "auto"  # "auto" or specific model name
    tts_device: str = "auto"  # "auto", "cuda", "cpu"

    # Ollama
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_cpu_fallback_port: int = 11435

    # Audio
    input_device: str | None = None  # None = system default
    monitor_device: str | None = None
    virtual_output_device: str | None = None  # VB-CABLE on Windows

    # Voice
    voice_mode: str = "fallback"  # "fallback" or "personal"
    voice_profile_id: str | None = None

    # Privacy
    local_only: bool = True
    persist_transcript: bool = False

    # Paths
    model_dir: str = ""  # Empty = auto-detect

    # Meeting takeover
    takeover_enabled: bool = True
    takeover_silence_s: float = 4.0
    takeover_min_hesitations: int = 2

    def __post_init__(self):
        if not self.model_dir:
            self.model_dir = str(models_dir())

    @classmethod
    def load(cls, path: Path | None = None) -> "VaaniConfig":
        """Load config from JSON file, falling back to defaults."""
        if path is None:
            path = config_dir() / "config.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Filter to only known fields
                known = {f.name for f in cls.__dataclass_fields__.values()}
                filtered = {k: v for k, v in data.items() if k in known}
                return cls(**filtered)
            except (json.JSONDecodeError, TypeError, OSError):
                pass
        return cls()

    def save(self, path: Path | None = None) -> None:
        """Save config to JSON file."""
        if path is None:
            path = config_dir() / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def effective_ollama_host(self, *, cpu_fallback: bool = False) -> str:
        """Return the Ollama host to use."""
        if cpu_fallback:
            return f"http://127.0.0.1:{self.ollama_cpu_fallback_port}"
        return self.ollama_host
