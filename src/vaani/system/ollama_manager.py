import urllib.request
import urllib.error
import json
import subprocess
import threading
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from vaani.system.platform import is_windows

@dataclass
class OllamaStatus:
    running: bool
    host: str
    gpu_healthy: bool
    cpu_fallback_running: bool
    cpu_fallback_host: Optional[str]
    models: List[str]
    selected_model: Optional[str]
    error: Optional[str]

class OllamaManager:
    def __init__(self, *, primary_host: str = 'http://127.0.0.1:11434', cpu_fallback_port: int = 11435, model_preference: Optional[List[str]] = None):
        self.primary_host = primary_host
        self.cpu_fallback_port = cpu_fallback_port
        self.cpu_fallback_host = f"http://127.0.0.1:{cpu_fallback_port}"
        self.model_preference = model_preference or []
        self._cpu_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._active_host = primary_host
        self._selected_model: Optional[str] = None

    def list_models(self, host: str) -> List[str]:
        try:
            req = urllib.request.Request(f"{host}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                return [m['name'] for m in data.get('models', [])]
        except Exception:
            return []

    def probe_gpu_inference(self, host: str, model: str) -> bool:
        try:
            payload = json.dumps({
                "model": model,
                "prompt": "Hello",
                "stream": False
            }).encode('utf-8')
            req = urllib.request.Request(
                f"{host}/api/generate",
                data=payload,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status == 200
        except Exception:
            return False

    def probe(self) -> OllamaStatus:
        with self._lock:
            models = self.list_models(self.primary_host)
            running = len(models) > 0
            gpu_healthy = False
            
            if running:
                model_to_test = models[0] if models else None
                if model_to_test:
                    gpu_healthy = self.probe_gpu_inference(self.primary_host, model_to_test)
            
            return OllamaStatus(
                running=running,
                host=self.primary_host,
                gpu_healthy=gpu_healthy,
                cpu_fallback_running=self._cpu_process is not None,
                cpu_fallback_host=self.cpu_fallback_host if self._cpu_process else None,
                models=models,
                selected_model=None,
                error=None if running else "Ollama is not running or unreachable"
            )

    def _find_ollama_exe(self) -> Optional[Path]:
        import shutil
        exe = shutil.which("ollama")
        if exe:
            return Path(exe)
        
        candidates = [
            Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Ollama' / 'ollama.exe',
            Path(os.environ.get('ProgramFiles', '')) / 'Ollama' / 'ollama.exe'
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def start_cpu_fallback(self, *, model: str) -> str:
        with self._lock:
            if self._cpu_process:
                return self.cpu_fallback_host
            
            if not is_windows():
                return self.primary_host # Fallback not supported, just return primary

            exe = self._find_ollama_exe()
            if not exe:
                raise RuntimeError("Could not find ollama.exe for CPU fallback")

            env = os.environ.copy()
            env["OLLAMA_LLM_LIBRARY"] = "cpu_avx2"
            env["OLLAMA_HOST"] = f"127.0.0.1:{self.cpu_fallback_port}"

            self._cpu_process = subprocess.Popen(
                [str(exe), "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if is_windows() else 0
            )

            # Wait for readiness
            for _ in range(10):
                if self.list_models(self.cpu_fallback_host):
                    break
                time.sleep(1)
            
            self._active_host = self.cpu_fallback_host
            return self.cpu_fallback_host

    def ensure_healthy(self, *, model: Optional[str] = None) -> OllamaStatus:
        status = self.probe()
        target_model = model or (status.models[0] if status.models else None)

        if not target_model:
            status.error = "No models available to ensure health."
            return status

        if status.running and status.gpu_healthy:
            self._active_host = self.primary_host
            self._selected_model = target_model
            status.selected_model = target_model
            return status
            
        if is_windows():
            try:
                host = self.start_cpu_fallback(model=target_model)
                status.cpu_fallback_running = True
                status.cpu_fallback_host = host
                status.host = host
                self._selected_model = target_model
                status.selected_model = target_model
                status.error = "Fell back to CPU."
            except Exception as e:
                status.error = f"CPU fallback failed: {e}"
        else:
            status.error = "GPU unhealthy and CPU fallback unsupported on Linux."

        return status

    def get_host(self) -> str:
        return self._active_host

    def get_model(self) -> Optional[str]:
        return self._selected_model

    def stop(self) -> None:
        with self._lock:
            if self._cpu_process:
                try:
                    self._cpu_process.terminate()
                    self._cpu_process.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    self._cpu_process.kill()
                self._cpu_process = None
                self._active_host = self.primary_host

    def __del__(self) -> None:
        if hasattr(self, '_cpu_process') and self._cpu_process:
            try:
                self._cpu_process.kill()
            except Exception:
                pass
