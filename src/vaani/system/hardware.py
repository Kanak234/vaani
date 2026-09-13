import os
import subprocess
import platform
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime, timezone
import importlib.util

@dataclass
class GpuInfo:
    name: str
    vram_total_mb: int
    vram_free_mb: int
    cuda_version: Optional[str]
    driver_version: Optional[str]

@dataclass
class CpuInfo:
    vendor: str
    model: str
    logical_cores: int
    physical_cores: Optional[int]
    features: List[str]

@dataclass
class AudioDeviceInfo:
    name: str
    id: str
    kind: str
    is_virtual: bool
    is_loopback: bool

@dataclass
class HardwareReport:
    os_name: str
    os_version: str
    cpu: CpuInfo
    ram_total_mb: int
    ram_available_mb: int
    gpus: List[GpuInfo]
    cuda_available: bool
    ctranslate2_gpu: bool
    ollama_available: bool
    ollama_models: List[str]
    audio_devices: List[AudioDeviceInfo]
    timestamp: str

    @property
    def has_nvidia(self) -> bool:
        return len(self.gpus) > 0

    @property
    def primary_gpu(self) -> Optional[GpuInfo]:
        return self.gpus[0] if self.gpus else None

    @property
    def has_virtual_cable(self) -> bool:
        return any(d.is_virtual for d in self.audio_devices)

def _run_cmd(cmd: List[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return result.stdout.strip()
    except Exception:
        return ""

def detect_cpu() -> CpuInfo:
    vendor, model = "Unknown", "Unknown"
    features = []
    try:
        import psutil
        logical = psutil.cpu_count(logical=True) or 1
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        logical, physical = os.cpu_count() or 1, None

    if platform.system() == "Linux":
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith('vendor_id') and vendor == "Unknown":
                        vendor = line.split(':')[1].strip()
                    elif line.startswith('model name') and model == "Unknown":
                        model = line.split(':')[1].strip()
                    elif line.startswith('flags') and not features:
                        flags = line.split(':')[1].strip().split()
                        if 'avx2' in flags: features.append('AVX2')
                        if 'avx512f' in flags: features.append('AVX512')
        except Exception:
            pass
    elif platform.system() == "Windows":
        model = platform.processor()
        try:
            # Approximation for Windows without external deps
            vendor = _run_cmd(["wmic", "cpu", "get", "manufacturer"]).split('\n')[1].strip()
        except Exception:
            pass
        # Basic check for features
        if 'avx2' in _run_cmd(["systeminfo"]).lower():
            features.append('AVX2')

    return CpuInfo(vendor=vendor, model=model, logical_cores=logical, physical_cores=physical, features=features)

def detect_gpus() -> List[GpuInfo]:
    gpus = []
    smi_paths = ["nvidia-smi"]
    if platform.system() == "Windows":
        smi_paths.extend([
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        ])
    
    for smi in smi_paths:
        try:
            out = _run_cmd([smi, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"])
            if out:
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpus.append(GpuInfo(
                            name=parts[0],
                            vram_total_mb=int(parts[1]) if parts[1].isdigit() else 0,
                            vram_free_mb=int(parts[2]) if parts[2].isdigit() else 0,
                            cuda_version=None, # Needs full smi output to parse accurately, omitted for simplicity
                            driver_version=parts[3]
                        ))
                break # Success
        except Exception:
            continue
    return gpus

def detect_cuda() -> Tuple[bool, Optional[str]]:
    try:
        if importlib.util.find_spec("torch"):
            import torch
            if torch.cuda.is_available():
                return True, torch.version.cuda
    except Exception:
        pass
    
    # Try CTranslate2 fallback check
    if detect_ctranslate2_gpu():
        return True, None
    return False, None

def detect_ctranslate2_gpu() -> bool:
    try:
        if importlib.util.find_spec("ctranslate2"):
            import ctranslate2
            types = ctranslate2.get_supported_compute_types("cuda")
            return bool(types)
    except Exception:
        pass
    return False

def detect_ollama(host: str = 'http://127.0.0.1:11434') -> Tuple[bool, List[str]]:
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                models = [m['name'] for m in data.get('models', [])]
                return True, models
    except Exception:
        pass
    return False, []

def detect_audio_devices() -> List[AudioDeviceInfo]:
    devices = []
    try:
        if platform.system() == "Windows":
            if importlib.util.find_spec("soundcard"):
                import soundcard as sc
                for mic in sc.all_microphones(include_loopback=True):
                    is_virtual = any(kw in mic.name for kw in ['CABLE', 'VB-Audio', 'Virtual'])
                    devices.append(AudioDeviceInfo(name=mic.name, id=str(mic.name), kind="input", is_virtual=is_virtual, is_loopback=mic.isloopback))
                for spk in sc.all_speakers():
                    is_virtual = any(kw in spk.name for kw in ['CABLE', 'VB-Audio', 'Virtual'])
                    devices.append(AudioDeviceInfo(name=spk.name, id=str(spk.name), kind="output", is_virtual=is_virtual, is_loopback=False))
        elif platform.system() == "Linux":
            # Simplified pactl parser
            out = _run_cmd(["pactl", "list", "short", "sources"])
            for line in out.splitlines():
                parts = line.split('\t')
                if len(parts) >= 2:
                    is_virtual = 'virtual' in parts[1].lower()
                    devices.append(AudioDeviceInfo(name=parts[1], id=parts[0], kind="input", is_virtual=is_virtual, is_loopback='monitor' in parts[1].lower()))
    except Exception:
        pass
    return devices

def get_ram_info() -> Tuple[int, int]:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total // (1024 * 1024), mem.available // (1024 * 1024)
    except ImportError:
        pass
    
    if platform.system() == "Linux":
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split(':')
                    meminfo[parts[0]] = int(parts[1].split()[0])
                total = meminfo.get('MemTotal', 0) // 1024
                available = meminfo.get('MemAvailable', meminfo.get('MemFree', 0)) // 1024
                return total, available
        except Exception:
            pass
            
    return 0, 0

def full_report(ollama_host: str = 'http://127.0.0.1:11434') -> HardwareReport:
    os_name = platform.system()
    os_version = platform.version()
    cpu = detect_cpu()
    ram_total, ram_avail = get_ram_info()
    gpus = detect_gpus()
    cuda_avail, _ = detect_cuda()
    ct2_gpu = detect_ctranslate2_gpu()
    ollama_avail, ollama_models = detect_ollama(ollama_host)
    audio_devices = detect_audio_devices()
    
    return HardwareReport(
        os_name=os_name,
        os_version=os_version,
        cpu=cpu,
        ram_total_mb=ram_total,
        ram_available_mb=ram_avail,
        gpus=gpus,
        cuda_available=cuda_avail,
        ctranslate2_gpu=ct2_gpu,
        ollama_available=ollama_avail,
        ollama_models=ollama_models,
        audio_devices=audio_devices,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
