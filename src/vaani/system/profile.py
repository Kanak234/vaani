from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List
from vaani.system.hardware import HardwareReport

class CapabilityLevel(Enum):
    LOW = auto()
    BALANCED = auto()
    HIGH = auto()

@dataclass
class ComponentDecision:
    component: str
    device: str
    model: Optional[str]
    reason: str

@dataclass
class ProfileResult:
    level: CapabilityLevel
    decisions: List[ComponentDecision]
    warnings: List[str]
    gpu_healthy: bool
    ollama_gpu_healthy: bool
    user_override: Optional[CapabilityLevel]

    @property
    def stt_decision(self) -> Optional[ComponentDecision]:
        return next((d for d in self.decisions if d.component == 'stt'), None)
    
    @property
    def llm_decision(self) -> Optional[ComponentDecision]:
        return next((d for d in self.decisions if d.component == 'llm'), None)

    @property
    def tts_decision(self) -> Optional[ComponentDecision]:
        return next((d for d in self.decisions if d.component == 'tts'), None)

def select_ollama_model(available_models: List[str], *, level: CapabilityLevel, ram_mb: int, vram_mb: Optional[int]) -> Optional[str]:
    # Skip coder models as requested
    models = [m for m in available_models if 'coder' not in m.lower()]
    if not models:
        return None

    def find_match(keywords: List[str], size_keywords: List[str] = None) -> Optional[str]:
        for kw in keywords:
            for model in models:
                if kw in model.lower():
                    if size_keywords:
                        if any(sk in model.lower() for sk in size_keywords):
                            return model
                    else:
                        return model
        return None

    # Model preference matrix
    families = ['qwen3', 'deepseek-r1', 'phi4']
    
    if level == CapabilityLevel.HIGH:
        # Prefer 14b, then 8b
        match = find_match(families, ['14b']) or find_match(families, ['8b']) or find_match(families)
        return match or models[0]
    elif level == CapabilityLevel.BALANCED:
        # Prefer 8b, then 7b
        match = find_match(families, ['8b']) or find_match(families, ['7b']) or find_match(families)
        return match or models[0]
    else: # LOW
        # Prefer smallest
        match = find_match(families, ['1.5b', '3b', '7b']) or find_match(families)
        return match or sorted(models, key=len)[0]

def determine_profile(
    report: HardwareReport, 
    *, 
    user_override: Optional[CapabilityLevel] = None, 
    gpu_inference_ok: bool = True, 
    ollama_gpu_ok: bool = True
) -> ProfileResult:
    warnings = []
    decisions = []
    
    ram_gb = report.ram_total_mb / 1024
    gpu = report.primary_gpu
    vram_gb = gpu.vram_total_mb / 1024 if gpu else 0
    has_working_gpu = gpu is not None and gpu_inference_ok
    
    # Determine Level
    level = CapabilityLevel.LOW
    if user_override:
        level = user_override
        warnings.append(f"Using user override profile: {user_override.name}")
    else:
        if ram_gb >= 32 and vram_gb >= 6 and has_working_gpu:
            level = CapabilityLevel.HIGH
        elif ram_gb >= 16 and has_working_gpu:
            level = CapabilityLevel.BALANCED
        else:
            level = CapabilityLevel.LOW

    # STT Decision
    if level == CapabilityLevel.LOW:
        decisions.append(ComponentDecision(
            component='stt', device='cpu', model='small', 
            reason="Low capability level or no GPU. Using small model on CPU with int8 if possible."
        ))
    elif level == CapabilityLevel.BALANCED:
        decisions.append(ComponentDecision(
            component='stt', device='cuda' if has_working_gpu else 'cpu', model='small', 
            reason="Balanced capability level. Using small model on auto device."
        ))
    else:
        decisions.append(ComponentDecision(
            component='stt', device='cuda', model='medium', 
            reason="High capability level. Using medium model on GPU."
        ))

    # TTS Decision
    if level == CapabilityLevel.HIGH and has_working_gpu:
        decisions.append(ComponentDecision(component='tts', device='cuda', model='auto', reason="High capability, using GPU for TTS."))
    else:
        decisions.append(ComponentDecision(component='tts', device='cpu', model='fallback', reason="Conserving VRAM or low resources, using CPU for TTS."))

    # LLM Decision
    vram_mb = gpu.vram_total_mb if gpu else None
    selected_llm = select_ollama_model(report.ollama_models, level=level, ram_mb=report.ram_total_mb, vram_mb=vram_mb)
    if not selected_llm:
        warnings.append("No suitable Ollama model found or Ollama not running.")
        decisions.append(ComponentDecision(component='llm', device='auto', model=None, reason="No compatible Ollama models available."))
    else:
        device_reason = "GPU" if ollama_gpu_ok else "CPU fallback"
        decisions.append(ComponentDecision(component='llm', device=device_reason, model=selected_llm, reason=f"Selected optimal model for {level.name} profile."))

    if not gpu_inference_ok and gpu is not None:
        warnings.append("GPU detected but inference failed (could be health/driver issue).")
    if not ollama_gpu_ok and gpu is not None:
        warnings.append("Ollama GPU inference failed, falling back to CPU.")

    return ProfileResult(
        level=level,
        decisions=decisions,
        warnings=warnings,
        gpu_healthy=has_working_gpu,
        ollama_gpu_healthy=ollama_gpu_ok,
        user_override=user_override
    )
