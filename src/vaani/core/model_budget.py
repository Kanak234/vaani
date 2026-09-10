"""Model placement under a hard memory ceiling.

The problem this exists to solve is specific to this machine and to any laptop
GPU. Three models now want to be resident at once:

    Whisper medium (int8_float16)   ~0.8 GB VRAM
    NLLB-200 600M  (int8)           ~0.7 GB VRAM
    XTTS-v2        (fp32 + torch)   ~2.2 GB VRAM

That is ~3.7 GB against a 4 GB card whose free space was measured at 1610 MiB with
a browser open and 3770 MiB with nothing else running. Loading all three onto the
GPU will succeed on an idle machine and then OOM the moment the user opens the
meeting client -- i.e. exactly when the product is used.

So placement is decided here, once, against *measured* free memory rather than the
card's nominal size, and the policy is explicit about what it sacrifices.

Priority order, and why:
  1. STT stays on the GPU. It is the largest measured latency contributor
     (578 ms of 1103 ms) and the one with the best GPU speedup (10x).
  2. TTS next. It is the second most expensive stage and the one the user hears.
  3. Translation last. NLLB at 600M is small and its CPU penalty is the least
     damaging of the three.

This is a scheduler, not an optimiser. It refuses to over-commit rather than
trying to be clever, because a mid-meeting OOM is a much worse outcome than one
stage running slower.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum


class Placement(Enum):
    CUDA = "cuda"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class ModelCost:
    """Approximate VRAM cost. These are ESTIMATES, and labelled as such.

    They are used only to decide placement, never reported as measurements. The
    real figures come from `nvidia-smi` after loading, via `observed_vram_mb()`.
    """

    key: str
    vram_mb: int
    priority: int          # lower = keeps the GPU longer


#: Estimated costs. Deliberately conservative -- under-estimating causes the OOM
#: this module exists to prevent.
DEFAULT_COSTS: dict[str, ModelCost] = {
    "stt": ModelCost("stt", vram_mb=900, priority=1),
    "tts": ModelCost("tts", vram_mb=2300, priority=2),
    "translation": ModelCost("translation", vram_mb=800, priority=3),
}

#: Never fill the card. Drivers, the compositor and CUDA's own context all take
#: memory that does not appear in a model's weight count.
RESERVE_MB = 400


def free_vram_mb() -> int | None:
    """Free VRAM as the driver reports it right now. None if unavailable.

    Deliberately queries live rather than caching: the whole point is that free
    memory changes when the user opens a browser or a meeting client.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def observed_vram_mb() -> int | None:
    """VRAM currently in use. For reporting real numbers after loading."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


@dataclass(slots=True)
class BudgetPlan:
    placements: dict[str, Placement]
    free_mb: int | None
    reserve_mb: int
    notes: list[str] = field(default_factory=list)

    def device_for(self, stage: str) -> str:
        return self.placements.get(stage, Placement.CPU).value

    def summary(self) -> str:
        free = f"{self.free_mb} MiB free" if self.free_mb is not None else "no GPU"
        assigned = ", ".join(f"{k}={v.value}" for k, v in sorted(self.placements.items()))
        return f"{free} (reserve {self.reserve_mb} MiB) -> {assigned}"


def plan_placement(stages: list[str] | None = None,
                   costs: dict[str, ModelCost] | None = None,
                   reserve_mb: int = RESERVE_MB,
                   cuda_usable: bool | None = None) -> BudgetPlan:
    """Decide where each stage's model runs, against measured free VRAM."""
    costs = costs or DEFAULT_COSTS
    stages = stages or list(costs)
    notes: list[str] = []

    if cuda_usable is None:
        try:
            from ..ai.stt.cuda_setup import cuda_available
            cuda_usable, detail = cuda_available()
            if not cuda_usable:
                notes.append(f"GPU not usable: {detail}")
        except Exception as exc:
            cuda_usable = False
            notes.append(f"GPU check failed: {exc}")

    if not cuda_usable:
        notes.append("all models placed on CPU")
        return BudgetPlan({s: Placement.CPU for s in stages}, None, reserve_mb, notes)

    free = free_vram_mb()
    if free is None:
        notes.append("could not read free VRAM; placing only STT on GPU")
        return BudgetPlan(
            {s: (Placement.CUDA if s == "stt" else Placement.CPU) for s in stages},
            None, reserve_mb, notes)

    budget = max(0, free - reserve_mb)
    placements: dict[str, Placement] = {}
    for stage in sorted(stages, key=lambda s: costs.get(s, DEFAULT_COSTS["stt"]).priority):
        cost = costs.get(stage)
        if cost is None:
            placements[stage] = Placement.CPU
            continue
        if cost.vram_mb <= budget:
            placements[stage] = Placement.CUDA
            budget -= cost.vram_mb
        else:
            placements[stage] = Placement.CPU
            notes.append(
                f"{stage} -> CPU: needs ~{cost.vram_mb} MiB, only {budget} MiB left "
                "after higher-priority models")

    if all(p is Placement.CUDA for p in placements.values()):
        notes.append(f"all models fit on GPU; {budget} MiB spare")
    return BudgetPlan(placements, free, reserve_mb, notes)


def resolve_device(stage: str, requested: str = "auto") -> str:
    """Resolve a provider's `device` setting against LIVE free VRAM.

    This is the function that makes the scheduler real rather than decorative.
    A provider that resolves "auto" by asking only whether a CUDA device EXISTS
    will happily load onto a card another application has already filled, and
    then die with `CUDA failed with error out of memory` on the first inference.

    That is not hypothetical: it was hit during development when ollama held
    2734 MiB of a 4096 MiB card, leaving 1028 MiB free. `get_cuda_device_count()`
    still returned 1, so every provider chose CUDA and every translation failed.

    An explicit "cuda" or "cpu" is honoured as given -- the user asked for it.
    """
    if requested != "auto":
        return requested
    return plan_placement([stage]).device_for(stage)
