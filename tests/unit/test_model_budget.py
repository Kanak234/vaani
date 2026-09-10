"""Model placement tests.

The scheduler exists to prevent a mid-meeting CUDA OOM, so these tests are mostly
about what it refuses to do.
"""
from __future__ import annotations

from vaani.core.model_budget import (
    DEFAULT_COSTS,
    ModelCost,
    Placement,
    plan_placement,
)

STAGES = ["stt", "translation", "tts"]


def plan(free_mb, **kw):
    """Plan against a fixed free-VRAM figure."""
    import vaani.core.model_budget as mb
    original = mb.free_vram_mb
    mb.free_vram_mb = lambda: free_mb
    try:
        return plan_placement(STAGES, cuda_usable=True, **kw)
    finally:
        mb.free_vram_mb = original


def test_no_gpu_puts_everything_on_cpu():
    p = plan_placement(STAGES, cuda_usable=False)
    assert all(v is Placement.CPU for v in p.placements.values())
    assert p.free_mb is None


def test_plenty_of_vram_puts_everything_on_gpu():
    p = plan(8000)
    assert all(v is Placement.CUDA for v in p.placements.values())


def test_stt_wins_the_gpu_when_memory_is_tight():
    """STT is the biggest latency contributor and has the best GPU speedup."""
    p = plan(1400)
    assert p.placements["stt"] is Placement.CUDA
    assert p.placements["tts"] is Placement.CPU
    assert p.placements["translation"] is Placement.CPU


def test_the_measured_baseline_case():
    """1610 MiB free -- the figure actually measured with a browser open."""
    p = plan(1610)
    assert p.placements["stt"] is Placement.CUDA
    # 1610 - 400 reserve - 900 stt = 310 MiB; neither other model fits.
    assert p.placements["tts"] is Placement.CPU


def test_the_measured_idle_case():
    """3770 MiB free -- measured with nothing else on the GPU."""
    p = plan(3770)
    assert p.placements["stt"] is Placement.CUDA
    assert p.placements["tts"] is Placement.CUDA
    # 3770 - 400 - 900 - 2300 = 170 MiB; NLLB's ~800 MiB does not fit.
    assert p.placements["translation"] is Placement.CPU


def test_never_overcommits_the_card():
    """The sum of GPU placements must always fit in free minus reserve."""
    for free in (500, 1000, 1610, 2500, 3770, 4096, 6000):
        p = plan(free)
        used = sum(DEFAULT_COSTS[s].vram_mb for s, v in p.placements.items()
                   if v is Placement.CUDA)
        assert used <= free - p.reserve_mb, f"overcommitted at {free} MiB"


def test_reserve_is_always_held_back():
    """Filling the card exactly would leave nothing for the CUDA context."""
    p = plan(DEFAULT_COSTS["stt"].vram_mb + 100)   # just over one model
    assert p.placements["stt"] is Placement.CPU


def test_priority_order_is_respected():
    assert (DEFAULT_COSTS["stt"].priority
            < DEFAULT_COSTS["tts"].priority
            < DEFAULT_COSTS["translation"].priority)


def test_notes_explain_every_cpu_demotion():
    """A user asking 'why is this slow?' must get an answer."""
    p = plan(1610)
    demoted = [s for s, v in p.placements.items() if v is Placement.CPU]
    assert demoted
    assert any(s in note for s in demoted for note in p.notes)


def test_unknown_stage_defaults_to_cpu():
    p = plan_placement(["mystery"], cuda_usable=True)
    assert p.placements["mystery"] is Placement.CPU


def test_custom_costs_are_honoured():
    costs = {"tiny": ModelCost("tiny", vram_mb=50, priority=1)}
    import vaani.core.model_budget as mb
    original = mb.free_vram_mb
    mb.free_vram_mb = lambda: 1000
    try:
        p = plan_placement(["tiny"], costs=costs, cuda_usable=True)
    finally:
        mb.free_vram_mb = original
    assert p.placements["tiny"] is Placement.CUDA


# --- resolve_device: the fix for a real CUDA OOM --------------------------

def _resolve(stage, requested, free_mb):
    import vaani.core.model_budget as mb
    orig_free, orig_cuda = mb.free_vram_mb, None
    mb.free_vram_mb = lambda: free_mb
    try:
        from vaani.ai.stt import cuda_setup
        orig_cuda = cuda_setup.cuda_available
        cuda_setup.cuda_available = lambda: (True, "stubbed")
        return mb.resolve_device(stage, requested)
    finally:
        mb.free_vram_mb = orig_free
        if orig_cuda is not None:
            from vaani.ai.stt import cuda_setup as cs
            cs.cuda_available = orig_cuda


def test_auto_falls_back_to_cpu_when_another_app_holds_the_gpu():
    """Regression: ollama held 2734 of 4096 MiB, leaving 1028 MiB free.

    get_cuda_device_count() still returned 1, so every provider chose CUDA and
    every inference died with `CUDA failed with error out of memory`. Choosing on
    device EXISTENCE rather than free memory is the bug.
    """
    assert _resolve("translation", "auto", 1028) == "cpu"
    assert _resolve("tts", "auto", 1028) == "cpu"


def test_auto_uses_the_gpu_when_it_is_actually_free():
    assert _resolve("stt", "auto", 3770) == "cuda"


def test_explicit_device_is_honoured_over_the_budget():
    """If the user names a device, that is their call, not ours."""
    assert _resolve("tts", "cuda", 200) == "cuda"
    assert _resolve("stt", "cpu", 8000) == "cpu"
