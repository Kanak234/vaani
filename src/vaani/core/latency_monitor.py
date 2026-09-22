"""Rolling latency monitor and budget-breach warning (AC-13.3).

Why this is not just `if latency > budget: warn` :

A single slow utterance is normal — a long sentence, a model reload, the OS
scheduling something else. Warning on it would train the user to ignore warnings,
which is worse than not warning at all. What matters is a *sustained* breach.

So the monitor requires the p50 of a rolling window to exceed the budget, over a
minimum number of samples, and it will not re-warn until the situation has
materially changed. It also names the stage responsible, because "translation is
slow" is actionable and "latency is high" is not.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ..core.types import PerformanceMode, UtteranceResult

#: End-to-end budgets per mode, in milliseconds (NFR-1, NFR-2).
BUDGET_MS: dict[PerformanceMode, float] = {
    PerformanceMode.LOW_LATENCY: 2500.0,
    PerformanceMode.BALANCED: 4000.0,
    PerformanceMode.QUALITY: 6000.0,
}


@dataclass(slots=True)
class LatencyWarning:
    p50_ms: float
    budget_ms: float
    worst_stage: str | None
    worst_stage_ms: float
    sample_count: int

    def message(self) -> str:
        over = self.p50_ms - self.budget_ms
        base = (f"Translation is running {over:.0f} ms over budget "
                f"(p50 {self.p50_ms:.0f} ms vs {self.budget_ms:.0f} ms)")
        if self.worst_stage:
            base += f"; slowest stage is {self.worst_stage} at {self.worst_stage_ms:.0f} ms"
        return base + "."


@dataclass(slots=True)
class LatencyMonitor:
    mode: PerformanceMode = PerformanceMode.BALANCED
    window: int = 10
    #: Do not judge from one or two utterances.
    min_samples: int = 5
    #: Do not repeat the same warning constantly.
    cooldown_s: float = 60.0

    _samples: deque = field(init=False, repr=False)
    _stage_samples: deque = field(init=False, repr=False)
    _last_warned: float = field(default=0.0, init=False)
    _budget: float = field(init=False)

    def __post_init__(self) -> None:
        self._samples = deque(maxlen=max(1, self.window))
        self._stage_samples = deque(maxlen=max(1, self.window))
        self._budget = BUDGET_MS[self.mode]

    @property
    def budget_ms(self) -> float:
        return self._budget

    def record(self, result: UtteranceResult) -> LatencyWarning | None:
        """Record one utterance; return a warning only on a sustained breach.

        Suppressed utterances are excluded: they did not produce speech, so their
        timing does not describe what the user experienced.
        """
        if result.suppressed or result.total_latency_ms is None:
            return None

        self._samples.append(result.total_latency_ms)
        stage_timing = {}
        for timing in result.timings:
            if timing.succeeded:
                stage_timing[timing.stage] = timing.duration_ms
        self._stage_samples.append(stage_timing)

        if len(self._samples) < self.min_samples:
            return None

        ordered = sorted(self._samples)
        p50 = ordered[len(ordered) // 2]
        if p50 <= self._budget:
            return None

        now = time.monotonic()
        if now - self._last_warned < self.cooldown_s:
            return None
        self._last_warned = now

        stage, stage_ms = self.slowest_stage()
        return LatencyWarning(p50_ms=p50, budget_ms=self._budget,
                              worst_stage=stage, worst_stage_ms=stage_ms,
                              sample_count=len(self._samples))

    def slowest_stage(self) -> tuple[str | None, float]:
        """Mean duration of the slowest stage — what to actually fix."""
        if not self._stage_samples:
            return None, 0.0

        totals = {}
        counts = {}
        for sample in self._stage_samples:
            for stage, duration in sample.items():
                totals[stage] = totals.get(stage, 0.0) + duration
                counts[stage] = counts.get(stage, 0) + 1

        if not totals:
            return None, 0.0

        stage, total = max(
            totals.items(),
            key=lambda kv: kv[1] / max(1, counts[kv[0]]))
        return stage, total / max(1, counts[stage])

    def percentiles(self) -> dict[str, float]:
        if not self._samples:
            return {}
        ordered = sorted(self._samples)
        def pct(p: float) -> float:
            return round(ordered[min(len(ordered)-1,
                                     int(round((len(ordered)-1) * p)))], 1)
        return {"count": len(ordered), "p50": pct(0.5), "p95": pct(0.95),
                "min": round(ordered[0], 1), "max": round(ordered[-1], 1),
                "budget": self._budget}

    def reset(self) -> None:
        self._samples.clear()
        self._stage_samples.clear()
        self._last_warned = 0.0
