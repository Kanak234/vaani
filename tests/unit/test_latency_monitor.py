"""Latency budget warnings (AC-13.3).

The design constraint: warn on a SUSTAINED breach, not a single slow utterance.
Warning on every outlier trains the user to ignore warnings.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from vaani.core.latency_monitor import BUDGET_MS, LatencyMonitor
from vaani.core.types import (
    PerformanceMode,
    StageTiming,
    SuppressionReason,
    Utterance,
    UtteranceResult,
)


def result(latency_ms, *, suppressed=False, stages=None):
    u = Utterance(session_id="s", seq=1, audio=np.zeros(16, dtype=np.float32),
                  sample_rate=16000, speech_end_time=time.monotonic(),
                  speech_ms=1000.0)
    r = UtteranceResult(utterance=u)
    r.total_latency_ms = latency_ms
    r.timings = stages or [StageTiming("stt", 578.0), StageTiming("tts", 1188.0)]
    if suppressed:
        r.suppressed = True
        r.suppression_reason = SuppressionReason.LOW_STT_CONFIDENCE
    return r


@pytest.fixture
def monitor():
    return LatencyMonitor(mode=PerformanceMode.BALANCED, min_samples=5, window=10)


def test_budgets_match_the_nfrs():
    assert BUDGET_MS[PerformanceMode.LOW_LATENCY] == 2500
    assert BUDGET_MS[PerformanceMode.BALANCED] == 4000
    assert BUDGET_MS[PerformanceMode.QUALITY] == 6000


def test_within_budget_never_warns(monitor):
    for _ in range(10):
        assert monitor.record(result(2411)) is None


def test_one_slow_utterance_does_not_warn(monitor):
    """A single outlier is normal. Warning on it would be noise."""
    for _ in range(4):
        monitor.record(result(2000))
    assert monitor.record(result(9000)) is None


def test_sustained_breach_warns(monitor):
    warning = None
    for _ in range(6):
        warning = monitor.record(result(5000)) or warning
    assert warning is not None
    assert warning.p50_ms >= 4000
    assert warning.budget_ms == 4000


def test_warning_names_the_slowest_stage(monitor):
    """'translation is slow' is actionable; 'latency is high' is not."""
    stages = [StageTiming("stt", 300.0), StageTiming("translate", 3800.0)]
    warning = None
    for _ in range(6):
        warning = monitor.record(result(5000, stages=stages)) or warning
    assert warning.worst_stage == "translate"
    assert "translate" in warning.message()


def test_below_min_samples_never_warns(monitor):
    for _ in range(4):
        assert monitor.record(result(9000)) is None


def test_cooldown_prevents_repeat_warnings(monitor):
    warnings = [w for w in (monitor.record(result(9000)) for _ in range(20))
                if w is not None]
    assert len(warnings) == 1


def test_suppressed_utterances_are_excluded(monitor):
    """They produced no speech, so their timing is not what the user experienced."""
    for _ in range(10):
        assert monitor.record(result(9000, suppressed=True)) is None
    assert monitor.percentiles() == {}


def test_percentiles_reported(monitor):
    for ms in (1000, 2000, 3000, 4000, 5000):
        monitor.record(result(ms))
    p = monitor.percentiles()
    assert p["count"] == 5 and p["p50"] == 3000 and p["budget"] == 4000


def test_window_is_bounded():
    m = LatencyMonitor(window=5, min_samples=1)
    for _ in range(20):
        m.record(result(1000))
    assert m.percentiles()["count"] == 5


def test_recovery_after_slow_period(monitor):
    for _ in range(6):
        monitor.record(result(9000))
    for _ in range(10):
        monitor.record(result(1500))
    assert monitor.percentiles()["p50"] < 4000


def test_reset_clears_state(monitor):
    for _ in range(6):
        monitor.record(result(9000))
    monitor.reset()
    assert monitor.percentiles() == {}


def test_quality_mode_tolerates_more(monitor):
    m = LatencyMonitor(mode=PerformanceMode.QUALITY, min_samples=5)
    for _ in range(10):
        assert m.record(result(5000)) is None      # under the 6000 ms budget
