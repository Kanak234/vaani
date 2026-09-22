"""Tests for core/supervisor.py.

Every test drives an injected fake clock rather than sleeping, so the sliding
window is exercised deterministically and the suite stays fast. A real-clock
version of these tests would need 60+ seconds of sleeping and would be flaky on
a loaded machine -- precisely the machine this code is meant to survive.
"""
from __future__ import annotations

import pytest

from vaani.core.supervisor import (
    Attempt, CrashWindow, StageSupervisor, looks_like_oom,
)


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --------------------------------------------------------------- CrashWindow

def test_window_trips_on_the_third_failure_inside_the_window():
    clock = FakeClock()
    w = CrashWindow(limit=3, window_s=60.0, clock=clock)
    assert w.record_failure() is False
    clock.advance(1)
    assert w.record_failure() is False
    clock.advance(1)
    # Third failure, 2 s apart: the stage is broken right now.
    assert w.record_failure() is True


def test_failures_spread_wider_than_the_window_never_trip():
    """The whole point of a window: a long healthy session with occasional
    hiccups must not be demoted."""
    clock = FakeClock()
    w = CrashWindow(limit=3, window_s=60.0, clock=clock)
    for _ in range(10):
        assert w.record_failure() is False
        clock.advance(61)       # each failure ages out before the next
    # The loop advances past the window after the last failure too, so by now
    # every one of the ten has aged out. Nothing accumulated.
    assert w.recent_failures == 0


def test_a_success_clears_the_window():
    clock = FakeClock()
    w = CrashWindow(limit=3, window_s=60.0, clock=clock)
    w.record_failure()
    w.record_failure()
    w.record_success()
    # Two more failures must NOT trip, because the success proved the stage works.
    assert w.record_failure() is False
    assert w.record_failure() is False


def test_old_failures_are_evicted_not_accumulated():
    """Guards against the deque growing without bound over a long session."""
    clock = FakeClock()
    w = CrashWindow(limit=100, window_s=10.0, clock=clock)
    for _ in range(500):
        w.record_failure()
        clock.advance(1)
    # Only the last 10 seconds' worth may remain.
    assert w.recent_failures <= 11


def test_invalid_configuration_is_rejected_loudly():
    with pytest.raises(ValueError):
        CrashWindow(limit=0)
    with pytest.raises(ValueError):
        CrashWindow(window_s=0)


# ------------------------------------------------------------------- OOM

@pytest.mark.parametrize("exc", [
    MemoryError("host ram"),
    RuntimeError("CUDA error: out of memory"),
    RuntimeError("CUDA out of memory. Tried to allocate 200.00 MiB"),
])
def test_memory_exhaustion_is_recognised(exc):
    assert looks_like_oom(exc) is True


def test_a_named_cuda_oom_class_is_recognised_without_importing_torch():
    """torch.cuda.OutOfMemoryError is matched by NAME, so this module stays
    importable on a machine with no torch at all."""
    class OutOfMemoryError(Exception):
        pass
    assert looks_like_oom(OutOfMemoryError("boom")) is True


def test_an_ordinary_failure_is_not_mistaken_for_oom():
    assert looks_like_oom(ValueError("bad audio")) is False


# -------------------------------------------------------------- Supervisor

def test_a_successful_call_returns_its_value():
    s = StageSupervisor("pipeline")
    attempt = s.run(lambda a, b: a + b, 2, 3)
    assert attempt.ok is True
    assert attempt.value == 5
    assert s.total_failures == 0


def test_an_exception_is_contained_not_propagated():
    """THE core guarantee: the caller's thread must survive."""
    s = StageSupervisor("pipeline")
    attempt = s.run(lambda: (_ for _ in ()).throw(RuntimeError("stage died")))
    assert attempt.ok is False
    assert isinstance(attempt.error, RuntimeError)
    assert s.total_failures == 1


def test_even_a_baseexception_is_contained():
    """Native extensions can surface things that are not Exception subclasses;
    containment would be worthless if those still killed the worker."""
    class Weird(BaseException):
        pass

    s = StageSupervisor("pipeline")
    attempt = s.run(lambda: (_ for _ in ()).throw(Weird()))
    assert attempt.ok is False
    assert isinstance(attempt.error, Weird)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_user_initiated_shutdown_is_never_swallowed(exc_type):
    """Ctrl-C must still stop Vaani, otherwise the virtual mic is never
    unloaded and the device leaks."""
    s = StageSupervisor("pipeline")
    with pytest.raises(exc_type):
        s.run(lambda: (_ for _ in ()).throw(exc_type()))


def test_three_fast_failures_exhaust_the_supervisor():
    clock = FakeClock()
    s = StageSupervisor("pipeline", limit=3, window_s=60.0, clock=clock)
    boom = lambda: (_ for _ in ()).throw(RuntimeError("x"))  # noqa: E731
    for _ in range(2):
        s.run(boom)
        assert s.exhausted is False
    s.run(boom)
    assert s.exhausted is True


def test_exhaustion_is_latched_so_the_banner_does_not_flicker():
    clock = FakeClock()
    s = StageSupervisor("pipeline", limit=2, window_s=10.0, clock=clock)
    boom = lambda: (_ for _ in ()).throw(RuntimeError("x"))  # noqa: E731
    s.run(boom)
    s.run(boom)
    assert s.exhausted is True
    # Let every failure age out of the window.
    clock.advance(100)
    # The window has self-healed, but the decision to fall back stays made.
    assert s.exhausted is True


def test_reset_gives_a_replacement_provider_a_clean_slate():
    s = StageSupervisor("pipeline", limit=2)
    boom = lambda: (_ for _ in ()).throw(RuntimeError("x"))  # noqa: E731
    s.run(boom)
    s.run(boom)
    assert s.exhausted is True
    s.reset()
    assert s.exhausted is False
    assert s.run(lambda: 1).ok is True


def test_callbacks_fire_once_on_exhaustion():
    failures: list[str] = []
    exhausted: list[str] = []
    s = StageSupervisor("pipeline", limit=2,
                        on_failure=lambda n, e: failures.append(n),
                        on_exhausted=lambda n: exhausted.append(n))
    boom = lambda: (_ for _ in ()).throw(RuntimeError("x"))  # noqa: E731
    for _ in range(4):
        s.run(boom)
    assert failures == ["pipeline"] * 4
    # Exhaustion is announced once, not on every subsequent failure.
    assert exhausted == ["pipeline"]


def test_a_broken_callback_cannot_kill_the_thread_it_is_protecting():
    """If the reporting path could raise, the supervisor would reintroduce the
    exact failure mode it exists to prevent."""
    def bad_callback(_name, _exc):
        raise ValueError("the reporter itself is broken")

    s = StageSupervisor("pipeline", on_failure=bad_callback)
    attempt = s.run(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert attempt.ok is False          # contained, despite the broken callback


def test_oom_is_flagged_on_the_attempt_for_the_caller_to_route():
    s = StageSupervisor("tts")
    attempt = s.run(lambda: (_ for _ in ()).throw(
        RuntimeError("CUDA out of memory")))
    assert attempt.ok is False
    assert attempt.out_of_memory is True


def test_duration_is_reported_for_post_hoc_budget_checks():
    """Hard preemption is impossible in-process (see the module docstring), so
    the supervisor reports duration and the caller enforces the budget."""
    clock = FakeClock()

    def slow():
        clock.advance(9.0)          # 9 seconds inside the "native call"
        return "done"

    s = StageSupervisor("stt", clock=clock)
    attempt = s.run(slow)
    assert attempt.ok is True
    assert attempt.duration_ms == pytest.approx(9000.0)


def test_attempt_is_generic_and_carries_typed_values():
    a: Attempt[int] = Attempt(ok=True, value=7)
    assert a.value == 7
    assert a.out_of_memory is False
