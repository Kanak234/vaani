"""Crash containment for long-running sessions.

WHY THIS FILE EXISTS
--------------------
Phase 1 asks for a session that survives 30+ minutes and recovers gracefully
when a stage dies. Before this module, the worker thread in
``session/engine.py`` ran the pipeline and then did five more things --
database write, audio injection, latency recording, result callback, state
transition -- with only *some* of them wrapped in try/except. Anything raising
from an unwrapped line killed the worker thread outright. Nothing restarted it,
and nothing told the user: capture kept running, the virtual mic kept emitting
silence, and the session was deaf for the rest of the meeting.

The repair is not "wrap everything in try/except". That converts a loud crash
into a silent one, which is worse in a meeting: the user keeps talking and
nothing comes out. What is needed is a component that answers three questions:

  1. Did this attempt fail, and was it the kind of failure worth retrying?
  2. Has it now failed so often, so quickly, that retrying is pointless?
  3. When we give up, what do we fall back to?

WHERE THE DATA FLOWS NEXT
-------------------------
``session/engine.py`` wraps each worker iteration in :meth:`StageSupervisor.run`.
On a clean failure the utterance is dropped and the loop continues. When
:meth:`StageSupervisor.exhausted` flips to True, the engine surfaces it through
the same ``on_latency_warning``-style callback the overlay already listens to,
so the user sees a degraded-mode banner rather than silence.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not preempt a stage that has hung. Whisper and XTTS spend their time
inside native C/CUDA calls, and CPython cannot interrupt another thread that is
not executing bytecode -- there is no portable in-process equivalent of a hard
timeout. Claiming otherwise would be a lie in the docs and a hang in practice.
What we *can* do honestly is measure after the fact: :meth:`run` reports how
long the call took, the caller compares it with the stage budget, and a stage
that repeatedly overruns is treated as failing and falls back. True preemption
needs a subprocess, which is recorded as future work in docs/TASK_PLAN.md
rather than pretended to exist here.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class CrashWindow:
    """A sliding-window failure counter.

    WHY A WINDOW rather than a plain total: over a 30-minute meeting a handful
    of scattered failures is normal and survivable -- a garbled utterance, one
    CUDA hiccup. Three failures inside a minute is a different thing: the stage
    is broken *now* and every further attempt burns latency the user is waiting
    on. A lifetime total cannot tell those apart, so it would either trip on a
    healthy long session or never trip at all.

    The clock is injectable so tests can drive it deterministically instead of
    sleeping, which keeps the suite fast and non-flaky.
    """

    def __init__(self, *, limit: int = 3, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        # Monotonic timestamps of recent failures, oldest first. Bounded in
        # practice by _evict(), so this cannot grow over a long session.
        self._failures: deque[float] = deque()

    def record_failure(self) -> bool:
        """Note one failure. Returns True when the window is now over its limit."""
        now = self._clock()
        self._failures.append(now)
        self._evict(now)
        return len(self._failures) >= self.limit

    def record_success(self) -> None:
        """A success clears the window.

        WHY clear rather than decay: the point of the counter is to detect a
        stage that is broken *right now*. One good utterance proves it is not,
        so carrying old failures forward would make the next unrelated hiccup
        trip the limit early and demote a working provider.
        """
        self._failures.clear()

    @property
    def tripped(self) -> bool:
        self._evict(self._clock())
        return len(self._failures) >= self.limit

    @property
    def recent_failures(self) -> int:
        self._evict(self._clock())
        return len(self._failures)

    def _evict(self, now: float) -> None:
        # Drop anything that has aged out of the window. Cheap because the
        # deque is ordered by construction: stop at the first live entry.
        cutoff = now - self.window_s
        while self._failures and self._failures[0] <= cutoff:
            self._failures.popleft()


@dataclass(slots=True)
class Attempt(Generic[T]):
    """The outcome of one supervised call.

    Carries the duration because the caller -- not the supervisor -- owns the
    per-stage budget, and because a stage that *succeeded* but took 9 seconds
    is still a problem worth acting on. See the module docstring on why this is
    measured after the fact rather than preempted.
    """
    ok: bool
    value: T | None = None
    error: BaseException | None = None
    duration_ms: float = 0.0
    #: True when the failure looks like memory exhaustion (host RAM or CUDA
    #: VRAM), which the caller treats as "drop to a smaller model / CPU"
    #: rather than "retry the same thing and fail identically".
    out_of_memory: bool = False


def looks_like_oom(exc: BaseException) -> bool:
    """Classify an exception as memory exhaustion.

    WHY STRING MATCHING: torch raises ``torch.cuda.OutOfMemoryError`` for VRAM,
    but importing torch here would pull a multi-second, multi-hundred-MB import
    into a module that must stay cheap -- and Vaani must also run on machines
    where torch is absent or CPU-only. The CUDA allocator's message is stable
    and distinctive, so matching on it costs nothing and keeps this module
    dependency-free. Host-RAM exhaustion is caught precisely via MemoryError.
    """
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__
    if name in {"OutOfMemoryError", "CudaOutOfMemoryError"}:
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


class StageSupervisor:
    """Runs a callable, contains its failures, and decides when to give up.

    Usage in the engine's worker loop::

        attempt = supervisor.run(self.pipeline.process, utterance)
        if not attempt.ok:
            continue            # utterance dropped; session stays alive

    WHERE IT GOES NEXT: ``exhausted`` is polled by the engine, which reports
    degraded mode upward to the overlay/CLI.
    """

    def __init__(self, name: str, *, limit: int = 3, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 on_failure: Callable[[str, BaseException], None] | None = None,
                 on_exhausted: Callable[[str], None] | None = None) -> None:
        self.name = name
        self._window = CrashWindow(limit=limit, window_s=window_s, clock=clock)
        self._clock = clock
        self._on_failure = on_failure
        self._on_exhausted = on_exhausted
        self._exhausted = False
        # Lifetime totals, for the end-of-session report and the soak harness.
        self.total_failures = 0
        self.total_calls = 0

    @property
    def exhausted(self) -> bool:
        """True once the failure window tripped. Latched until :meth:`reset`.

        WHY LATCHED: the window self-heals as failures age out, so an unlatched
        flag would flicker between degraded and healthy while the user is still
        looking at the banner. Falling back is a deliberate, announced decision;
        coming back from it should be too.
        """
        return self._exhausted

    def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Attempt[T]:
        """Call ``fn``, catching anything it raises.

        Catches ``BaseException``, not ``Exception``: a native extension can
        surface almost anything, and the entire purpose of this method is that
        the worker thread survives the call. KeyboardInterrupt and SystemExit
        are re-raised immediately -- those mean the *user* is stopping Vaani,
        and swallowing them would make Ctrl-C not work.
        """
        self.total_calls += 1
        started = self._clock()
        try:
            value = fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            # Deliberate shutdown, not a stage failure. Let it propagate so the
            # session's normal stop path runs and the virtual mic is unloaded.
            raise
        except BaseException as exc:        # noqa: BLE001 - containment is the point
            duration_ms = (self._clock() - started) * 1000.0
            self.total_failures += 1
            tripped = self._window.record_failure()
            if self._on_failure is not None:
                # A reporting callback must never be able to escalate a
                # contained failure into the thread death we are preventing.
                try:
                    self._on_failure(self.name, exc)
                except Exception:
                    pass
            if tripped and not self._exhausted:
                self._exhausted = True
                if self._on_exhausted is not None:
                    try:
                        self._on_exhausted(self.name)
                    except Exception:
                        pass
            return Attempt(ok=False, error=exc, duration_ms=duration_ms,
                           out_of_memory=looks_like_oom(exc))

        duration_ms = (self._clock() - started) * 1000.0
        self._window.record_success()
        return Attempt(ok=True, value=value, duration_ms=duration_ms)

    def reset(self) -> None:
        """Clear the latch and the window, e.g. after switching to a fallback
        provider -- the new provider deserves its own three chances."""
        self._window.record_success()
        self._exhausted = False
