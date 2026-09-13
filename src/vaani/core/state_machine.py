"""Session state machine (Architecture §D).

Enforces the invariant the whole safety story rests on:

    audio may only be written to the virtual microphone while in OUTPUTTING.

Every other state writes silence. Because transitions are validated here rather
than left to the pipeline's control flow, there is no code path from a failed
stage to the speaker -- a failed stage cannot reach OUTPUTTING at all.
"""
from __future__ import annotations

import enum
import threading
import collections
from collections.abc import Callable
from dataclasses import dataclass, field


class SessionState(enum.Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    GATING = "gating"
    SYNTHESIZING = "synthesizing"
    OUTPUTTING = "outputting"
    SUPPRESSED = "suppressed"
    PAUSED = "paused"
    MUTED = "muted"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    STOPPING = "stopping"
    FAILED = "failed"


#: The only state from which audio reaches the virtual microphone.
AUDIO_EMITTING_STATES = frozenset({SessionState.OUTPUTTING})

_ACTIVE = frozenset({
    SessionState.LISTENING, SessionState.TRANSCRIBING, SessionState.TRANSLATING,
    SessionState.GATING, SessionState.SYNTHESIZING, SessionState.OUTPUTTING,
    SessionState.SUPPRESSED, SessionState.DEGRADED,
})

_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.IDLE: frozenset({SessionState.INITIALIZING}),
    SessionState.INITIALIZING: frozenset({
        SessionState.LISTENING, SessionState.FAILED, SessionState.STOPPING}),
    SessionState.LISTENING: frozenset({
        SessionState.TRANSCRIBING, SessionState.PAUSED, SessionState.MUTED,
        SessionState.RECOVERING, SessionState.STOPPING, SessionState.DEGRADED, SessionState.FAILED}),
    SessionState.TRANSCRIBING: frozenset({
        SessionState.TRANSLATING, SessionState.SUPPRESSED,
        SessionState.LISTENING, SessionState.RECOVERING, SessionState.STOPPING, SessionState.FAILED}),
    SessionState.TRANSLATING: frozenset({
        SessionState.GATING, SessionState.SUPPRESSED,
        SessionState.LISTENING, SessionState.RECOVERING, SessionState.STOPPING, SessionState.FAILED}),
    SessionState.GATING: frozenset({
        SessionState.SYNTHESIZING, SessionState.SUPPRESSED, SessionState.STOPPING, SessionState.FAILED}),
    SessionState.SYNTHESIZING: frozenset({
        SessionState.OUTPUTTING, SessionState.SUPPRESSED,
        SessionState.LISTENING, SessionState.RECOVERING, SessionState.STOPPING, SessionState.FAILED}),
    SessionState.OUTPUTTING: frozenset({
        SessionState.LISTENING, SessionState.RECOVERING, SessionState.STOPPING, SessionState.FAILED}),
    # SUPPRESSED is a NORMAL terminal state for an utterance, not an error:
    # it returns straight to listening.
    SessionState.SUPPRESSED: frozenset({SessionState.LISTENING, SessionState.STOPPING}),
    SessionState.PAUSED: frozenset({
        SessionState.LISTENING, SessionState.STOPPING, SessionState.MUTED}),
    SessionState.MUTED: frozenset({
        SessionState.LISTENING, SessionState.PAUSED, SessionState.STOPPING}),
    SessionState.DEGRADED: frozenset({
        SessionState.LISTENING, SessionState.TRANSCRIBING,
        SessionState.RECOVERING, SessionState.STOPPING}),
    SessionState.RECOVERING: frozenset({
        SessionState.LISTENING, SessionState.DEGRADED,
        SessionState.FAILED, SessionState.STOPPING}),
    SessionState.STOPPING: frozenset({SessionState.IDLE}),
    SessionState.FAILED: frozenset({SessionState.IDLE, SessionState.INITIALIZING}),
}


class InvalidTransition(RuntimeError):
    def __init__(self, current: SessionState, target: SessionState) -> None:
        super().__init__(f"illegal transition {current.value} -> {target.value}")
        self.current, self.target = current, target


@dataclass
class SessionStateMachine:
    """Thread-safe. The audio thread and the pipeline thread both read it."""

    _state: SessionState = field(default=SessionState.IDLE, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _listeners: list[Callable[[SessionState, SessionState], None]] = field(
        default_factory=list, init=False, repr=False)
    _history: collections.deque[SessionState] = field(
        default_factory=lambda: collections.deque(maxlen=500), init=False, repr=False)

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def can_emit_audio(self) -> bool:
        """The invariant, in one place. Checked immediately before every write."""
        with self._lock:
            return self._state in AUDIO_EMITTING_STATES

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._state in _ACTIVE

    def subscribe(self, listener: Callable[[SessionState, SessionState], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def can_transition(self, target: SessionState) -> bool:
        with self._lock:
            return target in _TRANSITIONS.get(self._state, frozenset())

    def transition(self, target: SessionState) -> SessionState:
        with self._lock:
            current = self._state
            if target is current:
                return current
            if target not in _TRANSITIONS.get(current, frozenset()):
                raise InvalidTransition(current, target)
            self._state = target
            self._history.append(target)
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(current, target)
            except Exception:
                # A misbehaving UI listener must never take down the audio path.
                pass
        return target

    def emergency_stop(self) -> SessionState:
        """Reachable from ANY state (AC-12.1).

        Deliberately bypasses the transition table: an emergency stop that could be
        refused because of the current state would not be an emergency stop.
        """
        with self._lock:
            current = self._state
            self._state = SessionState.STOPPING
            self._history.append(SessionState.STOPPING)
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(current, SessionState.STOPPING)
            except Exception:
                pass
        return SessionState.STOPPING

    def force(self, target: SessionState) -> None:
        """Set state without validation. Recovery paths only; never the happy path."""
        with self._lock:
            current, self._state = self._state, target
            self._history.append(target)
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(current, target)
            except Exception:
                pass

    @property
    def history(self) -> list[SessionState]:
        with self._lock:
            return list(self._history)
