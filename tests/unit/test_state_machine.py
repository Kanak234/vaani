"""State machine tests, centred on the audio-emission invariant."""
from __future__ import annotations

import pytest

from vaani.core.state_machine import (
    AUDIO_EMITTING_STATES,
    InvalidTransition,
    SessionState,
    SessionStateMachine,
)


@pytest.fixture
def sm():
    return SessionStateMachine()


def to_listening(sm):
    sm.transition(SessionState.INITIALIZING)
    sm.transition(SessionState.LISTENING)


def test_starts_idle(sm):
    assert sm.state is SessionState.IDLE
    assert not sm.can_emit_audio


def test_happy_path(sm):
    to_listening(sm)
    for s in (SessionState.TRANSCRIBING, SessionState.TRANSLATING,
              SessionState.GATING, SessionState.SYNTHESIZING,
              SessionState.OUTPUTTING, SessionState.LISTENING):
        sm.transition(s)
    assert sm.state is SessionState.LISTENING


def test_illegal_transition_rejected(sm):
    with pytest.raises(InvalidTransition):
        sm.transition(SessionState.OUTPUTTING)   # straight from IDLE


def test_only_outputting_emits_audio(sm):
    """The core safety invariant."""
    assert AUDIO_EMITTING_STATES == {SessionState.OUTPUTTING}
    to_listening(sm)
    for s in (SessionState.TRANSCRIBING, SessionState.TRANSLATING,
              SessionState.GATING, SessionState.SYNTHESIZING):
        sm.transition(s)
        assert not sm.can_emit_audio, f"{s} must not emit audio"
    sm.transition(SessionState.OUTPUTTING)
    assert sm.can_emit_audio


def test_no_path_from_a_failed_stage_to_audio(sm):
    """A suppressed utterance cannot reach OUTPUTTING (AC-12.4)."""
    to_listening(sm)
    sm.transition(SessionState.TRANSCRIBING)
    sm.transition(SessionState.SUPPRESSED)
    assert not sm.can_emit_audio
    with pytest.raises(InvalidTransition):
        sm.transition(SessionState.OUTPUTTING)


def test_suppressed_returns_to_listening(sm):
    """Suppression is a normal outcome, not an error state."""
    to_listening(sm)
    sm.transition(SessionState.TRANSCRIBING)
    sm.transition(SessionState.SUPPRESSED)
    sm.transition(SessionState.LISTENING)
    assert sm.state is SessionState.LISTENING


@pytest.mark.parametrize("state", [
    SessionState.LISTENING, SessionState.TRANSCRIBING, SessionState.SYNTHESIZING,
    SessionState.OUTPUTTING, SessionState.PAUSED, SessionState.DEGRADED,
])
def test_emergency_stop_reachable_from_any_state(state):
    """AC-12.1: an emergency stop that can be refused is not one."""
    sm = SessionStateMachine()
    sm.force(state)
    assert sm.emergency_stop() is SessionState.STOPPING
    assert not sm.can_emit_audio


def test_emergency_stop_from_outputting_halts_audio():
    sm = SessionStateMachine()
    sm.force(SessionState.OUTPUTTING)
    assert sm.can_emit_audio
    sm.emergency_stop()
    assert not sm.can_emit_audio


def test_listeners_notified(sm):
    seen = []
    sm.subscribe(lambda a, b: seen.append((a, b)))
    sm.transition(SessionState.INITIALIZING)
    assert seen == [(SessionState.IDLE, SessionState.INITIALIZING)]


def test_listener_exception_does_not_break_transition(sm):
    """A crashing UI listener must not take down the audio path."""
    sm.subscribe(lambda a, b: (_ for _ in ()).throw(RuntimeError("ui bug")))
    sm.transition(SessionState.INITIALIZING)
    assert sm.state is SessionState.INITIALIZING


def test_degraded_is_active_not_fatal(sm):
    sm.force(SessionState.DEGRADED)
    assert sm.is_active
