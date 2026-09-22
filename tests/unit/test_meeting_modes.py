"""Phase 2: the audio actually routed by each input mode.

test_input_mode.py covers the decision; this file covers the consequence --
which samples reach the virtual microphone. It drives _capture_loop with a fake
capture stream instead of PulseAudio, so the routing is testable on a headless
machine rather than only by joining a real meeting and listening.
"""
from __future__ import annotations

import numpy as np
import pytest

from vaani.ai.tts.fallback import FallbackSynthesizer
from vaani.core.errors import ErrorCode, Severity, VaaniError
from vaani.core.types import Transcript, Translation
from vaani.session.engine import SAMPLE_RATE, SessionConfig, TranslationSession
from vaani.session.input_mode import InputMode

FRAME_N = SAMPLE_RATE * 20 // 1000      # 320 samples at 20 ms


class FakeCapture:
    """Yields a fixed number of marked frames, then reports the device as gone.

    WHY end with a VaaniError: that is the one exit _capture_loop already has
    besides the stop flag, so the loop terminates through its real code path
    rather than through something invented for the test.
    """

    def __init__(self, frames: int = 5, marker: float = 0.42):
        self.frames = frames
        self.marker = marker
        self.reads = 0

    def read_frame(self) -> np.ndarray:
        if self.reads >= self.frames:
            raise VaaniError(code=ErrorCode.DEVICE_DISCONNECTED,
                             message="end of fake capture",
                             severity=Severity.SESSION)
        self.reads += 1
        return np.full(FRAME_N, self.marker, dtype=np.float32)


class StubRecognizer:
    key = "stub_stt"

    def transcribe(self, audio, sample_rate, *, language_hint=None, context_prompt=None):
        return Transcript(text="नमस्ते", language_distribution={"hi": 1.0},
                          confidence=0.9)


class StubTranslator:
    key = "stub_mt"

    def translate(self, text, *, source_language, target_language, context=None):
        return Translation(text="Hello.", source_language=source_language,
                           target_language=target_language, confidence=0.9)


class CountingVad:
    """Never reports speech, so no utterance is ever segmented. That isolates
    the routing question from the segmentation question."""

    def __init__(self):
        self.calls = 0

    def is_speech(self, frame):
        self.calls += 1
        return 0.0

    def reset(self):
        pass


def make_session(**cfg) -> TranslationSession:
    return TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=CountingVad(),
        config=SessionConfig(use_virtual_mic=False, **cfg))


def drain(session) -> list[np.ndarray]:
    out = []
    while not session._out_queue.empty():
        out.append(session._out_queue.get_nowait())
    return out


# ------------------------------------------------------------------ bypass

def test_bypass_puts_the_real_microphone_on_the_virtual_mic():
    """THE point of bypass: the meeting hears the user's actual voice."""
    s = make_session()
    s._capture = FakeCapture(frames=5, marker=0.42)
    s.set_bypass(True)

    s._capture_loop()

    blocks = drain(s)
    assert len(blocks) == 5
    # The samples are the captured ones, untouched by STT/MT/TTS.
    assert all(np.allclose(b, 0.42) for b in blocks)


def test_bypass_does_not_run_the_vad_or_the_pipeline():
    """Passing the real voice through must not also transcribe it - that is
    wasted latency and a second voice waiting to happen."""
    s = make_session()
    vad = s.vad
    s._capture = FakeCapture(frames=5)
    s.set_bypass(True)

    s._capture_loop()

    assert vad.calls == 0
    assert s.total_utterances == 0


def test_entering_bypass_flushes_a_half_spoken_ai_sentence():
    """Otherwise the meeting hears the AI finishing its sentence underneath the
    user's real voice."""
    s = make_session()
    for _ in range(4):
        s._put_audio(np.zeros(FRAME_N, dtype=np.float32))
    assert s._out_queue.qsize() == 4

    s.set_bypass(True)

    assert s._out_queue.qsize() == 0


def test_leaving_bypass_stops_passing_the_real_voice_through():
    s = make_session()
    s.set_bypass(True)
    s.set_bypass(False)
    s._capture = FakeCapture(frames=5)

    s._capture_loop()

    # Back to normal: raw frames no longer reach the virtual mic.
    assert drain(s) == []


def test_toggle_bypass_reports_the_new_state():
    s = make_session()
    assert s.toggle_bypass() is True
    assert s.toggle_bypass() is False


# ------------------------------------------------------------ push-to-talk

@pytest.mark.parametrize("mode", [InputMode.PTT_TOGGLE, InputMode.PTT_HOLD])
def test_ptt_not_held_lets_nothing_into_the_pipeline(mode):
    s = make_session(input_mode=mode)
    vad = s.vad
    s._capture = FakeCapture(frames=5)

    s._capture_loop()

    assert vad.calls == 0            # never even reached the VAD
    assert drain(s) == []            # and nothing reached the virtual mic


@pytest.mark.parametrize("mode", [InputMode.PTT_TOGGLE, InputMode.PTT_HOLD])
def test_ptt_held_feeds_the_pipeline(mode):
    s = make_session(input_mode=mode)
    vad = s.vad
    s._capture = FakeCapture(frames=5)
    s.press_talk()

    s._capture_loop()

    assert vad.calls == 5            # every frame was considered for speech


def test_always_on_feeds_the_pipeline_without_a_key():
    s = make_session(input_mode=InputMode.ALWAYS_ON)
    vad = s.vad
    s._capture = FakeCapture(frames=5)

    s._capture_loop()

    assert vad.calls == 5


def test_toggle_talk_reports_whether_the_user_is_live():
    s = make_session(input_mode=InputMode.PTT_TOGGLE)
    assert s.toggle_talk() is True
    assert s.toggle_talk() is False


# ------------------------------------------------- mute still outranks modes

def test_mute_beats_an_active_ptt_press():
    """Mute is the user saying "nothing from me at all"; it must not be
    overridden by a held talk key."""
    s = make_session(input_mode=InputMode.PTT_HOLD)
    vad = s.vad
    s.press_talk()
    s.set_muted(True)
    s._capture = FakeCapture(frames=5)

    s._capture_loop()

    assert vad.calls == 0


def test_mute_beats_bypass():
    """Muting must stop the real voice reaching the meeting too, otherwise
    "mute" would not mean mute."""
    s = make_session()
    s.set_bypass(True)
    s.set_muted(True)
    s._capture = FakeCapture(frames=5)

    s._capture_loop()

    assert drain(s) == []


# -------------------------------------------------------- switching modes

def test_switching_into_ptt_mid_session_stops_translating():
    s = make_session(input_mode=InputMode.ALWAYS_ON)
    vad = s.vad
    s.set_input_mode(InputMode.PTT_TOGGLE)
    s._capture = FakeCapture(frames=5)

    s._capture_loop()

    assert vad.calls == 0
