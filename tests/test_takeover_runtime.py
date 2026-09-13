from __future__ import annotations

from types import SimpleNamespace

from vaani.session.takeover import TakeoverAction
from vaani.session.takeover_runtime import MeetingTakeoverRuntime


class FakeAssistant:
    def draft(self, text, **kwargs):
        return SimpleNamespace(final_text=f"Answering: {text}")


class FakeSynthesizer:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, *, voice_profile_id=None):
        self.calls.append((text, voice_profile_id))
        return "audio"


def test_runtime_does_not_speak_while_only_armed():
    synth = FakeSynthesizer()
    runtime = MeetingTakeoverRuntime(assistant=FakeAssistant(), synthesizer=synth)
    runtime.arm()
    runtime.observe_remote_question("Can you explain the approach?", now=10.0)

    assert runtime.poll(now=12.0) is TakeoverAction.WAIT
    assert runtime.respond_if_authorized(now=12.0) is None
    assert synth.calls == []


def test_runtime_speaks_after_hesitation_timeout():
    synth = FakeSynthesizer()
    runtime = MeetingTakeoverRuntime(assistant=FakeAssistant(), synthesizer=synth)
    runtime.arm()
    runtime.observe_remote_question("Can you explain the approach?", now=10.0)
    runtime.observe_user_speech("um", hesitation=True, now=10.5)
    runtime.observe_user_speech("uh", hesitation=True, now=11.0)

    response = runtime.respond_if_authorized(now=14.0)

    assert response is not None
    assert response.question == "Can you explain the approach?"
    assert response.text == "Answering: Can you explain the approach?"
    assert synth.calls == [(response.text, None)]


def test_runtime_stop_cancels_pending_takeover():
    synth = FakeSynthesizer()
    runtime = MeetingTakeoverRuntime(assistant=FakeAssistant(), synthesizer=synth)
    runtime.arm()
    runtime.observe_remote_question("Are you available?", now=10.0)
    runtime.observe_user_speech("um", hesitation=True, now=10.5)
    runtime.observe_user_speech("uh", hesitation=True, now=11.0)
    runtime.stop()

    assert runtime.respond_if_authorized(now=20.0) is None
    assert synth.calls == []
