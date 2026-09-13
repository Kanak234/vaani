from vaani.session.takeover import (
    MeetingTakeoverController,
    TakeoverAction,
    TakeoverConfig,
    TakeoverState,
)


def test_takeover_requires_arm() -> None:
    controller = MeetingTakeoverController()
    assert controller.take_over().action is TakeoverAction.WAIT
    assert controller.state is TakeoverState.OFF


def test_explicit_takeover_activates() -> None:
    controller = MeetingTakeoverController()
    controller.arm()
    decision = controller.observe_user_speech("Vaani, take over")
    assert decision.action is TakeoverAction.TAKEOVER
    assert controller.active


def test_question_plus_repeated_hesitation_can_trigger_takeover() -> None:
    controller = MeetingTakeoverController(
        config=TakeoverConfig(silence_after_question_s=4.0, min_hesitation_events=2)
    )
    controller.arm()
    controller.observe_question("Can you explain the architecture?", now=100.0)

    assert controller.observe_user_speech("uh", hesitation=True, now=101.0).action is TakeoverAction.WAIT
    assert controller.observe_user_speech("I...", hesitation=True, now=102.0).action is TakeoverAction.WAIT
    decision = controller.poll(now=106.1)

    assert decision.action is TakeoverAction.TAKEOVER
    assert decision.reason == "hesitation_timeout"
    assert controller.active


def test_substantive_user_answer_prevents_automatic_takeover() -> None:
    controller = MeetingTakeoverController(
        config=TakeoverConfig(silence_after_question_s=4.0, min_hesitation_events=2)
    )
    controller.arm()
    controller.observe_question("What did you change?", now=100.0)
    decision = controller.observe_user_speech(
        "I changed the translation router and the tests.", now=101.0
    )

    assert decision.action is TakeoverAction.USER
    assert controller.poll(now=110.0).action is TakeoverAction.WAIT


def test_stop_disables_active_takeover() -> None:
    controller = MeetingTakeoverController()
    controller.arm()
    controller.take_over()
    decision = controller.stop()

    assert decision.action is TakeoverAction.STOP
    assert controller.state is TakeoverState.OFF
    assert controller.poll(now=999.0).action is TakeoverAction.WAIT
