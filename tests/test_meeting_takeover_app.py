from vaani.session.meeting_takeover_app import looks_hesitant, looks_like_question


def test_question_detector_is_conservative():
    assert looks_like_question("What is the deployment plan?")
    assert looks_like_question("Could you explain the architecture")
    assert not looks_like_question("The deployment is tomorrow")
    assert not looks_like_question("")


def test_hesitation_detector_only_accepts_short_fragments():
    assert looks_hesitant("um maybe")
    assert looks_hesitant("I")
    assert not looks_hesitant("I will explain the architecture")
    assert not looks_hesitant("")
