"""Tests for system/evdev_ptt.py.

These run on a machine with NO evdev installed and NO access to /dev/input --
which is the machine this was developed on, and the normal case for a fresh
install. The event-dispatch rules are therefore tested against fake events,
so the behaviour that would otherwise only be checked by physically holding a
key during a real meeting is covered here in microseconds.
"""
from __future__ import annotations

from dataclasses import dataclass

from vaani.system import evdev_ptt
from vaani.system.evdev_ptt import (
    EV_KEY, KEY_DOWN, KEY_HOLD, KEY_UP, Availability, dispatch_events,
)

KEY_F8 = 66
EV_REL = 0x02


@dataclass
class FakeEvent:
    """Shaped like evdev.InputEvent in the only three fields we read."""
    type: int
    code: int
    value: int


class Recorder:
    def __init__(self):
        self.events: list[str] = []

    def press(self):
        self.events.append("press")

    def release(self):
        self.events.append("release")


def run(events, *, code: int = KEY_F8) -> list[str]:
    r = Recorder()
    dispatch_events(events, code=code, on_press=r.press, on_release=r.release)
    return r.events


# ------------------------------------------------------------- the basics

def test_press_then_release():
    assert run([FakeEvent(EV_KEY, KEY_F8, KEY_DOWN),
                FakeEvent(EV_KEY, KEY_F8, KEY_UP)]) == ["press", "release"]


def test_autorepeat_does_not_fire_repeatedly():
    """The kernel repeats KEY_HOLD many times a second while a key is held.
    Treating those as presses would hammer the callback dozens of times for one
    physical hold."""
    events = [FakeEvent(EV_KEY, KEY_F8, KEY_DOWN)]
    events += [FakeEvent(EV_KEY, KEY_F8, KEY_HOLD)] * 50
    events += [FakeEvent(EV_KEY, KEY_F8, KEY_UP)]
    assert run(events) == ["press", "release"]


def test_a_long_hold_is_one_press_and_one_release():
    events = ([FakeEvent(EV_KEY, KEY_F8, KEY_DOWN)]
              + [FakeEvent(EV_KEY, KEY_F8, KEY_HOLD)] * 500
              + [FakeEvent(EV_KEY, KEY_F8, KEY_UP)])
    assert run(events).count("press") == 1


# --------------------------------------------------- what must be ignored

def test_other_keys_are_ignored():
    """THE privacy-relevant test. This process can see every keystroke on the
    machine; it must react to exactly one key and nothing else."""
    other_keys = [FakeEvent(EV_KEY, c, KEY_DOWN) for c in range(1, 120)
                  if c != KEY_F8]
    assert run(other_keys) == []


def test_non_key_events_are_ignored():
    """A device reports pointer motion and much else besides keys."""
    assert run([FakeEvent(EV_REL, 0, 1), FakeEvent(EV_REL, 1, -3)]) == []


def test_a_different_configured_key_is_respected():
    events = [FakeEvent(EV_KEY, 30, KEY_DOWN), FakeEvent(EV_KEY, 30, KEY_UP)]
    assert run(events, code=30) == ["press", "release"]
    assert run(events, code=KEY_F8) == []


def test_unknown_event_values_are_ignored():
    """Kernels may grow new values; an unrecognised one must not be guessed at."""
    assert run([FakeEvent(EV_KEY, KEY_F8, 99)]) == []


# ------------------------------------------------------- odd event orders

def test_release_without_a_press_still_releases():
    """A key already held when the listener starts produces a lone key-up.
    Swallowing it would leave the user stuck live."""
    assert run([FakeEvent(EV_KEY, KEY_F8, KEY_UP)]) == ["release"]


def test_repeated_key_down_without_release():
    """Focus changes and device quirks can produce this; it must not crash."""
    assert run([FakeEvent(EV_KEY, KEY_F8, KEY_DOWN)] * 3) == ["press"] * 3


def test_an_empty_stream_does_nothing():
    assert run([]) == []


# ---------------------------------------------------------- containment

def test_a_raising_callback_does_not_stop_the_listener():
    """A bug in the UI must not strand the user unable to talk for the rest of
    the meeting."""
    calls: list[str] = []

    def bad_press():
        calls.append("press")
        raise ValueError("overlay bug")

    def good_release():
        calls.append("release")

    dispatch_events([FakeEvent(EV_KEY, KEY_F8, KEY_DOWN),
                     FakeEvent(EV_KEY, KEY_F8, KEY_UP)],
                    code=KEY_F8, on_press=bad_press, on_release=good_release)
    # The release still arrived despite the press callback raising.
    assert calls == ["press", "release"]


def test_malformed_events_are_skipped_not_fatal():
    class Weird:
        pass

    assert run([Weird(), FakeEvent(EV_KEY, KEY_F8, KEY_DOWN)]) == ["press"]


# ------------------------------------------------------- availability API

def test_availability_names_the_specific_problem():
    """"hold-to-talk unavailable" is useless: the user cannot tell whether to
    pip install something or change a unix group."""
    a = evdev_ptt.availability()
    assert isinstance(a, Availability)
    if not a.ok:
        assert a.reason
        # On this machine evdev is absent; either way the reason must name
        # which of the two prerequisites is missing.
        assert ("evdev" in a.reason or "input" in a.reason
                or "device" in a.reason)


def test_availability_offers_a_remedy_for_fixable_problems():
    a = evdev_ptt.availability()
    if not a.ok and "evdev" in a.reason and "not installed" in a.reason:
        assert a.remedy is not None
        assert "pip install evdev" in a.remedy


def test_helpers_degrade_quietly_without_evdev():
    """Everything must stay importable and callable on a machine with no evdev,
    because that is the default machine."""
    assert isinstance(evdev_ptt.find_keyboards(), list)


def test_listener_reports_instead_of_raising_when_unavailable():
    """Starting a MEETING must not fail because a convenience feature cannot
    start."""
    listener = evdev_ptt.PttListener(on_press=lambda: None,
                                     on_release=lambda: None)
    result = listener.start()
    assert isinstance(result, Availability)
    assert result.ok is False or result.ok is True      # never an exception
    listener.stop()                                     # idempotent, safe


def test_stop_is_safe_before_start():
    listener = evdev_ptt.PttListener(on_press=lambda: None,
                                     on_release=lambda: None)
    listener.stop()
    listener.stop()
