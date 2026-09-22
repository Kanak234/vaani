"""Physical hold-to-talk, read straight from the keyboard device.

WHY THIS IS OPTIONAL AND NOT THE DEFAULT
----------------------------------------
Every other Vaani control goes through the control socket, so a plain desktop
shortcut running `vaani talk` is enough and needs no special rights. True
hold-to-talk cannot work that way: a desktop shortcut fires on key PRESS and
tells you nothing about the RELEASE, and "translate only while I hold this key"
is defined by the release. Reading the key directly is the only way to get it.

That costs two things the other controls do not need:

  1. the ``evdev`` package, and
  2. read access to /dev/input/event*, which on a normal system means being in
     the ``input`` group.

Neither is present on a fresh install, and the second needs a privileged
``usermod`` plus a re-login. So ptt_hold is opt-in, :func:`availability` reports
exactly which of the two is missing, and the caller degrades to ptt_toggle
rather than failing. A meeting must never fail to start because a convenience
feature is unavailable.

SECURITY NOTE, stated plainly
-----------------------------
Reading /dev/input/event* means this process can see EVERY keystroke on the
machine, including passwords typed into other windows. That is inherent to the
mechanism, not to this implementation. This module therefore does the minimum
that is compatible with the feature:

  * it matches ONE configured key code and ignores every other key, and
  * it never stores, logs or forwards a key code -- not even the one it matches.
    Only "pressed" and "released" leave this module.

WHERE THE DATA FLOWS NEXT
-------------------------
``on_press``/``on_release`` are wired to ``TranslationSession.press_talk`` and
``release_talk``, which drive the same InputModeController the CLI and the
overlay drive. This module is one more input into that state machine, never a
second source of truth.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass

#: evdev's EV_KEY. Defined here rather than imported so that this module -- and
#: therefore anything that imports it -- stays importable on a machine with no
#: evdev at all. That is the normal case, so it must not be an import error.
EV_KEY = 0x01

#: Key event values as the kernel reports them.
KEY_UP = 0
KEY_DOWN = 1
KEY_HOLD = 2        # autorepeat while physically held


@dataclass(frozen=True, slots=True)
class Availability:
    """Whether hold-to-talk can actually run here, and if not, what to do."""
    ok: bool
    reason: str
    #: A command the user can run, when the problem is fixable. None when it is
    #: not, so the caller never prints an empty "try:" line.
    remedy: str | None = None


def availability() -> Availability:
    """Report whether evdev and device permissions are both present.

    Deliberately checks the two failures SEPARATELY and names the one that
    applies. "hold-to-talk unavailable" on its own is useless; the user cannot
    tell whether to install a package or change a group, and the two have very
    different costs (one is a pip install, the other needs sudo and a re-login).
    """
    try:
        import evdev                                     # noqa: F401
    except ImportError:
        return Availability(
            False, "the 'evdev' package is not installed",
            "pip install evdev   (or: vaani --input-mode ptt_toggle)")

    try:
        import evdev
        devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
        for d in devices:
            d.close()
    except PermissionError:
        return Availability(
            False, "no read access to /dev/input/event* (not in the 'input' group)",
            "sudo usermod -aG input $USER   then log out and back in")
    except Exception as exc:
        return Availability(False, f"could not enumerate input devices: {exc}", None)

    if not devices:
        return Availability(False, "no readable input devices found", None)
    return Availability(True, "evdev present and input devices readable")


def find_keyboards() -> list[str]:
    """Paths of devices that look like keyboards.

    A keyboard is identified by capability -- it reports EV_KEY and has the
    letter keys -- rather than by name. Name matching fails on exactly the
    hardware people care about: laptop internal keyboards, external keyboards
    and virtual ones are named inconsistently across vendors and locales.
    """
    try:
        import evdev
    except ImportError:
        return []

    found: list[str] = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except Exception:
            continue                    # a device we cannot open is not a candidate
        try:
            keys = set(dev.capabilities().get(EV_KEY, []))
            # KEY_A (30) and KEY_Z (44): present on a real keyboard, absent on a
            # mouse or a power button, both of which also report EV_KEY.
            if {30, 44} <= keys:
                found.append(path)
        except Exception:
            continue
        finally:
            try:
                dev.close()
            except Exception:
                pass
    return found


def key_code(name: str) -> int | None:
    """Translate a key NAME such as "KEY_F8" into its kernel code.

    Names rather than numbers in the config, because a user editing a config
    file can reason about KEY_F8 and cannot reason about 66.
    """
    try:
        from evdev import ecodes
    except ImportError:
        return None
    code = getattr(ecodes, name, None)
    return code if isinstance(code, int) else None


def dispatch_events(events: Iterable, *, code: int,
                    on_press: Callable[[], None],
                    on_release: Callable[[], None]) -> None:
    """Turn a stream of raw events into press/release callbacks.

    Split out from the thread and the device so it can be tested with a list of
    fake events -- no evdev, no /dev/input, no root. This is where every rule
    about WHICH events matter lives, so those rules are all covered by fast
    unit tests rather than by holding a key during a real meeting.

    Rules, and why:
      * non-EV_KEY events are ignored       - the device reports many types
      * other key codes are ignored         - see the security note above
      * KEY_HOLD (autorepeat) is ignored    - the kernel repeats while held;
                                              treating those as presses would
                                              fire the callback dozens of times
                                              per second
      * a callback that raises is contained - a UI bug must not kill the
                                              listener and strand the user
                                              unable to talk
    """
    for event in events:
        if getattr(event, "type", None) != EV_KEY:
            continue
        if getattr(event, "code", None) != code:
            continue
        value = getattr(event, "value", None)
        if value == KEY_DOWN:
            _safely(on_press)
        elif value == KEY_UP:
            _safely(on_release)
        # KEY_HOLD deliberately falls through: no callback.


def _safely(fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception:
        pass


class PttListener:
    """Watches one key on one device in a background thread.

    Start it and forget it; :meth:`stop` is idempotent and safe from any
    thread. If the device disappears mid-meeting (a USB keyboard unplugged),
    the thread ends quietly rather than raising into a meeting -- the user
    falls back to `vaani talk`, which always works.
    """

    def __init__(self, *, key: str = "KEY_F8", device_path: str | None = None,
                 on_press: Callable[[], None],
                 on_release: Callable[[], None]) -> None:
        self.key = key
        self.device_path = device_path
        self._on_press = on_press
        self._on_release = on_release
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._device = None

    def start(self) -> Availability:
        """Begin listening. Returns why it could not start, rather than raising.

        The caller is starting a MEETING. An unavailable convenience feature
        must be reportable without aborting that, which is why every failure
        here comes back as an Availability the caller can print and continue
        past.
        """
        avail = availability()
        if not avail.ok:
            return avail

        code = key_code(self.key)
        if code is None:
            return Availability(False, f"unknown key name {self.key!r}",
                                "use a name from `python -m evdev.evtest`, e.g. KEY_F8")

        path = self.device_path
        if path is None:
            keyboards = find_keyboards()
            if not keyboards:
                return Availability(False, "no keyboard device found", None)
            # First match. A machine with several keyboards needs an explicit
            # path; guessing between them would silently watch the wrong one.
            path = keyboards[0]

        try:
            import evdev
            self._device = evdev.InputDevice(path)
        except Exception as exc:
            return Availability(False, f"could not open {path}: {exc}", None)

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(code,),
                                        name="vaani-ptt", daemon=True)
        self._thread.start()
        return Availability(True, f"hold {self.key} to talk ({path})")

    def _run(self, code: int) -> None:
        try:
            dispatch_events(self._iter_events(), code=code,
                            on_press=self._on_press, on_release=self._on_release)
        except Exception:
            # Device unplugged, permissions revoked, evdev internal error. The
            # meeting continues; `vaani talk` still works.
            pass
        finally:
            self._close_device()

    def _iter_events(self):
        """Yield events until stopped.

        read_loop() blocks, so the stop flag is checked between events. A key
        that is never pressed again leaves the thread parked in the kernel --
        harmless, because it is a daemon thread and the device is closed in the
        finally above when the process exits.
        """
        for event in self._device.read_loop():
            if self._stop.is_set():
                return
            yield event

    def stop(self) -> None:
        self._stop.set()
        self._close_device()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            # Short join: the thread may be parked in a blocking read that only
            # ends on the next keystroke, and waiting for that would hang exit.
            thread.join(timeout=0.5)

    def _close_device(self) -> None:
        device, self._device = self._device, None
        if device is not None:
            try:
                device.close()
            except Exception:
                pass
