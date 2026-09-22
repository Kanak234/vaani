"""Windows audio-device enumeration and virtual-mic endpoint policy."""
from __future__ import annotations

import enum
from dataclasses import dataclass

from ..core.errors import ErrorCode, Severity, VaaniError

VIRTUAL_MIC_NAME = "VaaniVirtualMic"
VIRTUAL_MIC_DESCRIPTION = "Vaani Virtual Microphone"
VIRTUAL_SINK_NAME = "VaaniSink"


class DeviceKind(enum.Enum):
    INPUT = "input"
    OUTPUT = "output"
    VIRTUAL_SOURCE = "virtual_source"


@dataclass(frozen=True, slots=True)
class AudioDevice:
    key: str
    display_name: str
    kind: DeviceKind
    channels: int
    sample_rate: int
    is_virtual: bool = False
    is_monitor: bool = False

    def __str__(self) -> str:
        return f"{self.display_name} ({self.key})"


def _sc():
    try:
        import soundcard as sc
        return sc
    except ImportError as exc:
        raise VaaniError(code=ErrorCode.AUDIO_BACKEND_UNAVAILABLE,
                         message="Windows audio support requires the 'soundcard' package; install Vaani with the Windows extras",
                         severity=Severity.FATAL, cause=exc) from exc


_VIRTUAL_CABLE_PATTERNS = ('cable', 'vb-audio', 'virtual audio', 'voicemeeter', 'blackhole')

def _is_virtual_device(name: str) -> bool:
    lower = name.lower()
    return any(p in lower for p in _VIRTUAL_CABLE_PATTERNS)

def list_sources(*, include_monitors: bool = False) -> list[AudioDevice]:
    sc = _sc()
    out = []
    for d in sc.all_microphones(include_loopback=include_monitors):
        # API may not expose channels/samplerate; use defaults if absent
        ch = getattr(d, "channels", 1)
        if type(ch) is list: ch = len(ch)
        sr = getattr(d, "samplerate", 16000)
        is_loop = getattr(d, "isloopback", False) or "loopback" in d.name.lower()
        out.append(AudioDevice(
            key=d.name, display_name=d.name, kind=DeviceKind.INPUT,
            channels=ch if isinstance(ch, int) else 1, sample_rate=sr,
            is_virtual=_is_virtual_device(d.name),
            is_monitor=is_loop
        ))
    return out

def list_sinks() -> list[AudioDevice]:
    sc = _sc()
    out = []
    for d in sc.all_speakers():
        ch = getattr(d, "channels", 2)
        if type(ch) is list: ch = len(ch)
        sr = getattr(d, "samplerate", 48000)
        out.append(AudioDevice(
            key=d.name, display_name=d.name, kind=DeviceKind.OUTPUT,
            channels=ch if isinstance(ch, int) else 2, sample_rate=sr,
            is_virtual=_is_virtual_device(d.name),
        ))
    return out


def assert_no_feedback_loop(input_device_key: str, virtual_mic_name: str) -> None:
    if virtual_mic_name.lower() in input_device_key.lower():
        raise VaaniError(code=ErrorCode.FEEDBACK_LOOP_DETECTED,
                         message="the input device is Vaani's own virtual microphone; select your real microphone instead",
                         severity=Severity.FATAL,
                         detail={"input_device": input_device_key})


def virtual_mic_consumers(node_name: str = VIRTUAL_MIC_NAME):
    return []


@dataclass(slots=True)
class VirtualMicrophone:
    """Windows virtual-mic endpoint backed by an external virtual cable."""
    sink_name: str
    node_name: str = VIRTUAL_MIC_NAME
    sink_module_id: int | None = None
    source_module_id: int | None = None
    _owned: bool = False

    @classmethod
    def create(cls, *, reuse_existing: bool = True, sink_name: str | None = None):
        if not sink_name:
            for d in list_sinks():
                if d.is_virtual:
                    sink_name = d.key
                    break
        if not sink_name:
            raise VaaniError(
                code=ErrorCode.VIRTUAL_MIC_CREATE_FAILED,
                message="Virtual audio cable not detected. Install VB-CABLE or similar.",
                severity=Severity.SESSION
            )
        return cls(sink_name=sink_name)

    @staticmethod
    def find_existing() -> str | None:
        return None

    def exists(self) -> bool:
        return any(d.key == self.sink_name for d in list_sinks())

    def destroy(self) -> None:
        return

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.destroy()


def main() -> int:
    print("CAPTURE DEVICES (use with --input / --remote-input)")
    for d in list_sources(include_monitors=True):
        tag = " [LOOPBACK]" if d.is_monitor else ""
        print(f"  {d.display_name}{tag}\n      key: {d.key}")
    print("\nPLAYBACK DEVICES (use with --output-device / --monitor)")
    for d in list_sinks():
        print(f"  {d.display_name}\n      key: {d.key}")
    print("\nWindows meeting mode needs a virtual audio cable: Vaani writes to its playback endpoint,")
    print("and the meeting app selects the matching recording endpoint as its microphone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
