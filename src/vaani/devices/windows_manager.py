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


def list_sources(*, include_monitors: bool = False) -> list[AudioDevice]:
    sc = _sc()
    return [AudioDevice(key=d.name, display_name=d.name, kind=DeviceKind.INPUT,
                        channels=1, sample_rate=16000,
                        is_monitor=getattr(d, "isloopback", False))
            for d in sc.all_microphones(include_loopback=include_monitors)]


def list_sinks() -> list[AudioDevice]:
    sc = _sc()
    return [AudioDevice(key=d.name, display_name=d.name, kind=DeviceKind.OUTPUT,
                        channels=2, sample_rate=48000)
            for d in sc.all_speakers()]


def assert_no_feedback_loop(input_device_key: str, virtual_mic_name: str) -> None:
    if virtual_mic_name.lower() in input_device_key.lower():
        raise VaaniError(code=ErrorCode.FEEDBACK_LOOP_DETECTED,
                         message="the input device is Vaani's own virtual microphone; select your real microphone instead",
                         severity=Severity.FATAL,
                         detail={"input_device": input_device_key})


def virtual_mic_consumers(node_name: str = VIRTUAL_MIC_NAME):
    # Windows does not expose PulseAudio source-output introspection. A meeting
    # client consuming a virtual cable endpoint cannot be reliably identified by
    # Vaani, so this API intentionally returns an empty list rather than guessing.
    return []


@dataclass(slots=True)
class VirtualMicrophone:
    """Windows virtual-mic endpoint backed by an external virtual cable.

    Vaani cannot create a kernel audio driver from Python. On Windows the supported
    topology is therefore: Vaani writes to a virtual-cable playback endpoint and
    the meeting application selects the cable's recording endpoint as its mic.
    """
    sink_name: str
    node_name: str = VIRTUAL_MIC_NAME
    sink_module_id: int | None = None
    source_module_id: int | None = None
    _owned: bool = False

    @classmethod
    def create(cls, *, reuse_existing: bool = True, sink_name: str | None = None):
        if not sink_name:
            raise VaaniError(code=ErrorCode.VIRTUAL_MIC_CREATE_FAILED,
                             message="Windows requires a virtual audio cable output device; pass its playback endpoint as the Vaani virtual output",
                             severity=Severity.SESSION)
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
