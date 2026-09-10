"""Audio device enumeration and the virtual microphone lifecycle.

Device enumeration and module loading go through `pactl`, which is the stable,
documented interface to both PulseAudio and PipeWire's pulse compatibility layer.
Using the CLI here rather than binding libpulse's async introspection API is a
deliberate trade: enumeration happens on user action (not in the audio path), so
the ~15 ms of process spawn costs nothing, and it avoids binding a mainloop we
would otherwise need for this one purpose.
"""
from __future__ import annotations

import dataclasses
import enum
import shutil
import subprocess
from dataclasses import dataclass

from ..core.errors import ErrorCode, Severity, VaaniError

#: The source node meeting applications select as their microphone.
VIRTUAL_MIC_NAME = "VaaniVirtualMic"
VIRTUAL_MIC_DESCRIPTION = "Vaani Virtual Microphone"
#: The sink WE write synthesised audio into. Its monitor feeds the source above.
VIRTUAL_SINK_NAME = "VaaniSink"

_PACTL_TIMEOUT = 5.0


class DeviceKind(enum.Enum):
    INPUT = "input"
    OUTPUT = "output"
    VIRTUAL_SOURCE = "virtual_source"


@dataclass(frozen=True, slots=True)
class AudioDevice:
    key: str            # pactl node name -- what we pass to pa_simple_new
    display_name: str
    kind: DeviceKind
    channels: int
    sample_rate: int
    is_virtual: bool = False
    is_monitor: bool = False

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.display_name} ({self.key})"


def _pactl(*args: str) -> str:
    exe = shutil.which("pactl")
    if not exe:
        raise VaaniError(
            code=ErrorCode.AUDIO_BACKEND_UNAVAILABLE,
            message="`pactl` not found; Vaani needs PipeWire or PulseAudio",
            severity=Severity.FATAL,
        )
    try:
        proc = subprocess.run([exe, *args], capture_output=True, text=True,
                              timeout=_PACTL_TIMEOUT, check=False)
    except subprocess.TimeoutExpired as exc:
        raise VaaniError(code=ErrorCode.AUDIO_BACKEND_UNAVAILABLE,
                         message="audio server did not respond",
                         severity=Severity.SESSION, cause=exc) from exc
    if proc.returncode != 0:
        raise VaaniError(
            code=ErrorCode.AUDIO_BACKEND_UNAVAILABLE,
            message=f"pactl {' '.join(args)} failed: {proc.stderr.strip()}",
            severity=Severity.SESSION,
            detail={"returncode": proc.returncode},
        )
    return proc.stdout


def _parse_spec(spec: str) -> tuple[int, int]:
    """Parse a pactl format column like `s16le 2ch 48000Hz` -> (channels, rate)."""
    channels, rate = 1, 48000
    for token in spec.split():
        if token.endswith("ch"):
            try:
                channels = int(token[:-2])
            except ValueError:
                pass
        elif token.endswith("Hz"):
            try:
                rate = int(token[:-2])
            except ValueError:
                pass
    return channels, rate


def list_sources(*, include_monitors: bool = False) -> list[AudioDevice]:
    """All capture devices, including any Vaani virtual microphone."""
    devices: list[AudioDevice] = []
    for line in _pactl("list", "short", "sources").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        name = parts[1]
        is_monitor = name.endswith(".monitor")
        if is_monitor and not include_monitors:
            continue
        channels, rate = _parse_spec(parts[3])
        is_virtual = VIRTUAL_MIC_NAME in name
        devices.append(AudioDevice(
            key=name,
            display_name=VIRTUAL_MIC_DESCRIPTION if is_virtual else _pretty(name),
            kind=DeviceKind.VIRTUAL_SOURCE if is_virtual else DeviceKind.INPUT,
            channels=channels,
            sample_rate=rate,
            is_virtual=is_virtual,
            is_monitor=is_monitor,
        ))
    return devices


def list_sinks() -> list[AudioDevice]:
    """All playback devices -- used for the monitor output in conversation mode."""
    devices: list[AudioDevice] = []
    for line in _pactl("list", "short", "sinks").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        name = parts[1]
        channels, rate = _parse_spec(parts[3])
        devices.append(AudioDevice(
            key=name,
            display_name=_pretty(name),
            kind=DeviceKind.OUTPUT,
            channels=channels,
            sample_rate=rate,
            is_virtual=VIRTUAL_MIC_NAME in name,
        ))
    return devices


def _pretty(node_name: str) -> str:
    """Turn `alsa_input.pci-0000_06_00.6.HiFi__Mic1__source` into something readable."""
    name = node_name
    for prefix in ("alsa_input.", "alsa_output.", "bluez_input.", "bluez_output."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.endswith(".monitor"):
        name = name[: -len(".monitor")] + " (monitor)"
    name = name.replace("__", " ").replace("_", " ").replace(".", " ")
    return " ".join(name.split()).strip() or node_name


# --------------------------------------------------------------- virtual mic

@dataclass(slots=True)
class VirtualMicrophone:
    """A virtual microphone that meeting applications can select as an input.

    Built from TWO PipeWire modules, and the reason is worth recording because the
    obvious one-module version is broken in a way that is silent:

        module-null-sink media.class=Audio/Source/Virtual

    creates a node that appears correctly in the microphone list, but it is a
    SOURCE ONLY -- no sink exists. A PulseAudio playback stream targeting that name
    therefore cannot connect to it, and the Pulse API does not fail: it falls back
    to the default output. The result is a virtual mic that every app lists, that
    carries pure silence, while the translated speech plays out of the laptop
    speakers instead. This was observed and measured on the target machine.

    The working topology is:

        module-null-sink     sink_name=VaaniSink        <- we write here
                 |
                 v  (its .monitor)
        module-remap-source  source_name=VaaniVirtualMic <- apps read here

    Verified by round trip on PipeWire 1.6.2: a 0.500-amplitude tone written to the
    sink was recovered from the source at 0.500. `module-remap-source` is used
    rather than exposing the monitor directly because many applications filter
    monitor devices out of their microphone lists, and those that do not label them
    "Monitor of ...", which is not a name a user should have to decode mid-meeting.
    """

    sink_module_id: int | None = None
    source_module_id: int | None = None
    #: What applications select.
    node_name: str = VIRTUAL_MIC_NAME
    #: Where WE write. This is the playback target, and it is not the same node.
    sink_name: str = VIRTUAL_SINK_NAME
    _owned: bool = dataclasses.field(default=False, repr=False)

    @staticmethod
    def find_existing() -> str | None:
        """Return the virtual source's node name if one is already loaded.

        A crash leaves both modules loaded -- they belong to the audio server, not
        to our process. Detecting them lets us reclaim rather than stack a second
        pair (AC-02.2, AC-02.3).
        """
        for dev in list_sources():
            if dev.key == VIRTUAL_MIC_NAME:
                return dev.key
        return None

    @staticmethod
    def _sink_exists() -> bool:
        return any(d.key == VIRTUAL_SINK_NAME for d in list_sinks())

    @classmethod
    def create(cls, *, reuse_existing: bool = True) -> VirtualMicrophone:
        if reuse_existing and cls.find_existing() and cls._sink_exists():
            # Adopt the orphan. We did not load these modules, so we must not
            # unload them on exit -- another instance may be using them.
            return cls(node_name=VIRTUAL_MIC_NAME, sink_name=VIRTUAL_SINK_NAME,
                       _owned=False)

        sink_id = cls._load_module(
            "module-null-sink",
            f"sink_name={VIRTUAL_SINK_NAME}",
            "channel_map=mono",
            f"sink_properties=device.description={VIRTUAL_SINK_NAME}",
        )
        try:
            source_id = cls._load_module(
                "module-remap-source",
                f"master={VIRTUAL_SINK_NAME}.monitor",
                f"source_name={VIRTUAL_MIC_NAME}",
                "channel_map=mono",
                # Pulse module arguments are whitespace-separated, so the
                # description cannot contain spaces here. The UI shows
                # VIRTUAL_MIC_DESCRIPTION; this is the raw node property.
                "source_properties=device.description=Vaani_Virtual_Microphone",
            )
        except VaaniError:
            _unload_quietly(sink_id)
            raise

        mic = cls(sink_module_id=sink_id, source_module_id=source_id, _owned=True)
        if not mic.exists():
            mic.destroy()
            raise VaaniError(
                code=ErrorCode.VIRTUAL_MIC_CREATE_FAILED,
                message="the virtual microphone loaded but did not appear as a source",
                severity=Severity.SESSION,
            )
        return mic

    @staticmethod
    def _load_module(module: str, *args: str) -> int:
        try:
            out = _pactl("load-module", module, *args)
        except VaaniError as exc:
            raise VaaniError(
                code=ErrorCode.VIRTUAL_MIC_CREATE_FAILED,
                message=f"could not create the virtual microphone ({module}): {exc.message}",
                severity=Severity.SESSION, cause=exc,
            ) from exc
        try:
            return int(out.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise VaaniError(
                code=ErrorCode.VIRTUAL_MIC_CREATE_FAILED,
                message=f"audio server returned an unexpected module id for {module}",
                severity=Severity.SESSION,
                detail={"output": out.strip()[:200]}, cause=exc,
            ) from exc

    def exists(self) -> bool:
        """Both halves must be present; either one alone carries no audio."""
        return (any(d.key == self.node_name for d in list_sources())
                and self._sink_exists())

    def destroy(self) -> None:
        """Unload both modules, but only if we loaded them. Idempotent."""
        if not self._owned:
            self.sink_module_id = self.source_module_id = None
            return
        # Source first: unloading the sink out from under the remap leaves the
        # source pointing at a master that no longer exists.
        _unload_quietly(self.source_module_id)
        _unload_quietly(self.sink_module_id)
        self.sink_module_id = self.source_module_id = None

    def __enter__(self) -> VirtualMicrophone:
        return self

    def __exit__(self, *exc) -> None:
        self.destroy()


def _unload_quietly(module_id: int | None) -> None:
    """The server may already have dropped the module; that is not worth surfacing."""
    if module_id is None:
        return
    try:
        _pactl("unload-module", str(module_id))
    except VaaniError:
        pass


def assert_no_feedback_loop(input_device_key: str, virtual_mic_name: str) -> None:
    """Refuse to run with the virtual mic as our own input (AC-02.5).

    Without this check the app would transcribe its own synthetic output, translate
    that, speak it again, and compound the error indefinitely.
    """
    if virtual_mic_name in input_device_key:
        raise VaaniError(
            code=ErrorCode.FEEDBACK_LOOP_DETECTED,
            message="the input device is Vaani's own virtual microphone; "
                    "select your real microphone instead",
            severity=Severity.FATAL,
            detail={"input_device": input_device_key},
        )


@dataclass(frozen=True, slots=True)
class VirtualMicConsumer:
    """An application currently reading from the virtual microphone."""

    app_name: str
    index: str


def virtual_mic_consumers(node_name: str = VIRTUAL_MIC_NAME) -> list[VirtualMicConsumer]:
    """Which applications are actually listening to the virtual microphone.

    This answers the question that silently ruins a call: the app runs, the
    device exists, translation works -- and the meeting client is still on the
    real microphone, so participants hear the untranslated original.

    Nothing in the pipeline can detect that on its own, because from Vaani's side
    everything looks healthy. The only evidence is whether anyone is reading the
    device.
    """
    try:
        index = next((d.key for d in list_sources() if d.key == node_name), None)
        if index is None:
            return []
        out = _pactl("list", "source-outputs")
    except VaaniError:
        return []

    consumers: list[VirtualMicConsumer] = []
    current_source = None
    current_index = ""
    source_ids = {str(i) for i in _source_indices(node_name)}
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Source Output #"):
            current_index = stripped.split("#")[-1]
            current_source = None
        elif stripped.startswith("Source:"):
            current_source = stripped.split(":", 1)[1].strip()
        elif "application.name" in stripped and current_source in source_ids:
            name = stripped.split("=", 1)[-1].strip().strip('"')
            if name and name.lower() != "vaani":
                consumers.append(VirtualMicConsumer(name, current_index))
    return consumers


def _source_indices(node_name: str) -> list[int]:
    try:
        out = _pactl("list", "short", "sources")
    except VaaniError:
        return []
    ids = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == node_name:
            try:
                ids.append(int(parts[0]))
            except ValueError:
                pass
    return ids
