"""Move other applications' microphone input onto Vaani's virtual mic.

WHY: the previous design required the user to open Zoom/WhatsApp/Meet and change
the microphone by hand. That step is easy to forget, and when it is forgotten the
failure is completely silent -- Vaani reports healthy, translation runs, and the
other side simply hears the untranslated original. That happened on a real call.

PipeWire lets a capture stream be reassigned to a different source after it has
started, so the app never needs to be touched:

    pactl move-source-output <stream> VaaniVirtualMic

THE DANGEROUS CASE, guarded below: Vaani's own capture stream must never be moved
onto Vaani's own virtual microphone. That would feed synthesised output straight
back into recognition -- it would transcribe its own translation, translate that,
speak it again, and compound without limit. `_is_vaani_stream` exists solely to
make that impossible, and it errs toward excluding a stream when unsure.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..core.errors import ErrorCode, Severity, VaaniError
from .manager import VIRTUAL_MIC_NAME, VIRTUAL_SINK_NAME, _pactl

#: Streams belonging to Vaani itself, or to the audio plumbing it creates.
_OWN_MARKERS = ("vaani", VIRTUAL_MIC_NAME.lower(), VIRTUAL_SINK_NAME.lower(),
                "remap-source", "loopback")

#: Applications that capture audio but are not meeting clients. Moving these
#: would be surprising and is never what the user meant.
_NEVER_MOVE = ("pulseaudio volume control", "pavucontrol", "speech-dispatcher",
               "parec", "pw-record", "audacity")


@dataclass(frozen=True, slots=True)
class CaptureStream:
    index: str
    source_index: str
    app_name: str
    media_name: str
    is_own: bool
    on_virtual_mic: bool

    @property
    def movable(self) -> bool:
        if self.is_own or self.on_virtual_mic:
            return False
        return self.app_name.lower() not in _NEVER_MOVE

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.app_name} (#{self.index})"


def _virtual_mic_source_ids() -> set[str]:
    ids: set[str] = set()
    try:
        out = _pactl("list", "short", "sources")
    except VaaniError:
        return ids
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == VIRTUAL_MIC_NAME:
            ids.add(parts[0])
    return ids


def _is_vaani_stream(props: dict[str, str]) -> bool:
    """Conservative: any hint of Vaani's own plumbing counts as ours."""
    blob = " ".join(f"{k}={v}" for k, v in props.items()).lower()
    return any(marker in blob for marker in _OWN_MARKERS)


def list_capture_streams() -> list[CaptureStream]:
    """Every application currently capturing from a microphone."""
    try:
        out = _pactl("list", "source-outputs")
    except VaaniError:
        return []

    virtual_ids = _virtual_mic_source_ids()
    streams: list[CaptureStream] = []
    index = source = ""
    props: dict[str, str] = {}

    def flush() -> None:
        if not index:
            return
        app = props.get("application.name") or props.get("node.name") or "unknown"
        streams.append(CaptureStream(
            index=index, source_index=source, app_name=app,
            media_name=props.get("media.name", ""),
            is_own=_is_vaani_stream(props),
            on_virtual_mic=source in virtual_ids,
        ))

    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Source Output #"):
            flush()
            index, source, props = stripped.split("#")[-1], "", {}
        elif stripped.startswith("Source:"):
            source = stripped.split(":", 1)[1].strip()
        elif "=" in stripped and index:
            key, _, value = stripped.partition("=")
            props[key.strip()] = value.strip().strip('"')
    flush()
    return streams


def movable_streams() -> list[CaptureStream]:
    return [s for s in list_capture_streams() if s.movable]


def streams_on_virtual_mic() -> list[CaptureStream]:
    return [s for s in list_capture_streams()
            if s.on_virtual_mic and not s.is_own]


def move_to_virtual_mic(stream: CaptureStream) -> None:
    """Reassign one application's microphone to Vaani's virtual mic."""
    if stream.is_own:
        raise VaaniError(
            code=ErrorCode.FEEDBACK_LOOP_DETECTED,
            message="refusing to route Vaani's own capture into its own virtual "
                    "microphone; that would loop its output back into recognition",
            severity=Severity.FATAL,
            detail={"stream": stream.index, "app": stream.app_name},
        )
    try:
        _pactl("move-source-output", stream.index, VIRTUAL_MIC_NAME)
    except VaaniError as exc:
        raise VaaniError(
            code=ErrorCode.VIRTUAL_MIC_UNAVAILABLE,
            message=f"could not switch {stream.app_name} to the virtual microphone: "
                    f"{exc.message}",
            severity=Severity.SESSION, cause=exc,
        ) from exc


def move_back(stream: CaptureStream, source_name: str) -> None:
    """Return an application to a real microphone."""
    _pactl("move-source-output", stream.index, source_name)


def _state_path() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/vaani-{os.getuid()}"
    return Path(base) / "vaani-routing.json"


def reclaim_orphaned_routing(fallback_source: str | None = None) -> int:
    """Undo routing left behind by a crash. Call at EVERY startup.

    This exists because of a real incident. `AutoRouter` moves meeting apps onto
    the virtual microphone and moves them back on exit -- but a crash or
    `kill -9` skips the restore. The virtual mic then disappears with the
    process, and every app that had been moved is left pointing at a device that
    no longer carries anything.

    The user's symptom is not "Vaani is broken". It is "my microphone stopped
    working", system-wide, with no visible connection to this program. That is
    the worst kind of failure to leave behind, so recovery cannot depend on a
    clean shutdown.
    """
    path = _state_path()
    if not path.exists():
        return 0
    try:
        saved = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return 0

    target = fallback_source or saved.get("real_source") or _default_source()
    restored = 0
    for index, original in (saved.get("moved") or {}).items():
        for candidate in (target, original):
            if not candidate:
                continue
            try:
                _pactl("move-source-output", str(index), str(candidate))
                restored += 1
                break
            except VaaniError:
                continue
    path.unlink(missing_ok=True)
    return restored


def _default_source() -> str | None:
    try:
        return _pactl("get-default-source").strip() or None
    except VaaniError:
        return None


class AutoRouter:
    """Keeps meeting applications pointed at the virtual microphone.

    Polled rather than event-driven: a meeting client typically opens its capture
    stream when the call starts, which is *after* Vaani is running, so a one-shot
    sweep at startup would miss it. Polling every few seconds catches the app
    whenever it appears, which is the case that actually matters.
    """

    def __init__(self, *, real_source: str | None = None) -> None:
        self.real_source = real_source or _default_source()
        #: index -> original source index, so exit can put things back.
        self.moved: dict[str, str] = {}
        self.last_error: str | None = None
        # Recovery must survive a kill -9, so what has been moved is written to
        # disk as it happens rather than held only in memory.
        reclaim_orphaned_routing(self.real_source)

    def _persist(self) -> None:
        try:
            path = _state_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"real_source": self.real_source, "moved": self.moved}))
        except OSError:
            pass

    def sweep(self) -> list[CaptureStream]:
        """Move any newly-appeared capture stream. Returns what was moved."""
        moved_now: list[CaptureStream] = []
        for stream in list_capture_streams():
            if not stream.movable or stream.index in self.moved:
                continue
            try:
                move_to_virtual_mic(stream)
            except VaaniError as exc:
                self.last_error = exc.message
                continue
            self.moved[stream.index] = stream.source_index
            moved_now.append(stream)
        if moved_now:
            self._persist()
        return moved_now

    def restore_all(self, fallback_source: str | None = None) -> int:
        """Put every moved application back where it was.

        Called on exit. Without it a meeting client would be left pointing at a
        virtual microphone that no longer carries anything, and the user would
        appear muted in their next call with no obvious cause.
        """
        target = fallback_source or self.real_source
        restored = 0
        for index, original in list(self.moved.items()):
            try:
                _pactl("move-source-output", index, target or original)
                restored += 1
            except VaaniError:
                pass
            self.moved.pop(index, None)
        self._persist()
        _state_path().unlink(missing_ok=True)
        return restored
