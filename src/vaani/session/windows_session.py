"""Windows TranslationSession adapter.

The core session remains platform-neutral; only device ownership changes. On
Windows, Vaani writes to a selected playback endpoint. For meeting use, that
endpoint should be the playback side of a virtual audio cable, whose recording
side is selected as the meeting microphone.
"""
from __future__ import annotations

from dataclasses import dataclass

from .engine import SessionConfig, TranslationSession, SAMPLE_RATE, FRAME_MS
from ..audio.backend.windows_backend import WindowsCaptureStream, WindowsPlaybackStream
from ..devices.windows_manager import assert_no_feedback_loop


@dataclass(slots=True)
class WindowsSessionConfig(SessionConfig):
    #: Windows playback endpoint used as the virtual microphone feed.
    virtual_output_device: str | None = None


class WindowsTranslationSession(TranslationSession):
    """TranslationSession with native Windows WASAPI I/O."""

    config: WindowsSessionConfig

    def __init__(self, *, recognizer, translator, synthesizer, vad,
                 config: WindowsSessionConfig | None = None,
                 on_result=None, on_state=None, on_latency_warning=None) -> None:
        super().__init__(recognizer=recognizer, translator=translator,
                         synthesizer=synthesizer, vad=vad,
                         config=config or WindowsSessionConfig(),
                         on_result=on_result, on_state=on_state,
                         on_latency_warning=on_latency_warning)

    def _open_devices(self) -> None:
        if self.config.use_virtual_mic:
            output = self.config.virtual_output_device
            if not output:
                raise RuntimeError(
                    "Windows meeting mode requires --output-device: select the "
                    "playback endpoint of a virtual audio cable")
            if self.config.input_device:
                assert_no_feedback_loop(self.config.input_device, output)
            self._sink = WindowsPlaybackStream(
                device=output, sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS,
                stream_name="virtual-mic-out", require_device=True)

        if self.config.monitor_device:
            self._monitor = WindowsPlaybackStream(
                device=self.config.monitor_device, sample_rate=SAMPLE_RATE,
                frame_ms=FRAME_MS, stream_name="monitor-out", require_device=True)

        self._capture = WindowsCaptureStream(
            device=self.config.input_device, sample_rate=SAMPLE_RATE,
            frame_ms=FRAME_MS, stream_name="mic-in")
