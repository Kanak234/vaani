"""The running session: threads, device lifecycle, and the audio path.

Threading model, and why it is this shape:

  capture thread   blocks on pa_simple_read, runs VAD, feeds the segmenter.
                   Does no AI work, so it can never be the reason a frame is
                   dropped -- a dropped frame is unrecoverable audio loss.
  worker thread    pulls closed utterances off a queue and runs the pipeline.
                   Slow by nature (STT + TTS); isolated so it cannot stall capture.
  output thread    owns the only handle to the virtual microphone and keeps it fed
                   with silence when idle, so meeting clients see a continuously
                   live device rather than one that stalls between utterances.

The queue between capture and worker is bounded. If the worker falls behind, new
utterances are DROPPED rather than queued: in a live conversation, a translation
that arrives 30 seconds late is worse than no translation, because the meeting has
moved on and the user has no idea which sentence it belongs to.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

import os

if os.name == 'nt':
    from ..audio.backend.windows_backend import WindowsCaptureStream as CaptureStream, WindowsPlaybackStream as PlaybackStream
    from ..devices.windows_manager import VirtualMicrophone, assert_no_feedback_loop, list_sources
else:
    from ..audio.backend.pulse_backend import PulseCaptureStream as CaptureStream, PulsePlaybackStream as PlaybackStream
    from ..devices.manager import VirtualMicrophone, assert_no_feedback_loop

from ..audio.segmenter import SegmenterConfig, UtteranceSegmenter
from ..context.engine import ContextConfig, ContextEngine
from ..core.errors import ErrorCode, Severity, VaaniError
from ..core.gate import ConfidenceGate, GateConfig
from ..core.latency_monitor import LatencyMonitor
from ..core.pipeline import PipelineConfig, TranslationPipeline
from ..core.state_machine import SessionState, SessionStateMachine
from ..core.types import PerformanceMode, Utterance, UtteranceResult

SAMPLE_RATE = 16000
FRAME_MS = 20


@dataclass(slots=True)
class SessionConfig:
    input_device: str | None = None
    #: None => write to the virtual microphone (meeting mode).
    #: A sink name => also monitor through speakers (conversation mode).
    monitor_device: str | None = None
    use_virtual_mic: bool = True
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    voice_profile_id: str | None = None
    local_only: bool = False
    #: Utterances waiting on the worker. Small on purpose -- see module docstring.
    max_queued_utterances: int = 3
    #: Off by default. When False, transcript text is never written to disk.
    persist_transcript: bool = False
    #: None disables persistence entirely (no database file is touched).
    database: object | None = None
    gate: GateConfig = field(default_factory=GateConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


class TranslationSession:
    def __init__(self, *, recognizer, translator, synthesizer, vad,
                 config: SessionConfig | None = None,
                 on_result: Callable[[UtteranceResult], None] | None = None,
                 on_state: Callable[[SessionState, SessionState], None] | None = None,
                 on_latency_warning: Callable[[object], None] | None = None) -> None:
        self.config = config or SessionConfig()
        self.id = uuid.uuid4().hex
        self.sm = SessionStateMachine()
        if on_state:
            self.sm.subscribe(on_state)

        self.vad = vad
        self.segmenter = UtteranceSegmenter(
            config=SegmenterConfig.for_mode(self.config.performance_mode),
            sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS)
        self.context = ContextEngine(self.id, self.config.context)
        self.pipeline = TranslationPipeline(
            recognizer=recognizer, translator=translator, synthesizer=synthesizer,
            context=self.context, gate=ConfidenceGate(self.config.gate),
            state_machine=self.sm, config=self.config.pipeline,
            voice_profile_id=self.config.voice_profile_id,
            on_audio_chunk=self._enqueue_chunk)

        self.latency_monitor = LatencyMonitor(mode=self.config.performance_mode)
        self._on_latency_warning = on_latency_warning
        self._on_result = on_result
        self._queue: queue.Queue[Utterance | None] = queue.Queue(
            maxsize=self.config.max_queued_utterances)
        self._out_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._muted = threading.Event()
        self._threads: list[threading.Thread] = []

        self._capture: CaptureStream | None = None
        self._virtual_mic: VirtualMicrophone | None = None
        self._sink: PlaybackStream | None = None
        self._monitor: PlaybackStream | None = None

        self._seq = 0
        self._db = self.config.database
        self._db_session_id: int | None = None
        self.results: list[UtteranceResult] = []
        self.dropped_utterances = 0
        self.underruns = 0

    def _enqueue_chunk(self, chunk) -> None:
        """Play a streamed chunk immediately.

        Checked against the state machine at the moment of delivery, so an
        emergency stop part-way through an utterance drops the remaining chunks
        instead of letting them play out.
        """
        if self.sm.can_emit_audio and not self._stop.is_set():
            self._out_queue.put(chunk)

    def inject_audio(self, samples: np.ndarray) -> None:
        """Public API for injecting synthetic audio (like takeover answers)."""
        if self.sm.can_emit_audio and not self._stop.is_set():
            self._out_queue.put(samples)

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.sm.transition(SessionState.INITIALIZING)
        try:
            self._open_devices()
        except VaaniError:
            self.sm.force(SessionState.FAILED)
            self._close_devices()
            raise

        if self._db is not None:
            try:
                self._db_session_id = self._db.start_session(
                    mode="meeting" if self.config.use_virtual_mic else "conversation",
                    performance_mode=self.config.performance_mode.value,
                    input_device_id=self.config.input_device,
                    output_device_id=self.config.monitor_device,
                    local_only=self.config.local_only,
                    persist_transcript=self.config.persist_transcript)
            except Exception:
                # Persistence is a convenience; never let it block a meeting.
                self._db_session_id = None

        self._stop.clear()
        for target, name in ((self._capture_loop, "vaani-capture"),
                             (self._worker_loop, "vaani-worker"),
                             (self._output_loop, "vaani-output")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        self.sm.transition(SessionState.LISTENING)

    def _open_devices(self) -> None:
        if self.config.use_virtual_mic:
            if os.name == 'nt':
                output = getattr(self.config, 'virtual_output_device', None)
                if not output:
                    raise RuntimeError("Windows meeting mode requires virtual_output_device")
                if self.config.input_device:
                    assert_no_feedback_loop(self.config.input_device, output)
                self._sink = PlaybackStream(
                    device=output, sample_rate=SAMPLE_RATE,
                    frame_ms=FRAME_MS, stream_name="virtual-mic-out",
                    require_device=True)
            else:
                self._virtual_mic = VirtualMicrophone.create()
                if self.config.input_device:
                    # Refuse to transcribe our own output (AC-02.5).
                    assert_no_feedback_loop(self.config.input_device,
                                            self._virtual_mic.node_name)
                # NOTE: we write to the SINK, not to the source apps read from.
                # Targeting the source name would open successfully and then play to
                # the default output instead -- see VirtualMicrophone's docstring.
                self._sink = PlaybackStream(
                    device=self._virtual_mic.sink_name, sample_rate=SAMPLE_RATE,
                    frame_ms=FRAME_MS, stream_name="virtual-mic-out",
                    require_device=True)

        if self.config.monitor_device:
            self._monitor = PlaybackStream(
                device=self.config.monitor_device, sample_rate=SAMPLE_RATE,
                frame_ms=FRAME_MS, stream_name="monitor-out",
                require_device=True)

        self._capture = CaptureStream(
            device=self.config.input_device, sample_rate=SAMPLE_RATE,
            frame_ms=FRAME_MS, stream_name="mic-in")

    def _close_devices(self) -> None:
        for stream in (self._capture, self._sink, self._monitor):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        self._capture = self._sink = self._monitor = None
        if self._virtual_mic is not None:
            self._virtual_mic.destroy()
            self._virtual_mic = None

    def stop(self, *, timeout: float = 5.0) -> None:
        if self.sm.state is not SessionState.IDLE:
            try:
                self.sm.transition(SessionState.STOPPING)
            except Exception:
                self.sm.force(SessionState.STOPPING)
        self._stop.set()
        self._queue.put(None)
        self._out_queue.put(None)
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
        self._close_devices()
        if self._db is not None and self._db_session_id is not None:
            try:
                self._db.end_session(self._db_session_id, reason="user_stop")
            except Exception:
                pass
        self.sm.force(SessionState.IDLE)

    def emergency_stop(self) -> None:
        """AC-12.2: halt within 200 ms, never leave a partial word playing."""
        self.sm.emergency_stop()
        self._stop.set()
        # Drop everything already synthesised but not yet played.
        while True:
            try:
                self._out_queue.get_nowait()
            except queue.Empty:
                break
        if self._sink is not None:
            try:
                self._sink.flush()
            except Exception:
                pass

    def pause(self) -> None:
        self._paused.set()
        if self.sm.can_transition(SessionState.PAUSED):
            self.sm.transition(SessionState.PAUSED)

    def resume(self) -> None:
        self._paused.clear()
        if self.sm.can_transition(SessionState.LISTENING):
            self.sm.transition(SessionState.LISTENING)

    def set_muted(self, muted: bool) -> None:
        """Mute stops audio entering the pipeline at all (AC-10.3)."""
        self._muted.set() if muted else self._muted.clear()

    # -------------------------------------------------------------- threads

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._capture.read_frame()
            except VaaniError:
                self.sm.force(SessionState.RECOVERING)
                return

            if self._muted.is_set() or self._paused.is_set():
                # Reset so a mute in mid-sentence does not later emit a fragment
                # stitched to whatever is said after unmuting.
                self.segmenter.reset()
                self.vad.reset()
                continue

            try:
                prob = self.vad.is_speech(frame)
            except VaaniError:
                prob = 0.0

            segment = self.segmenter.push(frame, prob)
            if segment is None:
                continue

            self._seq += 1
            utterance = Utterance(
                session_id=self.id, seq=self._seq, audio=segment.audio,
                sample_rate=segment.sample_rate,
                speech_end_time=segment.speech_end_time,
                speech_ms=segment.speech_ms)
            try:
                self._queue.put_nowait(utterance)
            except queue.Full:
                # Better a visible gap than a translation that lands after the
                # conversation has moved on.
                self.dropped_utterances += 1

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                utterance = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if utterance is None:
                return

            result = self.pipeline.process(utterance)
            self.results.append(result)

            if self._db is not None and self._db_session_id is not None:
                try:
                    self._db.record_utterance(self._db_session_id, result)
                except Exception:
                    pass    # a storage failure must not drop the utterance

            # With a streaming synthesiser the chunks were already queued as they
            # were produced; queueing the assembled audio again would play the
            # whole utterance twice.
            already_streamed = any(t.stage == "tts_first_chunk"
                                   for t in result.timings)
            if (result.audio is not None and not result.suppressed
                    and not already_streamed):
                # The invariant, checked at the last possible moment: an emergency
                # stop between synthesis and playback must still win.
                if self.sm.can_emit_audio:
                    self.inject_audio(result.audio.samples)
            warning = self.latency_monitor.record(result)
            if warning is not None and self._on_latency_warning:
                try:
                    self._on_latency_warning(warning)
                except Exception:
                    pass

            if self._on_result:
                try:
                    self._on_result(result)
                except Exception:
                    pass
            if self.sm.can_transition(SessionState.LISTENING):
                self.sm.transition(SessionState.LISTENING)

    def _output_loop(self) -> None:
        """Keep the virtual microphone continuously fed.

        Writing silence while idle matters: a device that stops producing samples
        looks broken to a meeting client, and some clients will drop it. A steady
        stream of silence looks like a live microphone in a quiet room.
        """
        frame_n = SAMPLE_RATE * FRAME_MS // 1000
        silence = np.zeros(frame_n, dtype=np.float32)
        while not self._stop.is_set():
            try:
                chunk = self._out_queue.get(timeout=FRAME_MS / 1000.0)
            except queue.Empty:
                self._write(silence)
                continue
            if chunk is None:
                return
            for i in range(0, len(chunk), frame_n):
                if self._stop.is_set():
                    return
                block = chunk[i:i + frame_n]
                if block.size < frame_n:
                    block = np.pad(block, (0, frame_n - block.size))
                self._write(block)

    def _write(self, block: np.ndarray) -> None:
        for stream in (self._sink, self._monitor):
            if stream is None:
                continue
            try:
                stream.write(block)
            except VaaniError:
                self.underruns += 1

    # --------------------------------------------------------------- stats

    def latency_percentiles(self) -> dict[str, float]:
        """Measured end-to-end latency. Returns {} when nothing has run yet."""
        values = sorted(r.total_latency_ms for r in self.results
                        if r.total_latency_ms is not None and not r.suppressed)
        if not values:
            return {}
        def pct(p: float) -> float:
            idx = min(len(values) - 1, int(round((len(values) - 1) * p)))
            return round(values[idx], 1)
        return {"count": len(values), "p50": pct(0.5), "p95": pct(0.95),
                "min": round(values[0], 1), "max": round(values[-1], 1)}
