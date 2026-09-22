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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

import os

if os.name == 'nt':
    from ..audio.backend.windows_backend import WindowsCaptureStream as CaptureStream, WindowsPlaybackStream as PlaybackStream
    from ..devices.windows_manager import VirtualMicrophone, assert_no_feedback_loop
else:
    from ..audio.backend.pulse_backend import PulseCaptureStream as CaptureStream, PulsePlaybackStream as PlaybackStream
    from ..devices.manager import VirtualMicrophone, assert_no_feedback_loop

from ..audio.segmenter import SegmenterConfig, UtteranceSegmenter
from ..context.engine import ContextConfig, ContextEngine
from ..core.errors import VaaniError
from ..core.gate import ConfidenceGate, GateConfig
from ..core.latency_monitor import LatencyMonitor
from ..core.pipeline import PipelineConfig, TranslationPipeline
from ..core.state_machine import SessionState, SessionStateMachine
from ..core.supervisor import CrashWindow, StageSupervisor
from .input_mode import InputMode, InputModeController
from ..system.logging import get_logger
from ..core.types import (
    PerformanceMode, SuppressionReason, Utterance, UtteranceResult,
)

SAMPLE_RATE = 16000
FRAME_MS = 20

# WHY a module logger: after a meeting goes wrong the user has only the
# log file to hand back. Until now the rotating handler existed but
# nothing in the session ever wrote to it, so a failed 30-minute run
# left no trace at all. These are the events worth reconstructing.
_log = get_logger("session")


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
    #: How many finished results stay in memory. WHY BOUNDED: every
    #: UtteranceResult holds its synthesised audio, so an unbounded list grew
    #: by megabytes per utterance and was the single largest memory leak in a
    #: long meeting. Nothing needs the full history -- consumers want counts
    #: (now kept as counters) and a rolling latency window, which AC-13.2 says
    #: should be rolling anyway.
    max_retained_results: int = 200
    #: Audio blocks waiting on the output thread. At ~1 utterance per block
    #: this is minutes of speech: only a genuinely stuck output thread fills it.
    max_queued_audio_blocks: int = 256
    #: How the user signals they want to be translated. ALWAYS_ON is the
    #: default because in a normal meeting the user simply talks; needing to
    #: hold a key to be understood would defeat the product.
    input_mode: InputMode = InputMode.ALWAYS_ON
    #: A stage failing this many times inside the window drops to fallback.
    stage_crash_limit: int = 3
    stage_crash_window_s: float = 60.0
    #: Minimum gap between "this stage is degraded" announcements, per stage.
    #: Without it a persistently failing stage floods the log and the overlay.
    stage_degrade_cooldown_s: float = 120.0
    gate: GateConfig = field(default_factory=GateConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


class TranslationSession:
    def __init__(self, *, recognizer, translator, synthesizer, vad,
                 config: SessionConfig | None = None,
                 on_result: Callable[[UtteranceResult], None] | None = None,
                 on_state: Callable[[SessionState, SessionState], None] | None = None,
                 on_latency_warning: Callable[[object], None] | None = None,
                 on_stage_degraded: Callable[[str], None] | None = None) -> None:
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
        self._out_queue: queue.Queue[np.ndarray | None] = queue.Queue(
            maxsize=self.config.max_queued_audio_blocks)
        # WHERE IT IS READ: _capture_loop consults this for every frame to
        # decide translate / pass raw voice through / drop.
        self.input_mode = InputModeController(self.config.input_mode)
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
        # Bounded on purpose - see SessionConfig.max_retained_results. A deque
        # with maxlen evicts the oldest entry automatically, so the memory
        # ceiling is fixed no matter how long the meeting runs.
        # WHERE IT GOES NEXT: latency_percentiles() reads it for the rolling
        # p50/p95, and new_results_since() feeds the takeover app.
        self.results: deque[UtteranceResult] = deque(
            maxlen=self.config.max_retained_results)
        #: Shed by backpressure: the worker was busy and the queue was full.
        self.dropped_utterances = 0
        #: Attempted but raised. See _worker_loop for why these are separate.
        self.failed_utterances = 0
        self.underruns = 0
        # Lifetime totals. These must NOT be derived from len(self.results) any
        # more: that list is now a window, so len() would silently stop
        # counting once a meeting passed max_retained_results utterances and
        # the end-of-session report would understate the session.
        self.total_utterances = 0
        self.total_suppressed = 0
        #: Monotonic sequence number, incremented once per finished result.
        #: Consumers track "what have I already seen" with this instead of a
        #: list index, which breaks as soon as the window starts evicting.
        self.results_seq = 0
        #: Audio blocks discarded because the output thread fell too far behind.
        self.dropped_audio_blocks = 0
        # Contains a crash in the worker loop so one bad utterance cannot end
        # the session. WHERE IT GOES NEXT: on_stage_failure/on_degraded are
        # surfaced by cli.py to the overlay banner.
        # Per-stage failure windows for MID-SESSION fallback.
        # WHY these are separate from `supervisor`: the supervisor catches
        # EXCEPTIONS out of the worker, but TranslationPipeline.process()
        # already catches stage exceptions internally and returns a suppressed
        # result instead. So a translation engine that fails every utterance
        # never raises, never trips the supervisor, and the meeting degrades
        # into silence with nothing swapping the broken provider out.
        # These windows watch SUPPRESSED RESULTS instead, which is the signal
        # that actually exists on that path.
        self._stage_windows: dict[str, CrashWindow] = {}
        #: When each stage last ANNOUNCED degradation, for the cooldown below.
        self._stage_announced: dict[str, float] = {}
        self._on_stage_degraded = on_stage_degraded
        self.supervisor = StageSupervisor(
            "pipeline",
            limit=self.config.stage_crash_limit,
            window_s=self.config.stage_crash_window_s,
            on_failure=self._report_stage_failure,
            on_exhausted=self._report_degraded,
        )

    def _enqueue_chunk(self, chunk) -> None:
        """Play a streamed chunk immediately.

        Checked against the state machine at the moment of delivery, so an
        emergency stop part-way through an utterance drops the remaining chunks
        instead of letting them play out.
        """
        if self.sm.can_emit_audio and not self._stop.is_set():
            self._put_audio(chunk)

    def inject_audio(self, samples: np.ndarray) -> None:
        """Public API for injecting synthetic audio (like takeover answers)."""
        if self.sm.can_emit_audio and not self._stop.is_set():
            self._put_audio(samples)

    def _put_audio(self, block: np.ndarray) -> None:
        """Queue audio for the output thread, dropping the OLDEST block if full.

        WHY BOUNDED AT ALL: this queue was unbounded. The output thread drains
        it at real time, but a synthesiser can produce faster than real time,
        so a long meeting could accumulate audio faster than it was spoken and
        grow without limit -- a leak that only shows up in exactly the 30+
        minute session Phase 1 has to survive.

        WHY DROP THE OLDEST: by the time this queue is minutes deep the
        conversation has moved on, and the same reasoning the module docstring
        applies to late utterances applies here -- speaking a stale sentence
        into a live meeting is worse than saying nothing. Dropping the newest
        instead would pin the user permanently behind.

        The honest cost: a drop can cut an utterance mid-word. That is why the
        bound is generous (minutes, not seconds) and why every drop is counted
        rather than hidden -- dropped_audio_blocks being non-zero means the
        machine could not keep up, and the user is told.
        """
        try:
            self._out_queue.put_nowait(block)
            return
        except queue.Full:
            pass
        try:
            self._out_queue.get_nowait()        # evict the stalest block
            self.dropped_audio_blocks += 1
        except queue.Empty:                     # drained by the output thread
            pass
        try:
            self._out_queue.put_nowait(block)
        except queue.Full:
            # Still full: the output thread is wedged, not merely slow. Count
            # it and move on rather than blocking the worker behind it.
            self.dropped_audio_blocks += 1

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        # The session id ties every later line in the log to this run, which
        # matters when a user has several meetings in one rotated file.
        _log.info("session %s starting (mode=%s, virtual_mic=%s)",
                  self.id, self.config.performance_mode.value,
                  self.config.use_virtual_mic)
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
        # The end-of-session counters are the first thing to look at when the
        # user says "it went quiet halfway through": dropped means too slow,
        # failed means something threw, underruns means the audio device.
        _log.info("session %s stopping: %d utterances, %d suppressed, "
                  "%d dropped, %d failed, %d underruns, %d audio blocks dropped",
                  self.id, self.total_utterances, self.total_suppressed,
                  self.dropped_utterances, self.failed_utterances,
                  self.underruns, self.dropped_audio_blocks)
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

    # ----------------------------------------------------------- input modes

    def set_bypass(self, enabled: bool) -> None:
        """Pass the user's real voice through, untranslated.

        WHY THE FLUSH: bypass exists for the moment the translation is wrong or
        the user needs to speak for themselves right now. If a half-spoken AI
        sentence kept playing underneath their real voice, the meeting would
        hear two people at once - the exact failure the input-mode invariant
        exists to prevent, arriving through the audio queue instead of through
        the state machine. So entering bypass drops everything already
        synthesised but not yet played, the same way emergency_stop does.

        WHERE THE DATA FLOWS NEXT: _capture_loop starts routing raw frames to
        _put_audio, so the virtual mic carries the microphone directly.
        """
        if enabled:
            self._flush_pending_audio()
        self.input_mode.set_bypass(enabled)
        _log.info("bypass %s", "ON (real voice)" if enabled else "OFF")

    def toggle_bypass(self) -> bool:
        self.set_bypass(not self.input_mode.bypass)
        return self.input_mode.bypass

    def set_input_mode(self, mode: InputMode) -> None:
        self.input_mode.set_mode(mode)
        _log.info("input mode -> %s", mode.value)

    def toggle_talk(self) -> bool:
        """Push-to-talk, toggle flavour. Returns whether the user is now live."""
        state = self.input_mode.toggle()
        # Releasing mid-sentence must not leave a fragment to be stitched on.
        if not state.translating_enabled:
            self.segmenter.reset()
        return state.translating_enabled

    def press_talk(self) -> None:
        self.input_mode.press()

    def release_talk(self) -> None:
        self.input_mode.release()
        self.segmenter.reset()

    def _flush_pending_audio(self) -> None:
        """Drop synthesised audio that has not been played yet."""
        while True:
            try:
                self._out_queue.get_nowait()
            except queue.Empty:
                break

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

            # One immutable snapshot per frame. Read once rather than querying
            # the controller three times, so a hotkey pressed mid-frame cannot
            # produce a frame that is half translated and half passed through.
            mode_state = self.input_mode.state()

            if mode_state.passthrough_enabled:
                # BYPASS: the meeting hears the user's ACTUAL voice. The frame
                # goes straight to the virtual mic without touching VAD, STT,
                # translation or TTS.
                # WHY reset the segmenter too: whatever was part-spoken when
                # bypass was pressed must not be stitched onto the next
                # translated sentence after bypass is released.
                self.segmenter.reset()
                self.vad.reset()
                self._put_audio(frame)
                continue

            if not mode_state.translating_enabled:
                # A push-to-talk mode with the key not held. Identical handling
                # to mute: nothing enters the pipeline, and the segmenter is
                # reset so releasing the key mid-sentence does not later emit a
                # fragment glued to the next press.
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

            # THE containment boundary. Everything below used to run bare: an
            # exception from the pipeline, or from any of the five follow-up
            # steps, killed this thread. Capture kept running and the virtual
            # mic kept emitting silence, so the session looked alive while
            # being permanently deaf -- the worst possible failure in a
            # meeting. Now one bad utterance costs one utterance.
            attempt = self.supervisor.run(self._process_one, utterance)
            if not attempt.ok:
                # Counted SEPARATELY from dropped_utterances on purpose. That
                # counter means "the worker was too slow and we shed load"; this
                # one means "the worker tried and something threw". Merging them
                # would make the end-of-session report unable to distinguish a
                # machine that is too slow from a stage that is broken.
                self.failed_utterances += 1
            if self.sm.can_transition(SessionState.LISTENING):
                self.sm.transition(SessionState.LISTENING)

    def _process_one(self, utterance: Utterance) -> None:
        """One utterance, end to end. Runs INSIDE the supervisor.

        Split out of _worker_loop so that the supervisor wraps the whole
        sequence -- pipeline, persistence, playback, telemetry, callbacks --
        and not just the pipeline call. A crash in any of them is contained
        identically. WHERE THE DATA FLOWS NEXT: audio to _out_queue and the
        virtual mic; the result to the on_result callback (overlay/CLI).
        """
        result = self.pipeline.process(utterance)
        self.results.append(result)
        # Counters advance for every utterance, including evicted ones.
        self.total_utterances += 1
        self.results_seq += 1
        if result.suppressed:
            self.total_suppressed += 1

        self._note_stage_outcome(result)

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

    # ------------------------------------------------- degradation reporting

    def _report_stage_failure(self, stage: str, exc: BaseException) -> None:
        """One contained failure. Routed to the same callback the latency
        warnings use, so the overlay needs no new plumbing."""
        # exc_info so the traceback lands in the log; the user only sees the
        # one-line banner, but the file keeps what is needed to diagnose it.
        _log.warning("stage %s failed: %s", stage, exc, exc_info=exc)
        if self._on_latency_warning is None:
            return
        try:
            self._on_latency_warning(
                f"{stage} failed ({type(exc).__name__}); utterance dropped")
        except Exception:
            pass

    def _report_degraded(self, stage: str) -> None:
        """The crash window tripped: the stage is broken now, not unlucky."""
        _log.error("stage %s failed %d times in %.0fs - entering degraded mode",
                   stage, self.config.stage_crash_limit,
                   self.config.stage_crash_window_s)
        if self._on_latency_warning is None:
            return
        try:
            self._on_latency_warning(
                f"{stage} failed {self.config.stage_crash_limit}x in "
                f"{self.config.stage_crash_window_s:.0f}s - degraded mode")
        except Exception:
            pass

    #: Error codes that mean "there was nothing to do", NOT "the provider is
    #: broken". Found by the 30-minute soak: an idle microphone picks up room
    #: noise, the VAD segments it, Whisper correctly reports STT_EMPTY, and the
    #: pipeline turns that into a STAGE_FAILURE suppression. Counting those as
    #: failures made a SILENT ROOM look like a broken speech engine - the soak
    #: log filled with "stage stt failed 3 times in 60s" while nothing was
    #: wrong at all. In a real meeting with any background noise, Vaani would
    #: have declared STT degraded within the first minute.
    BENIGN_ERROR_CODES = frozenset({"STT_EMPTY", "TRANSLATION_EMPTY"})

    def failing_stage(self, result: UtteranceResult) -> str | None:
        """Which stage failed, for a result suppressed by a stage failure.

        Read off the LAST recorded timing: pipeline._timed() records a stage's
        duration on the failure path as well as the success path, so the final
        entry is the stage that was running when it broke. Returns None for any
        other kind of suppression - a low-confidence transcript is the gate
        working correctly, not a broken provider, and must never cause a swap.
        """
        if not result.suppressed:
            return None
        if result.suppression_reason is not SuppressionReason.STAGE_FAILURE:
            return None
        if result.error_code in self.BENIGN_ERROR_CODES:
            # Nothing to do is not a failure. See BENIGN_ERROR_CODES.
            return None
        if not result.timings:
            return None
        return result.timings[-1].stage

    def _note_stage_outcome(self, result: UtteranceResult) -> None:
        """Track per-stage failures and announce a stage that is now broken.

        WHERE IT GOES NEXT: on_stage_degraded(stage) is handled by cli.py,
        which swaps the provider for that stage and shows an overlay toast.
        This method decides WHEN; it deliberately does not decide WHAT to swap
        to, because the session has no business knowing the provider ladder.
        """
        stage = self.failing_stage(result)
        if stage is None:
            # A good utterance clears every window: whatever was wobbling is
            # demonstrably working now, so old failures must not accumulate
            # across a long meeting and trip on an unrelated later hiccup.
            if not result.suppressed:
                for window in self._stage_windows.values():
                    window.record_success()
            return

        window = self._stage_windows.get(stage)
        if window is None:
            window = CrashWindow(limit=self.config.stage_crash_limit,
                                 window_s=self.config.stage_crash_window_s)
            self._stage_windows[stage] = window

        if not window.record_failure():
            return

        # COOLDOWN. Without it this fires every few seconds for as long as the
        # stage keeps failing, because the window is reset on each trip. The
        # 30-minute soak produced 20 identical ERROR lines in four minutes,
        # which buries anything else in the log and tells the user nothing new.
        now = time.monotonic()
        last = self._stage_announced.get(stage, 0.0)
        if now - last < self.config.stage_degrade_cooldown_s:
            window.record_success()
            return
        self._stage_announced[stage] = now

        _log.error("stage %s failed %d times in %.0fs - degraded",
                   stage, self.config.stage_crash_limit,
                   self.config.stage_crash_window_s)
        # Reset immediately so the REPLACEMENT provider gets its own full
        # allowance rather than inheriting the dead one's failures and being
        # swapped out on its first stumble.
        window.record_success()
        if self._on_stage_degraded is not None:
            try:
                self._on_stage_degraded(stage)
            except Exception:
                # A failing swap must not take the meeting with it; the stage
                # keeps failing and the user keeps seeing suppressions, which
                # is strictly better than a dead session.
                _log.exception("fallback handler for stage %s raised", stage)

    def swap_translator(self, translator, *, reason: str) -> None:
        """Replace the translation engine mid-session.

        Safe to call from the control thread while the worker is running: the
        pipeline reads `self.translator` per utterance, and a single attribute
        assignment is atomic in CPython, so the worst case is one utterance
        translated by the outgoing engine.
        """
        self.pipeline.translator = translator
        _log.warning("translation engine swapped: %s", reason)
        if self._on_latency_warning is not None:
            try:
                self._on_latency_warning(f"translation fallback: {reason}")
            except Exception:
                pass

    def new_results_since(self, seq: int) -> tuple[list[UtteranceResult], int]:
        """Results finished since sequence ``seq``, plus the new sequence.

        WHY THIS EXISTS: consumers used to slice ``session.results`` by index.
        That was already fragile and became wrong once the list turned into a
        bounded window -- after eviction, index N is no longer the Nth
        utterance, so a consumer would silently re-read or skip results.
        Sequence numbers survive eviction.

        If the caller has fallen further behind than the retained window, it
        gets what is still there; the gap is visible as the jump in the
        returned sequence rather than being papered over.
        """
        pending = self.results_seq - seq
        if pending <= 0:
            return [], self.results_seq
        items = list(self.results)[-pending:] if pending <= len(self.results) \
            else list(self.results)
        return items, self.results_seq

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
