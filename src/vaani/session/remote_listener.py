"""Listen to the OTHER side of the call, and feed questions to the takeover policy.

WHY THIS MODULE EXISTS
----------------------
Takeover needs to hear the other participants. Everything else in Vaani listens
to the user's microphone; this is the only component that listens to what the
meeting is *playing*. On Linux that is a PipeWire **monitor** source (the
loopback of your output sink); on Windows it is a WASAPI loopback capture.

The loop used to live inside `MeetingTakeoverApp._remote_loop`, which made it
unreachable from `vaani meeting` -- that command had the overlay, the control
socket and the auto-routing, but could not hear the other side, so it could
never do takeover. Pulling the loop out here is what lets BOTH entry points have
it without duplicating the code (docs/15-MEETING-UNIFICATION.md §0):

    cli.py  `vaani meeting --takeover`  -> RemoteQuestionListener
    session/meeting_takeover_app.py     -> RemoteQuestionListener

FLOW
----
    monitor source -> frames -> VAD -> segmenter -> Whisper
      -> looks_like_question()? -> MeetingTakeoverRuntime.observe_remote_question()
        -> (policy decides) -> answer drafted, synthesised, injected

This module deliberately does NOT decide whether to speak. It only reports what
the other side asked. The authorisation boundary stays in
`session/takeover.py`, which is what keeps that policy unit-testable.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable

from ..audio.segmenter import SegmenterConfig, UtteranceSegmenter
from ..core.types import PerformanceMode

#: Must match the capture rate used everywhere else in the pipeline; Whisper and
#: the segmenter config are both built around 16 kHz / 20 ms frames.
SAMPLE_RATE = 16000
FRAME_MS = 20


def looks_like_question(text: str) -> bool:
    """Cheap heuristic for 'the other side just asked me something'.

    Deliberately simple and deliberately WIDE. A false positive costs nothing --
    the takeover policy still requires arming plus silence plus hesitation before
    anything is said, so a mis-detected question just sits as a pending turn that
    expires. A false negative, by contrast, means Vaani stays silent in exactly
    the moment it was armed for. So the bias is toward over-detecting.
    """
    text = text.strip().lower()
    if not text:
        return False
    if "?" in text:
        return True
    starters = (
        "what ", "why ", "how ", "when ", "where ", "who ", "which ",
        "can you ", "could you ", "would you ", "do you ", "did you ",
        "are you ", "is there ", "have you ", "will you ", "tell us ",
        "tell me ", "walk us ", "walk me ",
    )
    return text.startswith(starters)


def looks_hesitant(text: str) -> bool:
    """True for the short filler fragments that mean 'I am stuck'.

    Only short fragments count: `max 4 words`. A long sentence containing "well"
    is someone answering, not someone struggling, and treating it as hesitation
    would let Vaani interrupt a person who is mid-thought.
    """
    words = [w.strip(".,!?;:").lower() for w in text.split() if w]
    if not words or len(words) > 4:
        return False
    return any(w in {"uh", "um", "hmm", "maybe", "actually", "well", "so",
                     "i", "we", "yes", "no", "right", "okay", "ok"}
               for w in words)


class RemoteQuestionListener:
    """Transcribes the remote party on its own thread and reports questions.

    Owns exactly one capture stream and one thread. It shares the session's
    recognizer rather than loading a second Whisper: a second model would double
    VRAM on a 4 GiB card, and `docs/09` already records that STT + TTS together
    barely fit.
    """

    def __init__(self, *, device: str, recognizer,
                 on_question: Callable[[str], None],
                 performance_mode: PerformanceMode = PerformanceMode.BALANCED,
                 vad=None,
                 on_error: Callable[[str], None] | None = None) -> None:
        self.device = device
        self.recognizer = recognizer
        self.on_question = on_question
        self.on_error = on_error
        self.performance_mode = performance_mode
        # EnergyVad by default: this stream is meeting audio played through a
        # sink, so it is already clean and loud. Silero's extra accuracy is not
        # worth a second ONNX session competing for the same CPU as the user's
        # own VAD.
        if vad is None:
            from ..ai.vad.energy import EnergyVad
            vad = EnergyVad()
        self.vad = vad

        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        #: Counts for the verify script and `status`, so "takeover never fired"
        #: can be diagnosed as "heard nothing" vs "heard, but never a question".
        self.utterances_heard = 0
        self.questions_seen = 0

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Open the monitor source and begin listening. Raises if it cannot."""
        if os.name == "nt":
            from ..audio.backend.windows_backend import WindowsCaptureStream
            self._stream = WindowsCaptureStream(
                device=self.device, sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS,
                stream_name="takeover-remote-in", include_loopback=True)
        else:
            from ..audio.backend.pulse_backend import PulseCaptureStream
            self._stream = PulseCaptureStream(
                device=self.device, sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS,
                stream_name="takeover-remote-in")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="vaani-takeover-remote")
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -------------------------------------------------------------- thread

    def _loop(self) -> None:
        segmenter = UtteranceSegmenter(
            config=SegmenterConfig.for_mode(self.performance_mode),
            sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS)
        while not self._stop.is_set():
            try:
                frame = self._stream.read_frame()
                segment = segmenter.push(frame, self.vad.is_speech(frame))
            except Exception as exc:
                # E-3: the remote stream dying must not take the meeting with it.
                # Translation keeps running; takeover degrades to explicit-phrase
                # only, which is still useful and is stated rather than silent.
                if not self._stop.is_set() and self.on_error:
                    self.on_error(f"remote audio stopped: {exc}")
                return
            if segment is None:
                continue
            try:
                transcript = self.recognizer.transcribe(segment.audio, SAMPLE_RATE)
            except Exception as exc:
                if not self._stop.is_set() and self.on_error:
                    self.on_error(f"remote STT error: {exc}")
                continue
            text = transcript.text.strip()
            if not text:
                continue
            self.utterances_heard += 1
            if not looks_like_question(text):
                continue
            self.questions_seen += 1
            try:
                self.on_question(text)
            except Exception as exc:
                if self.on_error:
                    self.on_error(f"question handler failed: {exc}")
