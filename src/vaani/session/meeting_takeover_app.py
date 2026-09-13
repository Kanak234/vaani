"""End-to-end meeting takeover host.

This is the integration layer for the existing session and takeover runtime:

  user mic -> existing TranslationSession -> Vaani Virtual Microphone
  meeting source -> Whisper -> question detector -> takeover policy
                                     -> local answer LLM -> personal voice
                                     -> the same Vaani Virtual Microphone

The meeting source is a separate Pulse/PipeWire capture device (normally a
meeting application's monitor/loopback source). It is never routed into the
translation pipeline, preventing the assistant from translating its own output.
"""
from __future__ import annotations

import argparse
import threading
import time

from .engine import SessionConfig, TranslationSession
from .takeover import TakeoverConfig
from .takeover_runtime import MeetingTakeoverRuntime
from ..ai.assist import AnswerAssistant
from ..ai.stt.whisper import FasterWhisperRecognizer
from ..ai.tts.fallback import FallbackSynthesizer
from ..ai.translate.ollama import OllamaTranslator
from ..ai.vad.energy import EnergyVad
from ..audio.backend.pulse_backend import PulseCaptureStream
from ..audio.segmenter import SegmenterConfig, UtteranceSegmenter
from ..core.types import PerformanceMode


def looks_like_question(text: str) -> bool:
    """Conservative detector; statements are never handed to the answer model."""
    text = text.strip().lower()
    if not text:
        return False
    if "?" in text:
        return True
    starters = (
        "what ", "why ", "how ", "when ", "where ", "who ", "which ",
        "can you ", "could you ", "would you ", "do you ", "did you ",
        "are you ", "is there ", "have you ", "will you ",
    )
    return text.startswith(starters)


def looks_hesitant(text: str) -> bool:
    words = [w.strip(".,!?;:").lower() for w in text.split() if w]
    if not words or len(words) > 4:
        return False
    return any(w in {"uh", "um", "hmm", "maybe", "actually", "well", "so",
                       "i", "we", "yes", "no", "right", "okay", "ok"}
               for w in words)


class MeetingTakeoverApp:
    """Owns remote-audio capture while reusing the normal translation session."""

    def __init__(self, *, input_device: str | None, remote_input_device: str,
                 model: str = "small", llm_model: str = "qwen3:8b",
                 voice: str = "personal", voice_profile_id: str | None = None,
                 performance_mode: PerformanceMode = PerformanceMode.BALANCED) -> None:
        self.stop_event = threading.Event()
        self.session = None
        self._remote = None
        self._thread = None
        self._monitor_thread = None
        self._last_result_count = 0

        recognizer = FasterWhisperRecognizer(model_size=model, device="auto")
        recognizer.warmup()
        translator = OllamaTranslator(model=llm_model)
        translator.warmup()

        synthesizer = FallbackSynthesizer()
        profile_id = None
        if voice == "personal":
            from ..ai.tts.xtts import XttsSynthesizer
            from ..voice.consent import ConsentLedger
            from ..voice.enrollment import VoiceProfileStore
            store, ledger = VoiceProfileStore(), ConsentLedger()
            profile = store.active()
            if profile is None:
                raise RuntimeError("no active voice profile; run `vaani enroll record` first")
            if not ledger.has_active_consent():
                raise RuntimeError("voice consent is not active; run `vaani consent grant` first")
            synthesizer = XttsSynthesizer(profile_store=store, consent_ledger=ledger)
            synthesizer.warmup()
            profile_id = voice_profile_id or profile.id

        assistant = AnswerAssistant(translator=OllamaTranslator(model=llm_model))
        runtime = MeetingTakeoverRuntime(
            assistant=assistant,
            synthesizer=synthesizer,
            config=TakeoverConfig(),
            voice_profile_id=profile_id,
        )
        runtime.arm()

        self.session = TranslationSession(
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            vad=EnergyVad(),
            config=SessionConfig(
                input_device=input_device,
                performance_mode=performance_mode,
                voice_profile_id=profile_id,
                use_virtual_mic=True,
            ),
        )
        self.runtime = runtime
        self.remote_input_device = remote_input_device

    def start(self) -> None:
        self.session.start()
        self._remote = PulseCaptureStream(
            device=self.remote_input_device,
            sample_rate=16000,
            frame_ms=20,
            stream_name="takeover-remote-in",
        )
        self._thread = threading.Thread(target=self._remote_loop,
                                         name="vaani-takeover-remote", daemon=True)
        self._monitor_thread = threading.Thread(target=self._user_signal_loop,
                                                name="vaani-takeover-user-signals", daemon=True)
        self._thread.start()
        self._monitor_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.runtime.stop()
        if self._remote is not None:
            self._remote.close()
            self._remote = None
        for t in (self._thread, self._monitor_thread):
            if t is not None:
                t.join(timeout=2.0)
        self._thread = self._monitor_thread = None
        if self.session is not None:
            self.session.stop()

    def _remote_loop(self) -> None:
        recognizer = self.session.pipeline.recognizer
        segmenter = UtteranceSegmenter(
            config=SegmenterConfig.for_mode(self.session.config.performance_mode),
            sample_rate=16000, frame_ms=20,
        )
        vad = EnergyVad()
        while not self.stop_event.is_set():
            try:
                frame = self._remote.read_frame()
                segment = segmenter.push(frame, vad.is_speech(frame))
            except Exception:
                return
            if segment is None:
                continue
            try:
                transcript = recognizer.transcribe(segment.audio, 16000)
            except Exception:
                continue
            text = transcript.text.strip()
            if not looks_like_question(text):
                continue
            self.runtime.observe_remote_question(text)
            # Explicit takeover phrases are evaluated through the same policy.
            if any(p in text.lower() for p in self.runtime.controller.config.explicit_phrases):
                response = self.runtime.respond_if_authorized()
                self._emit_response(response)

    def _user_signal_loop(self) -> None:
        """Feed already-transcribed user utterances into the takeover policy."""
        while not self.stop_event.is_set():
            results = self.session.results
            if self._last_result_count < len(results):
                for result in results[self._last_result_count:]:
                    if result.transcript is not None:
                        text = result.transcript.text
                        self.runtime.observe_user_speech(
                            text, hesitation=looks_hesitant(text))
                self._last_result_count = len(results)
            response = self.runtime.respond_if_authorized()
            self._emit_response(response)
            time.sleep(0.1)

    def _emit_response(self, response) -> None:
        if response is None or response.audio is None:
            return
        if not self.session.sm.can_emit_audio or self.stop_event.is_set():
            return
        self.session._out_queue.put(response.audio.samples)
        print(f"[TAKEOVER] {response.question}")
        print(f"          -> {response.text}")
        print(f"          reason={response.reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vaani meeting takeover")
    parser.add_argument("--input", help="your physical microphone source")
    parser.add_argument("--remote-input", required=True,
                        help="meeting/remote-audio source; use `vaani devices --all`")
    parser.add_argument("--model", default="small")
    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--voice", choices=["personal", "fallback"], default="personal")
    parser.add_argument("--mode", choices=[m.value for m in PerformanceMode], default="balanced")
    args = parser.parse_args(argv)

    app = MeetingTakeoverApp(
        input_device=args.input,
        remote_input_device=args.remote_input,
        model=args.model,
        llm_model=args.llm_model,
        voice=args.voice,
        performance_mode=PerformanceMode(args.mode),
    )
    print("Vaani meeting takeover armed.")
    print("Explicit command: say 'Vaani, take over'.")
    print("Automatic handoff requires the configured hesitation/silence policy.")
    print("Press Ctrl+C to stop.")
    try:
        app.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
