"""End-to-end meeting takeover host for Linux and Windows.

Windows topology:
  user mic -> Vaani translation -> virtual-cable playback -> meeting mic input
  meeting speaker loopback -> Whisper -> question policy -> local answer LLM
  -> personal voice -> the same virtual-cable playback

Vaani does not install a Windows kernel audio driver. A virtual audio cable is
therefore an explicit endpoint selected with ``--output-device``.
"""
from __future__ import annotations

import argparse
import os
import threading
import time

from .engine import SessionConfig, TranslationSession
from .takeover import TakeoverConfig
from .takeover_runtime import MeetingTakeoverRuntime
# The remote-capture loop and its two heuristics now live in `remote_listener`
# so that `vaani meeting --takeover` can use the same code instead of a second
# copy that drifts (docs/15-MEETING-UNIFICATION.md §0). They are re-exported
# here because tests and the Windows runtime already import them from this
# module, and breaking those imports would be a pointless churn.
from .remote_listener import (  # noqa: F401  (re-exported for existing callers)
    RemoteQuestionListener,
    looks_hesitant,
    looks_like_question,
)
from ..ai.assist import AnswerAssistant
from ..ai.stt.whisper import FasterWhisperRecognizer
from ..ai.tts.fallback import FallbackSynthesizer
from ..ai.translate.ollama import OllamaTranslator
from ..ai.vad.energy import EnergyVad
from ..core.types import PerformanceMode


class MeetingTakeoverApp:
    """Own remote-audio capture while reusing Vaani's normal pipeline."""

    def __init__(self, *, input_device: str | None, remote_input_device: str,
                 output_device: str | None = None, model: str = "small",
                 llm_model: str = "qwen3:8b", voice: str = "personal",
                 voice_profile_id: str | None = None,
                 performance_mode: PerformanceMode = PerformanceMode.BALANCED,
                 ollama_host: str | None = None,
                 stt_device: str = "auto",
                 stt_compute_type: str | None = None) -> None:
        self.stop_event = threading.Event()
        self.session = None
        self._remote = None
        self._thread = None
        self._monitor_thread = None
        self._last_result_count = 0
        self.output_device = output_device

        recognizer = FasterWhisperRecognizer(
            model_size=model, device=stt_device, compute_type=stt_compute_type)
        recognizer.warmup()
        translator_kwargs = {"model": llm_model}
        if ollama_host:
            translator_kwargs["host"] = ollama_host
        translator = OllamaTranslator(**translator_kwargs)
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

        assistant = AnswerAssistant(translator=translator)
        runtime = MeetingTakeoverRuntime(
            assistant=assistant, synthesizer=synthesizer,
            config=TakeoverConfig(), voice_profile_id=profile_id)
        runtime.arm()

        if os.name == "nt":
            from .windows_session import WindowsSessionConfig, WindowsTranslationSession
            if not output_device:
                raise RuntimeError("Windows meeting mode requires --output-device")
            session_config = WindowsSessionConfig(
                input_device=input_device,
                virtual_output_device=output_device,
                performance_mode=performance_mode,
                voice_profile_id=profile_id,
                use_virtual_mic=True,
            )
            self.session = WindowsTranslationSession(
                recognizer=recognizer, translator=translator,
                synthesizer=synthesizer, vad=EnergyVad(), config=session_config)
        else:
            self.session = TranslationSession(
                recognizer=recognizer, translator=translator,
                synthesizer=synthesizer, vad=EnergyVad(),
                config=SessionConfig(
                    input_device=input_device, performance_mode=performance_mode,
                    voice_profile_id=profile_id, use_virtual_mic=True))
        self.runtime = runtime
        self.remote_input_device = remote_input_device

    def start(self) -> None:
        try:
            self.session.start()
            # Shared implementation -- see remote_listener.py. It owns its own
            # stream and thread, so this class no longer manages either.
            self._remote = RemoteQuestionListener(
                device=self.remote_input_device,
                recognizer=self.session.pipeline.recognizer,
                on_question=self.runtime.observe_remote_question,
                performance_mode=self.session.config.performance_mode,
                on_error=lambda msg: print(f"[TAKEOVER] {msg}"))
            self._remote.start()
            self.stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._user_signal_loop,
                                                    name="vaani-takeover-user-signals", daemon=True)
            self._monitor_thread.start()
        except Exception:
            self.stop_event.set()
            if self._remote is not None:
                self._remote.stop()
                self._remote = None
            if self.session is not None:
                try:
                    self.session.stop()
                except Exception:
                    pass
            raise

    def stop(self) -> None:
        self.stop_event.set()
        self.runtime.stop()
        if self._remote is not None:
            self._remote.stop()
            self._remote = None
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
        self._thread = self._monitor_thread = None
        if self.session is not None:
            self.session.stop()

    def _user_signal_loop(self) -> None:
        while not self.stop_event.is_set():
            results = self.session.results
            if self._last_result_count < len(results):
                for result in results[self._last_result_count:]:
                    if result.transcript is not None:
                        text = result.transcript.text
                        self.runtime.observe_user_speech(
                            text, hesitation=looks_hesitant(text))
                        if any(p in text.lower() for p in self.runtime.controller.config.explicit_phrases):
                            self._emit_response(self.runtime.respond_if_authorized())
                self._last_result_count = len(results)
            self._emit_response(self.runtime.respond_if_authorized())
            time.sleep(0.1)

    def _emit_response(self, response) -> None:
        if response is None or response.audio is None:
            return
        if not self.session.sm.can_emit_audio or self.stop_event.is_set():
            return
        self.session.inject_audio(response.audio.samples)
        print(f"[TAKEOVER] {response.question}")
        print(f"          -> {response.text}")
        print(f"          reason={response.reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vaani meeting takeover")
    parser.add_argument("--input", help="your physical microphone device name")
    parser.add_argument("--remote-input", required=True,
                        help="meeting speaker loopback/remote-audio device name")
    parser.add_argument("--output-device",
                        help="Windows virtual-cable playback endpoint used as meeting mic feed")
    parser.add_argument("--model", default="small")
    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--voice", choices=["personal", "fallback"], default="personal")
    parser.add_argument("--mode", choices=[m.value for m in PerformanceMode], default="balanced")
    args = parser.parse_args(argv)

    app = MeetingTakeoverApp(
        input_device=args.input, remote_input_device=args.remote_input,
        output_device=args.output_device, model=args.model,
        llm_model=args.llm_model, voice=args.voice,
        performance_mode=PerformanceMode(args.mode))
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
