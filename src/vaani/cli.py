"""Command-line entry point.

The GUI described in the UI/UX brief is not implemented in this MVP. This CLI
exposes the same capabilities so the engine can be exercised and measured now,
and so every claim in the docs is checkable by running something.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

def _get_user_id() -> str:
    if os.name == 'nt':
        return os.environ.get('USERNAME', 'user')
    return str(os.getuid())

from .core.types import PerformanceMode


def cmd_devices(args) -> int:
    from .devices.manager import list_sinks, list_sources
    print("CAPTURE DEVICES (use with --input)")
    for d in list_sources(include_monitors=args.all):
        tag = " [VIRTUAL]" if d.is_virtual else ""
        print(f"  {d.display_name}{tag}")
        print(f"      key: {d.key}")
    print("\nPLAYBACK DEVICES (use with --monitor)")
    for d in list_sinks():
        print(f"  {d.display_name}\n      key: {d.key}")
    return 0


def cmd_doctor(args) -> int:
    from .diagnostics.checks import run_all, CheckResult
    print("=" * 72)
    print("VAANI DIAGNOSTICS")
    print("=" * 72)
    results = run_all()
    for r in results:
        print(r.line())
    failed = [r for r in results if r.status == "fail"]
    print("=" * 72)
    passed = sum(1 for r in results if r.ok)
    print(f"{passed}/{len(results)} passed"
          + (f", {len(failed)} FAILED" if failed else ""))
    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  - {r.name}: {r.detail}")
    return 1 if failed else 0


def cmd_virtualmic(args) -> int:
    from .devices.manager import VirtualMicrophone
    if args.action == "create":
        mic = VirtualMicrophone.create()
        print(f"Virtual microphone ready.")
        print(f"  Applications select : {mic.node_name}  ('Vaani Virtual Microphone')")
        print(f"  Vaani writes to     : {mic.sink_name}")
        print(f"  Owned by this call  : {mic._owned}")
        print("\nNOTE: it is unloaded when Vaani exits. Use `vaani run` to keep it up.")
        return 0
    if args.action == "status":
        existing = VirtualMicrophone.find_existing()
        print(f"Virtual microphone: {'present (' + existing + ')' if existing else 'not loaded'}")
        return 0
    # destroy
    if os.name == 'nt':
        print("Virtual mic management not available on Windows.")
        return 0
    mic = VirtualMicrophone(node_name="VaaniVirtualMic", sink_name="VaaniSink",
                            _owned=True)
    import subprocess
    out = subprocess.run(["pactl", "list", "short", "modules"],
                         capture_output=True, text=True).stdout
    removed = 0
    for line in out.splitlines():
        if "VaaniSink" in line or "VaaniVirtualMic" in line:
            subprocess.run(["pactl", "unload-module", line.split("\t")[0]],
                           capture_output=True)
            removed += 1
    print(f"Unloaded {removed} Vaani audio module(s).")
    return 0



def cmd_consent(args) -> int:
    from .voice.consent import (
        CONFIRMATION_PHRASE, CONSENT_TEXT, ConsentLedger,
    )
    from . import __version__
    ledger = ConsentLedger()

    if args.action == "status":
        grant = ledger.active_grant()
        if grant:
            print(f"Active consent: {grant.subject_label}  granted {grant.granted_at_utc}")
            print(f"  record id: {grant.id}")
        else:
            print("No active voice consent on record.")
        print(f"Ledger: {ledger.path}  ({len(ledger.all_records())} record(s))")
        return 0

    if args.action == "show":
        print(CONSENT_TEXT)
        return 0

    if args.action == "revoke":
        from .voice.enrollment import VoiceProfileStore
        store = VoiceProfileStore(ledger=ledger)
        if not ledger.has_active_consent():
            print("No active consent to revoke.")
            return 0
        removed = store.revoke_consent_and_delete_all(app_version=__version__)
        print(f"Consent revoked. Deleted {removed} voice profile(s).")
        return 0

    # grant
    print(CONSENT_TEXT)
    print("-" * 72)
    subject = args.subject or input("Whose voice is being enrolled? ").strip()
    print(f"\nType exactly {CONFIRMATION_PHRASE!r} to confirm (anything else aborts).")
    typed = args.confirm or input("> ").strip()
    try:
        record = ledger.grant(subject_label=subject, confirmation=typed,
                              app_version=__version__)
    except Exception as exc:
        print(f"Consent NOT recorded: {exc}", file=sys.stderr)
        return 1
    print(f"\nConsent recorded at {record.granted_at_utc} (id {record.id}).")
    print("You can revoke it at any time with:  vaani consent revoke")
    return 0


def cmd_enroll(args) -> int:
    """Record enrollment audio and build a voice profile."""
    import numpy as np
    from .audio.backend.pulse_backend import PulseCaptureStream
    from .voice.consent import ConsentLedger
    from .voice.enrollment import (
        ENROLLMENT_PROMPTS, VoiceProfileStore, analyse_quality,
    )
    from .ai.vad.energy import EnergyVad

    ledger = ConsentLedger()
    store = VoiceProfileStore(ledger=ledger)

    if not ledger.has_active_consent():
        print("Voice enrollment requires consent. Run:  vaani consent grant",
              file=sys.stderr)
        return 2

    if args.action == "list":
        profiles = store.list_profiles()
        if not profiles:
            print("No voice profiles.")
            return 0
        for p in profiles:
            print(f"  {p.id}  {p.name:<16} {p.status:<8} "
                  f"{p.speech_seconds:.0f}s  SNR {p.quality_snr_db:.1f}dB  "
                  f"{p.created_at_utc[:19]}")
        return 0

    if args.action == "delete":
        if not args.profile_id:
            print("Specify --profile-id (see `vaani enroll list`).", file=sys.stderr)
            return 2
        print(f"This permanently deletes the voice profile and its recordings.")
        if (args.yes or input("Type DELETE to confirm: ").strip()) not in ("DELETE", True):
            if not args.yes:
                print("Aborted.")
                return 1
        print("Deleted." if store.delete(args.profile_id) else "No such profile.")
        return 0

    # record
    print("=" * 72)
    print("VOICE ENROLLMENT")
    print("=" * 72)
    print(f"Read the prompts below aloud, naturally. Need >= 45s of speech.\n")
    for i, prompt in enumerate(ENROLLMENT_PROMPTS, 1):
        print(f"  {i}. {prompt}")
    print(f"\nRecording starts now for up to {args.seconds}s. Ctrl+C to stop early.\n")

    stream = PulseCaptureStream(device=args.input, sample_rate=16000,
                                frame_ms=20, stream_name="enroll")
    vad = EnergyVad()
    frames, mask = [], []
    try:
        n_frames = int(args.seconds * 1000 / 20)
        for i in range(n_frames):
            f = stream.read_frame()
            frames.append(f)
            mask.append(vad.is_speech(f) > 0.5)
            if i % 50 == 0:
                speech_s = sum(mask) * 0.02
                bar = "#" * min(40, int(speech_s))
                print(f"\r  speech: {speech_s:5.1f}s  {bar:<40}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n  stopped early.")
    finally:
        stream.close()

    audio = np.concatenate(frames) if frames else np.zeros(0, dtype="float32")
    sample_mask = np.repeat(np.array(mask, dtype=bool), 320)[: audio.size]
    print(f"\n\nRecorded {audio.size / 16000:.1f}s wall-clock.")

    report = analyse_quality(audio, 16000, sample_mask)
    print(f"Quality: {report.summary()}")
    if not report.passed:
        print("\nNot good enough for a voice profile:")
        for f in report.failures:
            print(f"  - {f}")
        return 1

    try:
        profile = store.create(name=args.name, audio=audio, sample_rate=16000,
                               provider_key="xtts_v2", model_id="xtts_v2",
                               speech_mask=sample_mask)
    except Exception as exc:
        print(f"\nEnrollment failed: {exc}", file=sys.stderr)
        return 1
    print(f"\nVoice profile created: {profile.id}")
    print(f"  reference audio: {profile.reference_audio_path}")
    return 0


def cmd_run(args) -> int:
    from .ai.tts.fallback import FallbackSynthesizer
    from .ai.translate.passthrough import PassthroughTranslator
    from .ai.vad.energy import EnergyVad
    from .ai.vad.silero import SileroVad
    from .core.state_machine import SessionState
    from .core.types import UtteranceResult
    from .session.engine import SessionConfig, TranslationSession

    # Decide model placement against LIVE free VRAM before loading anything.
    # Loading all three models onto a 4 GB laptop GPU succeeds when idle and OOMs
    # the moment a meeting client opens -- exactly when this product runs.
    from .core.model_budget import plan_placement
    stages = ["stt"]
    if args.translate != "none":
        stages.append("translation")
    if args.voice != "fallback":
        stages.append("tts")
    plan = plan_placement(stages)
    print(f"Model placement: {plan.summary()}")
    for note in plan.notes:
        print(f"  {note}")

    # STT: real Whisper if available, otherwise stop with a clear message rather
    # than pretending to translate.
    try:
        from .ai.stt.whisper import FasterWhisperRecognizer
        stt_device = (args.device if args.device != "auto"
                      else plan.device_for("stt"))
        recognizer = FasterWhisperRecognizer(model_size=args.model,
                                             device=stt_device)
        print(f"Loading Whisper '{args.model}' on {args.device}...")
        t0 = time.perf_counter()
        recognizer.warmup()
        print(f"  ready in {time.perf_counter() - t0:.1f}s "
              f"({recognizer.device_used}/{recognizer.compute_type_used})")
    except Exception as exc:
        print(f"ERROR: speech recognition unavailable: {exc}", file=sys.stderr)
        print("Install it with:  uv pip install faster-whisper", file=sys.stderr)
        return 2

    # Translation: real NMT if requested and available, else pass-through.
    translator = PassthroughTranslator()
    translation_note = "PASS-THROUGH ONLY - Hindi is NOT translated"
    if args.translate != "none":
        try:
            from .ai.translate.nllb import NllbTranslator
            nllb = NllbTranslator(
                device=args.device if args.device != "auto"
                else plan.device_for("translation"))
            if args.translate == "nllb":
                translator = nllb
                print("Loading translation model (NLLB-200 600M)...")
                t0 = time.perf_counter()
                translator.warmup()
                print(f"  ready in {time.perf_counter() - t0:.1f}s "
                      f"({translator.device_used}/{translator.compute_type_used})")
                translation_note = ("NLLB-200 local - romanised Hinglish is "
                                    "SUPPRESSED, no conversational context")
            else:
                from .ai.translate.ollama import OllamaTranslator
                from .ai.translate.router import RoutingTranslator
                translator = RoutingTranslator(
                    fast=nllb,
                    accurate=OllamaTranslator(model=args.llm_model))
                print(f"Loading translation (NLLB-200 + {args.llm_model} via Ollama)...")
                t0 = time.perf_counter()
                translator.warmup()
                degraded = translator.accurate is None
                print(f"  ready in {time.perf_counter() - t0:.1f}s"
                      + ("  [LLM unavailable - romanised Hinglish will be suppressed]"
                         if degraded else ""))
                translation_note = ("routing: NLLB for Devanagari/English, LLM for "
                                    "romanised Hinglish"
                                    + (" (LLM DOWN)" if degraded else ""))
        except Exception as exc:
            print(f"  translation unavailable ({exc}); falling back to pass-through",
                  file=sys.stderr)

    # Voice: the user's enrolled voice if one exists and consent stands.
    synthesizer = FallbackSynthesizer()
    voice_note = "FALLBACK SYNTHESISER - this is NOT your voice"
    voice_profile_id = None
    if args.voice != "fallback":
        try:
            from .ai.tts.xtts import XttsSynthesizer
            from .voice.consent import ConsentLedger
            from .voice.enrollment import VoiceProfileStore
            store, ledger = VoiceProfileStore(), ConsentLedger()
            profile = store.active()
            if profile is None:
                print("  no voice profile enrolled; run `vaani enroll record`",
                      file=sys.stderr)
            elif not ledger.has_active_consent():
                print("  voice consent is not active; run `vaani consent grant`",
                      file=sys.stderr)
            else:
                synthesizer = XttsSynthesizer(
                    profile_store=store, consent_ledger=ledger,
                    device=args.device if args.device != "auto"
                    else plan.device_for("tts"))
                print("Loading voice model (XTTS-v2)...")
                t0 = time.perf_counter()
                synthesizer.warmup()
                voice_profile_id = profile.id
                voice_note = f"YOUR VOICE (profile {profile.id[:8]})"
                print(f"  ready in {time.perf_counter() - t0:.1f}s "
                      f"on {synthesizer.device_used}")
        except Exception as exc:
            print(f"  personal voice unavailable ({exc}); using the fallback voice",
                  file=sys.stderr)

    print("\n" + "!" * 72)
    print("ACTIVE CONFIGURATION:")
    print(f"  translation : {translation_note}")
    print(f"  voice       : {voice_note}")
    print("!" * 72 + "\n")

    def on_result(r: UtteranceResult) -> None:
        if r.suppressed:
            reason = r.suppression_reason.value if r.suppression_reason else "unknown"
            print(f"  [{r.utterance.seq}] SUPPRESSED ({reason})"
                  + (f" - {r.transcript.text[:60]}" if r.transcript else ""))
            return
        stages = " ".join(f"{t.stage}={t.duration_ms:.0f}ms"
                          for t in r.timings if t.duration_ms >= 1)
        print(f"  [{r.utterance.seq}] {r.transcript.text}")
        print(f"      -> {r.translation.text}")
        print(f"      {r.total_latency_ms:.0f}ms total  ({stages})")

    def on_latency_warning(w) -> None:
        print(f"  !! {w.message()}")

    def on_state(old: SessionState, new: SessionState) -> None:
        if args.verbose:
            print(f"      . {old.value} -> {new.value}")

    database = None
    if not args.no_db:
        try:
            from .storage.db import Database
            database = Database()
            crashed = database.mark_crashed_sessions()
            if crashed:
                print(f"  recovered {crashed} session(s) left open by a crash")
            purged = database.apply_retention()
            if any(purged.values()):
                print(f"  retention: {purged}")
        except Exception as exc:
            print(f"  persistence unavailable ({exc}); continuing without it",
                  file=sys.stderr)

    config = SessionConfig(
        database=database,
        persist_transcript=args.save_transcript,
        input_device=args.input,
        monitor_device=args.monitor,
        use_virtual_mic=not args.no_virtual_mic,
        performance_mode=PerformanceMode(args.mode),
    )
    config.voice_profile_id = voice_profile_id
    try:
        vad = SileroVad()
        vad.is_speech(__import__("numpy").zeros(320, dtype="float32"))
    except Exception as exc:
        print(f"Silero VAD unavailable ({exc}); using the energy VAD", file=sys.stderr)
        vad = EnergyVad()

    session = TranslationSession(
        recognizer=recognizer, translator=translator,
        synthesizer=synthesizer, vad=vad,
        config=config, on_result=on_result, on_state=on_state,
        on_latency_warning=on_latency_warning)

    try:
        session.start()
    except Exception as exc:
        print(f"ERROR: could not start: {exc}", file=sys.stderr)
        return 2

    if session._virtual_mic:
        print(f"Virtual microphone live: 'Vaani Virtual Microphone'")
        print(f"  Select it as your mic in Zoom / Meet / Teams.")
    print(f"Mode: {args.mode}   Listening. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        session.stop()

    stats = session.latency_percentiles()
    print("\n" + "=" * 72)
    print(f"Utterances: {len(session.results)}   "
          f"suppressed: {sum(1 for r in session.results if r.suppressed)}   "
          f"dropped: {session.dropped_utterances}")
    if stats:
        print(f"End-to-end latency (measured): p50 {stats['p50']}ms  "
              f"p95 {stats['p95']}ms  min {stats['min']}ms  max {stats['max']}ms")
    if database is not None:
        reasons = database.suppression_summary(days=1)
        if reasons:
            print("Suppressed because: "
                  + ", ".join(f"{k}={v}" for k, v in reasons.items()))
        print(f"Transcript text on disk: "
              f"{'YES (you passed --save-transcript)' if args.save_transcript else 'NO'}")
    else:
        print("No completed utterances - nothing to report.")
    print("=" * 72)
    return 0



def cmd_transcribe(args) -> int:
    """Microphone -> text only. No translation, no synthesis, no virtual mic.

    The simplest way to check that recognition works on your voice, and the
    fastest loop for judging accuracy on your own accent and Hinglish.
    """
    import numpy as np

    from .ai.stt.whisper import FasterWhisperRecognizer
    from .ai.vad.energy import EnergyVad
    from .audio.backend.pulse_backend import PulseCaptureStream
    from .audio.segmenter import SegmenterConfig, UtteranceSegmenter
    from .core.types import PerformanceMode

    mode = PerformanceMode(args.mode)
    print(f"Loading Whisper ({args.model})...")
    t0 = time.perf_counter()
    try:
        recognizer = FasterWhisperRecognizer(model_size=args.model,
                                             device=args.device)
        recognizer.warmup()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"  ready in {time.perf_counter() - t0:.1f}s on "
          f"{recognizer.device_used}/{recognizer.compute_type_used}")

    vad = EnergyVad()
    if not args.simple_vad:
        try:
            from .ai.vad.silero import SileroVad
            v = SileroVad()
            v.is_speech(np.zeros(320, dtype=np.float32))
            vad = v
        except Exception as exc:
            print(f"  (Silero VAD unavailable: {exc}; using the energy VAD)")

    seg_cfg = SegmenterConfig.for_mode(mode)
    segmenter = UtteranceSegmenter(config=seg_cfg, sample_rate=16000, frame_ms=20)
    try:
        capture = PulseCaptureStream(device=args.input, sample_rate=16000,
                                     frame_ms=20, stream_name="transcribe")
    except Exception as exc:
        print(f"ERROR: could not open the microphone: {exc}", file=sys.stderr)
        return 2

    print(f"\nListening. Speak in Hindi, English or Hinglish. Ctrl+C to stop.")
    print(f"(pause {seg_cfg.hangover_ms} ms to end an utterance)\n")

    seq = 0
    latencies: list[float] = []
    try:
        while True:
            frame = capture.read_frame()
            segment = segmenter.push(frame, vad.is_speech(frame))
            if segment is None:
                continue
            seq += 1
            t0 = time.perf_counter()
            try:
                tr = recognizer.transcribe(segment.audio, 16000)
            except Exception as exc:
                print(f"  [{seq}] (not recognised: {exc})")
                continue
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)
            langs = " ".join(f"{k}:{v:.0%}" for k, v in
                             sorted(tr.language_distribution.items(),
                                    key=lambda kv: -kv[1]))
            print(f"  [{seq}] {tr.text}")
            print(f"       conf {tr.confidence:.2f}  {langs}  "
                  f"{segment.speech_ms/1000:.1f}s speech  {ms:.0f}ms")
            if args.save:
                import soundfile as sf
                path = f"{args.save}/utterance_{seq:03d}.wav"
                sf.write(path, segment.audio, 16000)
                print(f"       saved {path}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        capture.close()

    if latencies:
        ordered = sorted(latencies)
        print(f"\n{len(latencies)} utterance(s)   "
              f"STT p50 {ordered[len(ordered)//2]:.0f}ms   max {ordered[-1]:.0f}ms")
    return 0



def cmd_meeting(args) -> int:
    """Background meeting mode: starts hidden, driven by the control socket.

    Designed to be started BEFORE the call and then left alone. Nothing appears
    on screen unless asked for; translation runs regardless of whether the
    overlay is visible.
    """
    import tkinter as tk

    from .ai.stt.whisper import FasterWhisperRecognizer
    from .ai.tts.fallback import FallbackSynthesizer
    from .ai.translate.nllb import NllbTranslator
    from .ai.translate.ollama import OllamaTranslator
    from .ai.translate.router import RoutingTranslator
    from .ai.vad.energy import EnergyVad
    from .core.state_machine import SessionState
    from .ipc.control import ControlServer
    from .session.engine import SessionConfig, TranslationSession
    from .ui.overlay import MeetingOverlay

    print("Starting Vaani in meeting mode...")
    try:
        recognizer = FasterWhisperRecognizer.for_mode(PerformanceMode(args.mode))
        recognizer.warmup()
        print(f"  recognition: {recognizer.model_size} on {recognizer.device_used}")
    except Exception as exc:
        print(f"ERROR: recognition unavailable: {exc}", file=sys.stderr)
        return 2

    translator = NllbTranslator()
    try:
        translator = RoutingTranslator(fast=translator,
                                       accurate=OllamaTranslator(model=args.llm_model))
        translator.warmup()
        print("  translation: routing (NMT + local LLM)")
    except Exception as exc:
        print(f"  translation: NLLB only ({exc})")
        translator.warmup()

    synthesizer, voice_profile_id, voice_note = FallbackSynthesizer(), None, "fallback"
    if args.voice != "fallback":
        try:
            from .ai.tts.xtts import XttsSynthesizer
            from .voice.enrollment import VoiceProfileStore
            profile = VoiceProfileStore().active()
            if profile is not None:
                synthesizer = XttsSynthesizer()
                synthesizer.warmup()
                voice_profile_id, voice_note = profile.id, f"your voice ({profile.name})"
        except Exception as exc:
            print(f"  personal voice unavailable: {exc}", file=sys.stderr)
    print(f"  voice: {voice_note}")

    root = tk.Tk()
    root.withdraw()          # no main window at all in this mode

    session = TranslationSession(
        recognizer=recognizer, translator=translator, synthesizer=synthesizer,
        vad=EnergyVad(),
        config=SessionConfig(input_device=args.input,
                             performance_mode=PerformanceMode(args.mode),
                             voice_profile_id=voice_profile_id))

    overlay = MeetingOverlay(
        root, start_hidden=not args.show,
        corner=args.corner,
        on_panic=lambda: (session.emergency_stop(), root.quit()),
        on_toggle_pause=lambda: (session.resume()
                                 if session.sm.state is SessionState.PAUSED
                                 else session.pause()))

    def on_result(result) -> None:
        stats = session.latency_percentiles()
        src = result.transcript.text if result.transcript else ""
        if result.suppressed:
            reason = (result.suppression_reason.value
                      if result.suppression_reason else "low confidence")
            # `suppressed` is keyword-only, so it cannot ride along in after()'s
            # positional arguments -- bind the call in a closure instead.
            root.after(0, lambda: overlay.set_lines(src, reason, suppressed=True))
        else:
            out = result.translation.text if result.translation else ""
            root.after(0, lambda: overlay.set_lines(src, out))
        p50 = stats.get("p50") if stats else None
        root.after(0, lambda: overlay.set_state(session.sm.state.value, p50))

    session._on_result = on_result

    def handle(command: str) -> dict:
        if command == "show":
            root.after(0, overlay.show)
        elif command == "hide":
            root.after(0, overlay.hide)
        elif command == "toggle":
            root.after(0, overlay.toggle)
        elif command == "pause":
            session.pause()
        elif command == "resume":
            session.resume()
        elif command == "mute":
            session.set_muted(True)
        elif command == "unmute":
            session.set_muted(False)
        elif command == "panic":
            session.emergency_stop()
            root.after(0, root.quit)
        elif command == "quit":
            root.after(0, root.quit)
        stats = session.latency_percentiles()
        return {"state": session.sm.state.value, "visible": overlay.visible,
                "utterances": len(session.results),
                "p50_ms": stats.get("p50") if stats else None}

    import signal

    def _bail(_signum, _frame):
        # Ctrl+C and `kill` must still put the microphone back. Only kill -9
        # cannot be caught, which is what reclaim_orphaned_routing covers.
        if router is not None:
            router.restore_all(fallback_source=args.input)
        try:
            root.quit()
        except Exception:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _bail)
        except (ValueError, OSError):
            pass
    if hasattr(signal, 'SIGHUP'):
        try:
            signal.signal(signal.SIGHUP, _bail)
        except (ValueError, OSError):
            pass

    control = ControlServer(handle)
    try:
        control.start()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        session.start()
    except Exception as exc:
        control.stop()
        print(f"ERROR: could not start: {exc}", file=sys.stderr)
        return 2

    print()
    print("=" * 66)
    print("RUNNING" + ("  (overlay visible)" if args.show else "  (hidden)"))
    print("  In your meeting app, select microphone:")
    print("    Vaani Virtual Microphone")
    print()
    from .devices.routing import AutoRouter, list_capture_streams
    router = AutoRouter(real_source=args.input) if args.auto_route else None

    def check_routing() -> None:
        """Poll for a meeting app on the virtual mic.

        Without this the failure is completely silent: Vaani reports healthy,
        translation runs, and the meeting still hears the untranslated original
        because the client never switched microphones.
        """
        if router is not None:
            for stream in router.sweep():
                print(f"  [routing] switched {stream.app_name} to the Vaani "
                      f"virtual microphone")
        from .devices.manager import virtual_mic_consumers
        consumers = virtual_mic_consumers()
        if consumers:
            if not getattr(check_routing, "announced", False):
                print(f"  [routing] connected: "
                      f"{', '.join(c.app_name for c in consumers)}")
                check_routing.announced = True
        else:
            if getattr(check_routing, "announced", False):
                print("  [routing] no app is reading the virtual mic any more")
            check_routing.announced = False
        root.after(3000, check_routing)

    root.after(3000, check_routing)

    print("  Control it from any terminal, or bind these to KDE shortcuts:")
    print("    vaani show     vaani hide     vaani toggle")
    print("    vaani pause    vaani mute     vaani panic")
    print("=" * 66)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        control.stop()
        if router is not None:
            # Put meeting apps back on a real microphone. Leaving them on a dead
            # virtual mic would make the user silent in their next call.
            restored = router.restore_all(fallback_source=args.input)
            if restored:
                print(f"Restored {restored} application(s) to the real microphone.")
        session.stop()
        try:
            root.destroy()
        except Exception:
            pass
    print("Stopped.")
    return 0


def cmd_control(args) -> int:
    from .ipc.control import send
    try:
        reply = send(args.command)
    except ConnectionError:
        print("Vaani is not running. Start it with:  vaani meeting",
              file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"control failed: {exc}", file=sys.stderr)
        return 1
    if not reply.get("ok"):
        print(reply.get("error", "failed"), file=sys.stderr)
        return 1
    if args.command == "status":
        from .devices.manager import virtual_mic_consumers
        consumers = virtual_mic_consumers()
        if consumers:
            print(f"meeting app  CONNECTED: "
                  f"{', '.join(c.app_name for c in consumers)}")
        else:
            print("meeting app  NOT CONNECTED  <-- participants will hear your")
            print("             untranslated voice. Select 'Vaani Virtual")
            print("             Microphone' as the mic inside your meeting app.")
        print(f"state      {reply.get('state')}")
        print(f"overlay    {'visible' if reply.get('visible') else 'hidden'}")
        print(f"utterances {reply.get('utterances')}")
        p50 = reply.get("p50_ms")
        print(f"latency    {f'{p50:.0f} ms' if p50 else '—'}")
    return 0


def cmd_fix_audio(args) -> int:
    """Undo anything Vaani did to the system's audio routing.

    The escape hatch. If Vaani is killed rather than stopped, applications can be
    left pointing at a virtual microphone that no longer exists, and the symptom
    is that the user's microphone appears broken everywhere -- with nothing on
    screen connecting it to this program.
    """
    if os.name == 'nt':
        print("Virtual mic management not available on Windows.")
        return 0

    import subprocess

    from .devices.manager import VIRTUAL_MIC_NAME
    from .devices.routing import reclaim_orphaned_routing

    print("Repairing audio routing...")

    killed = 0
    try:
        out = subprocess.run(["pgrep", "-f", "vaani.cli"], capture_output=True,
                             text=True, check=False).stdout.split()
        for pid in out:
            if pid and int(pid) != os.getpid():
                subprocess.run(["kill", "-9", pid], capture_output=True)
                killed += 1
    except Exception:
        pass
    if killed:
        print(f"  stopped {killed} leftover Vaani process(es)")

    default = subprocess.run(["pactl", "get-default-source"], capture_output=True,
                             text=True, check=False).stdout.strip()
    restored = reclaim_orphaned_routing(default or None)
    if restored:
        print(f"  moved {restored} application(s) back to the real microphone")

    # Anything still on the virtual mic, whether we recorded moving it or not.
    try:
        srcs = subprocess.run(["pactl", "list", "short", "sources"],
                              capture_output=True, text=True, check=False).stdout
        vid = next((l.split("\t")[0] for l in srcs.splitlines()
                    if len(l.split("\t")) > 1 and l.split("\t")[1] == VIRTUAL_MIC_NAME),
                   None)
        if vid and default:
            listing = subprocess.run(["pactl", "list", "source-outputs"],
                                     capture_output=True, text=True,
                                     check=False).stdout
            idx = None
            for line in listing.splitlines():
                st = line.strip()
                if st.startswith("Source Output #"):
                    idx = st.split("#")[-1]
                elif st.startswith("Source:") and st.split(":")[1].strip() == vid and idx:
                    subprocess.run(["pactl", "move-source-output", idx, default],
                                   capture_output=True)
                    print(f"  moved stream #{idx} off the virtual microphone")
    except Exception:
        pass

    modules = subprocess.run(["pactl", "list", "short", "modules"],
                             capture_output=True, text=True, check=False).stdout
    removed = 0
    for line in modules.splitlines():
        if "vaani" in line.lower():
            subprocess.run(["pactl", "unload-module", line.split("\t")[0]],
                           capture_output=True)
            removed += 1
    if removed:
        print(f"  removed {removed} leftover Vaani audio device(s)")

    from pathlib import Path
    if os.name != 'nt':
        Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/vaani-{_get_user_id()}")
             ).joinpath("vaani.sock").unlink(missing_ok=True)

    print(f"\nDone. Your microphone is back on: {default or 'the system default'}")
    print("Test it with:  vaani transcribe")
    return 0


def cmd_route(args) -> int:
    """Show which apps are capturing audio, and switch them over."""
    from .devices.manager import VirtualMicrophone
    from .devices.routing import (
        list_capture_streams,
        move_to_virtual_mic,
    )

    if VirtualMicrophone.find_existing() is None:
        print("The virtual microphone does not exist yet. Start Vaani first:")
        print("  vaani meeting")
        return 1

    streams = list_capture_streams()
    if not streams:
        print("No application is capturing audio right now.")
        print("Start your call first, then run this again.")
        return 0

    print("Applications capturing audio:")
    for s_ in streams:
        if s_.is_own:
            where = "Vaani itself"
        elif s_.on_virtual_mic:
            where = "ALREADY on the Vaani virtual mic"
        else:
            where = "on a real microphone"
        print(f"  #{s_.index:<5} {s_.app_name[:28]:28s} {where}")

    targets = [s_ for s_ in streams if s_.movable]
    if not targets:
        print("\nNothing to switch.")
        return 0

    print()
    moved = 0
    for s_ in targets:
        try:
            move_to_virtual_mic(s_)
            print(f"  switched {s_.app_name} -> Vaani Virtual Microphone")
            moved += 1
        except Exception as exc:
            print(f"  could not switch {s_.app_name}: {exc}", file=sys.stderr)
    print(f"\n{moved} application(s) now receive your translated voice.")
    return 0


def cmd_gui(args) -> int:
    try:
        if os.name == 'nt':
            from .ui.windows_console import main as gui_main
        else:
            from .ui.app import main as gui_main
    except ImportError as exc:
        print(f"GUI unavailable: {exc}\n"
              "tkinter is required (apt install python3-tk).", file=sys.stderr)
        return 2
    return gui_main()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vaani",
                                 description="Real-time personal voice translation")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("transcribe",
                       help="microphone -> text only (no translation or speech)")
    p.add_argument("--input", help="capture device key (see `vaani devices`)")
    p.add_argument("--model", default="small",
                   help="Whisper model: tiny/base/small/medium/large-v3")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--mode", default="balanced",
                   choices=[m.value for m in PerformanceMode])
    p.add_argument("--simple-vad", action="store_true",
                   help="use the energy VAD instead of Silero")
    p.add_argument("--save", help="also save each utterance as a wav in this dir")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("meeting",
                       help="background meeting mode (starts hidden)")
    p.add_argument("--input", help="capture device key")
    p.add_argument("--mode", default="balanced",
                   choices=[m.value for m in PerformanceMode])
    p.add_argument("--voice", default="personal",
                   choices=["personal", "fallback"])
    p.add_argument("--llm-model", default="qwen3-coder:latest")
    p.add_argument("--show", action="store_true",
                   help="start with the overlay visible (default: hidden)")
    p.add_argument("--corner", default="bottom-right",
                   choices=["top-left", "top-right", "bottom-left", "bottom-right"])
    p.add_argument("--no-auto-route", dest="auto_route", action="store_false",
                   help="do not move meeting apps onto the virtual mic "
                        "automatically; switch the microphone yourself instead")
    p.set_defaults(func=cmd_meeting, auto_route=True)

    for name, helptext in (("show", "show the overlay"),
                           ("hide", "hide the overlay (translation keeps running)"),
                           ("toggle", "toggle the overlay"),
                           ("pause", "pause translation"),
                           ("resume", "resume translation"),
                           ("mute", "stop capturing audio"),
                           ("unmute", "resume capturing audio"),
                           ("panic", "emergency stop and exit"),
                           ("status", "show the running session's status")):
        sp = sub.add_parser(name, help=helptext)
        sp.set_defaults(func=cmd_control, command=name)

    p = sub.add_parser("fix-audio",
                       help="undo Vaani's audio routing if your mic stops working")
    p.set_defaults(func=cmd_fix_audio)

    p = sub.add_parser("route",
                       help="switch running apps onto the Vaani virtual mic")
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("gui", help="launch the desktop application")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("devices", help="list audio devices")
    p.add_argument("--all", action="store_true", help="include monitor sources")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("doctor", help="run diagnostics")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("virtualmic", help="manage the virtual microphone")
    p.add_argument("action", choices=["create", "destroy", "status"])
    p.set_defaults(func=cmd_virtualmic)

    p = sub.add_parser("consent", help="manage voice-cloning consent")
    p.add_argument("action", choices=["grant", "revoke", "status", "show"])
    p.add_argument("--subject", help="whose voice (non-interactive)")
    p.add_argument("--confirm", help="confirmation phrase (non-interactive)")
    p.set_defaults(func=cmd_consent)

    p = sub.add_parser("enroll", help="record and manage voice profiles")
    p.add_argument("action", choices=["record", "list", "delete"])
    p.add_argument("--name", default="me")
    p.add_argument("--seconds", type=float, default=90.0)
    p.add_argument("--input", help="capture device key")
    p.add_argument("--profile-id")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("run", help="start a translation session")
    p.add_argument("--input", help="capture device key (see `vaani devices`)")
    p.add_argument("--monitor", help="also play output to this sink")
    p.add_argument("--no-virtual-mic", action="store_true")
    p.add_argument("--model", default="small", help="Whisper model size")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--mode", default="balanced",
                   choices=[m.value for m in PerformanceMode])
    p.add_argument("--translate", default="routing",
                   choices=["routing", "nllb", "none"],
                   help="'routing' = NLLB + LLM for romanised Hinglish (best "
                        "accuracy); 'nllb' = fast only; 'none' = pass-through")
    p.add_argument("--llm-model", default="qwen3-coder:latest",
                   help="Ollama model used for romanised Hinglish")
    p.add_argument("--voice", default="personal", choices=["personal", "fallback"],
                   help="'personal' uses your enrolled voice if available")
    p.add_argument("--save-transcript", action="store_true",
                   help="store transcript text on disk (OFF by default; without "
                        "this, text is never written)")
    p.add_argument("--no-db", action="store_true",
                   help="disable all persistence, including latency metrics")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
