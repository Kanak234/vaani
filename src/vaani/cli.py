"""Command-line entry point.

The GUI described in the UI/UX brief is not implemented in this MVP. This CLI
exposes the same capabilities so the engine can be exercised and measured now,
and so every claim in the docs is checkable by running something.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

def _get_user_id() -> str:
    if os.name == 'nt':
        return os.environ.get('USERNAME', 'user')
    return str(os.getuid())

from .core.errors import ErrorCode, Severity, VaaniError
from .session.input_mode import InputMode
from .system.logging import setup_logging
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


def cmd_models(args) -> int:
    """Show what Vaani would choose for each stage, and WHY.

    WHY THIS COMMAND EXISTS: auto-detection is only trustworthy if the user can
    see its reasoning. "Vaani picked a model" is not checkable; "picked
    phi4-mini because coder builds were measured dropping dates" is. This is
    also the fastest way to answer the two questions that actually come up --
    "is it using my GPU?" and "why is it slow?" -- without starting a meeting.

    Read-only: it loads nothing and changes nothing unless --probe is passed.
    """
    from .ai.model_detect import (
        is_discouraged, list_models, rank_models, resolve_llm_model,
    )
    from .ai.stt.cuda_setup import cuda_available
    from .core.types import PerformanceMode

    print("=" * 72)
    print("VAANI MODEL SELECTION")
    print("=" * 72)

    # --- hardware -----------------------------------------------------------
    gpu_ok, gpu_reason = cuda_available()
    print(f"\nGPU        {'CUDA usable' if gpu_ok else 'CPU only'}  ({gpu_reason})")
    if not gpu_ok and "VAANI_FORCE_CPU" not in gpu_reason:
        print("           set VAANI_FORCE_CPU=1 to force this deliberately")
    elif gpu_ok:
        print("           VAANI_FORCE_CPU=1 forces CPU, to test the fallback path")

    # --- speech recognition -------------------------------------------------
    from .ai.stt.whisper import FasterWhisperRecognizer
    print("\nSPEECH RECOGNITION (Whisper)")
    for mode in PerformanceMode:
        rec = FasterWhisperRecognizer.for_mode(mode)
        cached = "cached" if rec.is_cached() else "NOT downloaded yet"
        marker = " <- default" if mode is PerformanceMode.BALANCED else ""
        print(f"  {mode.value:<12} {rec.model_size:<8} ({cached}){marker}")
    if not gpu_ok:
        print("  reason: stepped down from 'medium' because medium on CPU is")
        print("          RTF ~0.6+, which would blow every latency budget")

    # --- translation --------------------------------------------------------
    print("\nTRANSLATION (local LLM via Ollama)")
    installed = list_models()
    if not installed:
        print("  no Ollama models found. Translation falls back to NLLB alone,")
        print("  which cannot use conversation context (see ADR-004).")
        print("  Fix: ollama serve  &&  ollama pull phi4-mini")
    else:
        for index, name in enumerate(rank_models(installed)):
            flag = "  DISCOURAGED (coder/embed build)" if is_discouraged(name) else ""
            print(f"  {index + 1}. {name}{flag}")
        choice = resolve_llm_model(pinned=args.model, probe=args.probe)
        print(f"\n  chosen: {choice.describe()}")
        if choice.rejected:
            for name, why in choice.rejected.items():
                print(f"  rejected {name}: {why}")
        if not args.probe:
            print("  (ranking only; pass --probe to actually run each candidate)")

    print("\nRanking rationale: ai/model_detect.PREFERRED_MODELS - it is")
    print("evidence-led, and every entry records what was measured.")
    return 0


def cmd_doctor(args) -> int:
    from .diagnostics.checks import run_all
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
        print("Virtual microphone ready.")
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
        print("This permanently deletes the voice profile and its recordings.")
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
    print("Read the prompts below aloud, naturally. Need >= 45s of speech.\n")
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
        print("Virtual microphone live: 'Vaani Virtual Microphone'")
        print("  Select it as your mic in Zoom / Meet / Teams.")
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
    # Lifetime counters, not len(session.results): that is now a bounded
    # rolling window, so counting it would understate any meeting longer than
    # max_retained_results utterances.
    print(f"Utterances: {session.total_utterances}   "
          f"suppressed: {session.total_suppressed}   "
          f"dropped: {session.dropped_utterances}   "
          f"failed: {session.failed_utterances}")
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

    print("\nListening. Speak in Hindi, English or Hinglish. Ctrl+C to stop.")
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



def _build_translator(fast, args) -> tuple[object, str]:
    """Build the best translator that actually warms up, and say which it is.

    This is the single degradation ladder used by every entry point, which is
    the D-3 fix: `meeting`, `run` and the takeover app used to hardcode three
    DIFFERENT default models (`qwen3-coder:latest`, `phi4-mini:latest`,
    `qwen3:8b`), two of which were not installed on the development machine.
    Now they all come through here and all get `--llm-model` = None = detect.

    The ladder, best first:

        1. routing  NLLB for Devanagari/English + local LLM for romanised
                    Hinglish. Measured 14/14 -- the only 100% configuration.
        2. NLLB     the LLM could not load. Romanised Hinglish then comes back
                    untranslated and the gate suppresses it (ADR-003), so the
                    user gets silence and a reason, never wrong speech.
        3. LLM      NLLB could not load -- which is the CURRENT state of a
                    `pip install -e ".[stt,translate]"` venv, because NLLB needs
                    torch and that extra does not pull it (D-4). The LLM alone
                    handles every register; it is just slower.
        4. fail     nothing loaded. The caller aborts before opening a mic.

    Returns `(translator, note)`. The note is printed, because a silent
    downgrade from "your voice, translated" to "suppressed" looks like a bug.
    """
    from .ai.translate.ollama import OllamaTranslator
    from .ai.translate.router import RoutingTranslator

    # None => auto-detect (F-22). An explicit --llm-model is still honoured and
    # still hard-fails if absent, rather than being silently substituted.
    pinned = getattr(args, "llm_model", None)
    llm = OllamaTranslator(model=pinned) if pinned else OllamaTranslator(model=None)

    llm_error: str | None = None
    try:
        llm.warmup()
    except Exception as exc:
        llm_error = str(exc)
        llm = None

    fast_error: str | None = None
    try:
        fast.warmup()
    except Exception as exc:
        # NLLB is the one that fails on a torch-less install, so this branch is
        # not hypothetical -- it is the default state of the documented install.
        fast_error = str(exc)
        fast = None

    if fast is not None and llm is not None:
        router = RoutingTranslator(fast=fast, accurate=llm)
        # Both halves are already warm; RoutingTranslator.warmup() would only
        # re-warm them. Setting health directly keeps start-up off a second
        # model load.
        from .ai.base import HealthState
        router._set_health(HealthState.READY, "routing ready")
        return router, f"routing (NLLB + {llm.model})"

    if fast is not None:
        return fast, f"NLLB only -- romanised Hinglish will be suppressed ({llm_error})"

    if llm is not None:
        return llm, f"local LLM only ({llm.model}) -- NLLB unavailable ({fast_error})"

    raise VaaniError(
        code=ErrorCode.PROVIDER_UNAVAILABLE,
        message=f"no translation engine could start. NLLB: {fast_error}. "
                f"LLM: {llm_error}.",
        severity=Severity.FATAL)


def _build_takeover(args, *, session, recognizer, synthesizer,
                    voice_profile_id, overlay, root):
    """Wire up F-21 takeover, or refuse with a reason. Returns (runtime, listener).

    Returns `(None, None)` after printing WHY when a precondition fails. Each
    refusal is a separate, actionable message (AC-21.5) because "takeover
    unavailable" on its own is useless at the start of a call.

    Preconditions, and why each one is hard rather than a warning:

      E-2 remote == input   we would transcribe the USER as the remote party,
                            so Vaani would answer the user's own questions in
                            the user's own voice. Nonsense, and confusing to
                            debug from the symptom.
      E-5 no enrolled voice no consent, or no profile, means the answer would be
                            spoken in the FALLBACK voice -- a synthetic robot
                            claiming to be the owner in front of colleagues.
                            AC-21.6 forbids this outright.
      E-4 no LLM            there is nothing to draft an answer WITH. Translation
                            is unaffected and keeps running if the user restarts
                            without --takeover.

    Note what is NOT a precondition: arming. `--takeover` opens the remote stream
    but leaves the policy OFF (AC-21.2), because listening is observable and
    harmless while authorising generated speech is neither.
    """
    from .ai.assist import AnswerAssistant
    from .ai.translate.ollama import OllamaTranslator
    from .session.remote_listener import RemoteQuestionListener, looks_hesitant
    from .session.takeover import TakeoverConfig
    from .session.takeover_runtime import MeetingTakeoverRuntime

    remote = getattr(args, "remote_input", None)
    if not remote:
        # E-1. Also enforced at parse time; kept here so the programmatic path
        # cannot bypass it.
        print("ERROR: --takeover needs --remote-input (the meeting's monitor "
              "source).\n       Find it with: vaani devices --all",
              file=sys.stderr)
        return None, None

    if args.input and remote == args.input:
        print(f"ERROR: --remote-input and --input are the same device "
              f"({remote!r}).\n       Vaani would hear you as the other party. "
              "Pick the monitor of your OUTPUT sink.", file=sys.stderr)
        return None, None

    if voice_profile_id is None:
        print("ERROR: takeover needs your enrolled voice, and none is active.\n"
              "       Run: vaani consent grant && vaani enroll record\n"
              "       (Refusing rather than speaking for you in the fallback "
              "robot voice.)", file=sys.stderr)
        return None, None

    # The assistant needs `generate_raw`, which only OllamaTranslator has -- the
    # router does not implement it, so it cannot be passed here.
    llm = OllamaTranslator(model=getattr(args, "llm_model", None))
    try:
        llm.warmup()
    except Exception as exc:
        print(f"ERROR: takeover needs a local LLM to draft answers: {exc}\n"
              "       Start Ollama (`ollama serve`) and pull one "
              "(`ollama pull phi4-mini`).", file=sys.stderr)
        return None, None

    runtime = MeetingTakeoverRuntime(
        assistant=AnswerAssistant(translator=llm),
        synthesizer=synthesizer,
        config=TakeoverConfig(
            silence_after_question_s=getattr(args, "takeover_silence", 4.0)),
        voice_profile_id=voice_profile_id)

    def emit(response) -> None:
        """Speak an authorised answer and record it where the user can audit it."""
        if response is None or response.audio is None:
            return
        if not session.sm.can_emit_audio:
            return          # a panic between authorisation and playback wins
        session.inject_audio(response.audio.samples)
        # Printed AND shown: docs/15 §2.7 requires every generated sentence to be
        # attributable after the fact, with the reason it was triggered.
        print(f"[TAKEOVER] heard: {response.question}")
        print(f"[TAKEOVER] said : {response.text}   (reason={response.reason})")
        root.after(0, lambda: overlay.set_lines(
            f"[them] {response.question}", f"[Vaani spoke] {response.text}"))

    def on_question(text: str) -> None:
        runtime.observe_remote_question(text)
        root.after(0, lambda: overlay.set_takeover(runtime.controller.state.value))

    listener = RemoteQuestionListener(
        device=remote, recognizer=recognizer, on_question=on_question,
        performance_mode=PerformanceMode(args.mode),
        on_error=lambda msg: print(f"[TAKEOVER] {msg}", file=sys.stderr))

    # The user-signal side: every utterance the MAIN pipeline produces is also
    # evidence about whether the user is answering or stuck. Hooking the
    # session's existing result callback avoids a second STT pass over the same
    # audio -- the transcript already exists by the time this runs.
    previous_on_result = session._on_result

    def on_result_with_takeover(result) -> None:
        if previous_on_result is not None:
            previous_on_result(result)
        if result.transcript is None:
            return
        runtime.observe_user_speech(result.transcript.text,
                                    hesitation=looks_hesitant(result.transcript.text))
        emit(runtime.respond_if_authorized())
        root.after(0, lambda: overlay.set_takeover(runtime.controller.state.value))

    session._on_result = on_result_with_takeover

    # Poll for the silence-based handoff. Nothing in the result callback fires
    # while the user is SILENT, which is precisely the case takeover exists for,
    # so a timer is required rather than optional.
    def poll_takeover() -> None:
        if runtime.controller.armed:
            try:
                emit(runtime.respond_if_authorized())
            except VaaniError:
                pass        # no pending question yet; normal
            overlay.set_takeover(runtime.controller.state.value)
        root.after(1000, poll_takeover)

    root.after(1000, poll_takeover)
    runtime._emit = emit            # reused by the `takeover-now` control command
    return runtime, listener


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
    from .ai.vad.energy import EnergyVad
    from .core.state_machine import SessionState
    from .ipc.control import ControlServer
    from .session.engine import SessionConfig, TranslationSession
    from .ui.overlay import MeetingOverlay

    print("Starting Vaani in meeting mode...")
    try:
        recognizer = FasterWhisperRecognizer.for_mode(PerformanceMode(args.mode))
        # WHY THIS WARNING EXISTS: warmup() downloads the Whisper model on
        # first run - up to ~1.5 GB for `medium` - through huggingface_hub,
        # whose progress bars do not reach the terminal here. The user got
        # "Starting Vaani in meeting mode..." followed by several minutes of
        # complete silence, which is indistinguishable from a hang. It was
        # mistaken for one during Phase 2 testing, which is exactly what a
        # first-time user would conclude before killing it.
        # The real fix is to pre-fetch models in install.sh (Phase 6); until
        # then, at least say what is happening and that it happens once.
        if not recognizer.is_cached():
            print(f"  first run: downloading the '{recognizer.model_size}' "
                  f"speech model (this can take several minutes on a slow "
                  f"link, and happens only once)...", flush=True)
        recognizer.warmup()
        print(f"  recognition: {recognizer.model_size} on {recognizer.device_used}")
    except Exception as exc:
        print(f"ERROR: recognition unavailable: {exc}", file=sys.stderr)
        return 2

    # D-1 FIX (docs/15 §4). The old code did:
    #
    #     translator = NllbTranslator()
    #     try:
    #         translator = RoutingTranslator(fast=translator, accurate=...)
    #         translator.warmup()
    #     except Exception:
    #         translator.warmup()      # <-- `translator` is the ROUTER now
    #
    # The assignment inside the `try` clobbered the name, so the recovery path
    # re-warmed the object that had just failed and raised during handling of
    # the first exception. Reproduced: `vaani meeting` died with a traceback
    # instead of degrading. Keeping the fast engine in its OWN name is the whole
    # fix -- `fast` is never reassigned, so the fallback can always reach it.
    fast = NllbTranslator()
    try:
        translator, translation_note = _build_translator(fast, args)
    except VaaniError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"  translation: {translation_note}")

    def _degrade_stage(stage: str) -> None:
        """MID-SESSION fallback: this stage has failed repeatedly, swap it.

        Called from the worker thread by TranslationSession when a stage's
        failure window trips. The session decides WHEN; the ladder lives here,
        because the session has no business knowing the provider order.

        Translation degrades down the same ladder _build_translator climbs:
        routing (NLLB + LLM) -> NLLB alone. WHY that direction: the LLM half is
        the part that needs Ollama up, a model resident and VRAM free, so it is
        overwhelmingly the half that breaks mid-meeting. NLLB alone still
        handles Devanagari and English correctly; romanised Hinglish then comes
        back untranslated and the gate suppresses it, so the user gets silence
        and a reason rather than wrong speech (ADR-003).

        There is deliberately NO further rung. Dropping to passthrough would
        put untranslated Hindi into an English meeting, which is worse than
        saying nothing.
        """
        if stage != "translate":
            # STT and TTS have no second engine configured in this build; the
            # overlay warning has already told the user. Saying so honestly
            # beats pretending a fallback happened.
            return
        current = session.pipeline.translator
        # NOTE THE NAMES: RoutingTranslator is built as
        # RoutingTranslator(fast=nllb, accurate=llm), so `fast` is NLLB and
        # `accurate` is the LLM. Degrading means keeping `fast`. Reaching for
        # `accurate` here would hand the session the very engine that just
        # failed three times - which is what the first version of this did.
        nllb = getattr(current, "fast", None)
        llm = getattr(current, "accurate", None)
        if nllb is None or llm is None:
            return              # already a single engine; nothing to drop to
        session.swap_translator(
            nllb, reason="LLM failed repeatedly; NLLB only "
                         "(romanised Hinglish will now be suppressed)")

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
        on_stage_degraded=lambda stage: _degrade_stage(stage),
        config=SessionConfig(input_device=args.input,
                             performance_mode=PerformanceMode(args.mode),
                             # Chosen at startup, changeable mid-meeting with
                             # `vaani mode-*`. See session/input_mode.py.
                             input_mode=InputMode(args.input_mode),
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

    # ------------------------------------------------- Takeover A (continue)
    # Finishes the USER'S OWN point when they stall or press the key. Distinct
    # from Takeover B below, which answers the OTHER participant. See
    # session/continuation.py for why the risk profile is different and every
    # default here is conservative.
    #
    # Only built when an LLM is actually available: without one there is
    # nothing to draft with, and a `vaani continue` that silently does nothing
    # is worse than one that says why.
    continuation = None
    _llm_for_continue = getattr(translator, "accurate", None) or (
        translator if hasattr(translator, "generate_raw") else None)
    if _llm_for_continue is not None:
        from .session.continuation import (
            ContinuationEngine, OllamaCompletion,
        )
        continuation = ContinuationEngine(
            llm=OllamaCompletion(_llm_for_continue))

        _previous_on_result = session._on_result

        def on_result_with_continuation(result) -> None:
            # CHAINS onto the existing callback rather than replacing it - the
            # same trap documented for takeover below.
            if _previous_on_result is not None:
                _previous_on_result(result)
            # Feed the user's OWN words, not the English translation: the LLM
            # must continue what they actually said, not a paraphrase of it.
            if result.transcript is not None and not result.suppressed:
                continuation.note_user_speech(result.transcript.text)

        session._on_result = on_result_with_continuation

    def _speak_continuation(trigger) -> dict:
        """Draft the finish of the user's point and speak it in their voice.

        The AI-SPEAKING indicator is raised BEFORE synthesis and lowered only
        after the audio is queued, so the badge can never lag behind what the
        meeting hears. overlay.set_takeover("active") also force-shows a hidden
        overlay and refuses to be dismissed while it is true.
        """
        if continuation is None:
            return {"error": "no local LLM available; `vaani models` explains why"}
        result = continuation.draft(trigger)
        if result.text is None:
            # Declining is normal and must not look like a failure.
            return {"continuation": None, "reason": result.reason}

        root.after(0, lambda: overlay.set_takeover("active"))
        try:
            audio = synthesizer.synthesize(
                result.text, voice_profile_id=voice_profile_id)
            # Checked again here: the user may have cancelled DURING synthesis,
            # which on this hardware takes seconds.
            if continuation._cancelled.is_set():
                return {"continuation": None, "reason": "cancelled during synthesis"}
            session.inject_audio(audio.samples)
        except Exception as exc:
            return {"error": f"could not speak the continuation: {exc}"}
        finally:
            root.after(0, lambda: overlay.set_takeover("off"))
        return {"continuation": result.text, "reason": result.reason}

    # ------------------------------------------------------------- takeover
    # F-21. This is the unification: until now `vaani meeting` could translate
    # but never hear the other side, while MeetingTakeoverApp could take over
    # but had no overlay, no control socket and no auto-routing. Both now run in
    # this one process, sharing one recognizer, one session and one control
    # socket.
    #
    # Placed AFTER `session._on_result = on_result` on purpose: _build_takeover
    # CHAINS onto whatever callback is already installed. Building it earlier
    # would work and then be silently clobbered by the assignment above, which
    # is the sort of bug that only shows up in a live call.
    #
    # `takeover_rt` stays None unless --takeover was passed and every use below
    # is guarded, so the default path is exactly the old behaviour.
    takeover_rt = None
    remote_listener = None
    if getattr(args, "takeover", False):
        takeover_rt, remote_listener = _build_takeover(
            args, session=session, recognizer=recognizer,
            synthesizer=synthesizer, voice_profile_id=voice_profile_id,
            overlay=overlay, root=root)
        if takeover_rt is None:
            # _build_takeover already explained which precondition failed
            # (E-1/E-2/E-4/E-5). Refuse rather than start a meeting that is
            # quietly not what was asked for.
            return 2
        if getattr(args, "arm", False):
            # --arm is the one way takeover is live from the first second. It is
            # a separate flag from --takeover precisely so that the common case
            # (start before the call, arm only if I actually freeze) does not
            # authorise generated speech by accident.
            takeover_rt.arm()
            print("  takeover: ARMED from start (--arm)")
        else:
            print("  takeover: listening, OFF until armed "
                  "(`vaani takeover-arm`, or say 'Vaani, take over')")

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
            # Panic must also disarm takeover, not just stop the audio. Leaving
            # it ARMED after an emergency stop would let it speak again the
            # moment the session recovered -- the opposite of what the user
            # asked for by hitting panic.
            if takeover_rt is not None:
                takeover_rt.stop()
            session.emergency_stop()
            root.after(0, root.quit)
        elif command == "quit":
            root.after(0, root.quit)
        elif command in ("bypass-on", "bypass-off", "bypass"):
            # Bypass = the user's real voice on the virtual mic. The overlay is
            # updated from the returned state rather than recomputed, so the
            # banner can never disagree with what the audio path is doing.
            if command == "bypass":
                on = session.toggle_bypass()
            else:
                on = command == "bypass-on"
                session.set_bypass(on)
            root.after(0, lambda: overlay.set_input_mode(
                session.input_mode.state().label))
        elif command == "talk":
            # Push-to-talk, toggle flavour: the one that needs no special
            # permissions, so a plain desktop shortcut can drive it.
            session.toggle_talk()
            root.after(0, lambda: overlay.set_input_mode(
                session.input_mode.state().label))
        elif command.startswith("mode-"):
            mode = {"mode-always-on": InputMode.ALWAYS_ON,
                    "mode-ptt-toggle": InputMode.PTT_TOGGLE,
                    "mode-ptt-hold": InputMode.PTT_HOLD}[command]
            session.set_input_mode(mode)
            root.after(0, lambda: overlay.set_input_mode(
                session.input_mode.state().label))
        elif command in ("continue", "continue-arm", "continue-off", "cancel"):
            if continuation is None:
                return {"error": "no local LLM available for continuations"}
            if command == "cancel":
                # Must be the cheapest path in the file: it is the key the user
                # reaches for when Vaani is saying something they disagree with.
                continuation.cancel()
                session.emergency_stop_audio_only() \
                    if hasattr(session, "emergency_stop_audio_only") \
                    else session._flush_pending_audio()
                root.after(0, lambda: overlay.set_takeover("off"))
                return {"cancelled": True}
            if command == "continue-arm":
                continuation.arm()
                return {"continue_armed": True}
            if command == "continue-off":
                continuation.disarm()
                return {"continue_armed": False}
            from .session.continuation import ContinuationTrigger as _Trig
            return _speak_continuation(_Trig.HOTKEY)
        elif command.startswith("takeover-"):
            if takeover_rt is None:
                return {"error": "takeover is not enabled; restart with "
                                 "`vaani meeting --takeover --remote-input ...`"}
            if command == "takeover-arm":
                takeover_rt.arm()
            elif command == "takeover-stop":
                takeover_rt.stop()
            elif command == "takeover-now":
                # Skip the policy wait and speak to the pending question now.
                # Still refuses when OFF: `take_over()` returns WAIT/
                # "takeover_not_armed", so this cannot be used to bypass arming.
                takeover_rt.controller.take_over(reason="control_socket")
                try:
                    takeover_rt._emit(takeover_rt.respond_if_authorized())
                except VaaniError as exc:
                    return {"error": str(exc)}
            root.after(0, lambda: overlay.set_takeover(
                takeover_rt.controller.state.value))
        stats = session.latency_percentiles()
        if takeover_rt is not None:
            # Surfaced in `vaani status` so the user can check from a terminal
            # whether Vaani is currently authorised to speak for them.
            return {"state": session.sm.state.value, "visible": overlay.visible,
                    "utterances": session.total_utterances,
                    "p50_ms": stats.get("p50") if stats else None,
                    "takeover": takeover_rt.controller.state.value,
                    "remote_heard": (remote_listener.utterances_heard
                                     if remote_listener else 0),
                    "remote_questions": (remote_listener.questions_seen
                                         if remote_listener else 0)}
        return {"state": session.sm.state.value, "visible": overlay.visible,
                "input_mode": session.input_mode.state().label,
                "utterances": session.total_utterances,
                "p50_ms": stats.get("p50") if stats else None}

    import signal

    # D-2 FIX (docs/15 §4). `_bail` closes over `router`, which used to be
    # assigned ~40 lines BELOW this point, after the model warm-up and
    # session.start(). A Ctrl+C during those seconds therefore raised
    # UnboundLocalError *inside the signal handler* -- so `root.quit()` was never
    # reached and, worse, the meeting apps were left routed to a virtual mic
    # belonging to a dead process. Binding the name first makes the handler
    # correct at every instant of start-up.
    router = None

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

    # Physical hold-to-talk, only when the user asked for it. WHY it is wired
    # here rather than inside the session: it needs the CLI's ability to PRINT
    # why it could not start. A meeting must never fail to start because a
    # convenience feature is unavailable, so an unavailable listener degrades
    # to ptt_toggle (driven by `vaani talk`) with the reason on screen.
    ptt_listener = None
    if session.input_mode.mode is InputMode.PTT_HOLD:
        from .system.evdev_ptt import PttListener
        ptt_listener = PttListener(
            key=args.ptt_key,
            on_press=session.press_talk,
            on_release=session.release_talk)
        result = ptt_listener.start()
        if result.ok:
            print(f"  push-to-talk: {result.reason}")
        else:
            print(f"  push-to-talk: {result.reason}", file=sys.stderr)
            if result.remedy:
                print(f"                try: {result.remedy}", file=sys.stderr)
            print("                falling back to `vaani talk` (toggle).",
                  file=sys.stderr)
            ptt_listener = None
            # Degrade explicitly rather than leaving the user in a mode whose
            # key nothing is reading - that would look like a dead microphone.
            session.set_input_mode(InputMode.PTT_TOGGLE)

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

    if remote_listener is not None:
        # Started after the session so the two capture streams never race for
        # the same PulseAudio context during device setup.
        try:
            remote_listener.start()
            print(f"  takeover: listening to {args.remote_input}")
        except Exception as exc:
            # E-3 at start-up: translation is already running and is the more
            # important half, so do not abort the meeting over this.
            print(f"  takeover: remote capture unavailable ({exc}); "
                  "explicit 'Vaani, take over' still works", file=sys.stderr)

    print()
    print("=" * 66)
    print("RUNNING" + ("  (overlay visible)" if args.show else "  (hidden)"))
    print("  In your meeting app, select microphone:")
    print("    Vaani Virtual Microphone")
    print()
    from .devices.routing import AutoRouter
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

    def check_stall() -> None:
        """Poll for a stall and continue the user's point if one is armed.

        WHY ITS OWN TICK rather than riding check_routing's 3 s timer: the
        stall threshold is 4 s, so a 3 s poll would fire anywhere between 4 and
        7 s after the user trailed off - by which point the moment has passed
        and Vaani would be interrupting whatever came next.

        WHY A THREAD: drafting calls the LLM, which takes seconds on this
        hardware. Doing that on the Tk thread would freeze the overlay - and
        with it the AI-SPEAKING badge and the cancel path, which is precisely
        when the user needs both.
        """
        if continuation is not None and continuation.armed:
            ready, reason = continuation.should_continue()
            if ready:
                # Disarm first so this cannot re-fire while the draft is in
                # flight and stack two continuations on top of each other.
                continuation.disarm()
                from .session.continuation import ContinuationTrigger as _T

                def run() -> None:
                    outcome = _speak_continuation(_T.STALL)
                    text = outcome.get("continuation")
                    if text:
                        print(f"  [continued for you] {text}")
                    elif outcome.get("error"):
                        print(f"  [continue] {outcome['error']}", file=sys.stderr)

                threading.Thread(target=run, name="vaani-continue",
                                 daemon=True).start()
        root.after(500, check_stall)

    root.after(500, check_stall)

    print("  Control it from any terminal, or bind these to KDE shortcuts:")
    print("    vaani show     vaani hide     vaani toggle")
    print("    vaani pause    vaani mute     vaani panic")
    if takeover_rt is not None:
        print("    vaani takeover-arm   takeover-now   takeover-stop")
    print("=" * 66)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        control.stop()
        if remote_listener is not None:
            # Stopped FIRST: its thread holds a capture stream and calls into
            # the runtime, so tearing the session down underneath it would race.
            remote_listener.stop()
        if takeover_rt is not None:
            takeover_rt.stop()
        if ptt_listener is not None:
            # Stopped before the session so its callbacks cannot fire into a
            # half-torn-down session.
            ptt_listener.stop()
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


def cmd_takeover(args) -> int:
    """Headless takeover: `MeetingTakeoverApp`, reachable from the CLI at last.

    AC-21.7. This used to be invokable only as
    `python -m vaani.session.meeting_takeover_app`, which is why nobody could
    find it. Prefer `vaani meeting --takeover`, which additionally gives you the
    overlay, the control socket and automatic mic routing; this is for headless
    use and is the same runtime the Windows console drives.
    """
    import time as _time

    from .session.meeting_takeover_app import MeetingTakeoverApp

    if args.input and args.remote_input == args.input:
        print(f"ERROR: --remote-input and --input are the same device "
              f"({args.remote_input!r}); Vaani would hear you as the other "
              "party.", file=sys.stderr)
        return 2

    # Resolve the LLM up front so a missing model is a clear error here rather
    # than a stack trace several seconds into MeetingTakeoverApp's constructor.
    from .ai.model_detect import resolve_llm_model
    choice = resolve_llm_model(pinned=args.llm_model)
    if not choice.ok:
        print(f"ERROR: {choice.describe()}", file=sys.stderr)
        return 2
    print(f"  LLM: {choice.describe()}")

    try:
        app = MeetingTakeoverApp(
            input_device=args.input, remote_input_device=args.remote_input,
            output_device=args.output_device, model=args.model,
            llm_model=choice.name, voice=args.voice,
            performance_mode=PerformanceMode(args.mode))
    except Exception as exc:
        print(f"ERROR: could not start takeover: {exc}", file=sys.stderr)
        return 2

    print("Vaani takeover armed. Say 'Vaani, take over'. Ctrl+C to stop.")
    try:
        app.start()
        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
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
    # WHY THESE ECHO: a control command that prints nothing is
    # indistinguishable from one that silently failed. That matters most for
    # `continue`, where the honest outcome is usually a REFUSAL ("not enough
    # context to continue a point") - and a user who sees no output concludes
    # the feature is broken rather than that Vaani declined on purpose.
    if args.command in ("continue", "continue-arm", "continue-off", "cancel"):
        if "continuation" in reply:
            text = reply.get("continuation")
            if text:
                print(f"spoke for you: {text}")
            else:
                print(f"declined: {reply.get('reason', 'no reason given')}")
        elif reply.get("cancelled"):
            print("cancelled - Vaani has stopped speaking for you")
        elif "continue_armed" in reply:
            print("continuations ARMED - Vaani may finish your point when "
                  "you stall" if reply["continue_armed"]
                  else "continuations off")
        return 0

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
        # The most important line for someone mid-meeting: BYPASS means the
        # room is hearing raw Hindi, MUTED means it is hearing nothing.
        mode = reply.get("input_mode")
        if mode:
            hint = {"BYPASS": "  <-- your REAL voice, untranslated",
                    "MUTED": "  <-- nothing is being sent"}.get(mode, "")
            print(f"input      {mode}{hint}")
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
    # Global, so every subcommand inherits it. DEBUG is what to ask a user for
    # after a mid-meeting failure; the file handler rotates at 5 MiB x 3, so
    # leaving it on does not fill the disk.
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="file and console log verbosity (default: INFO)")
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
    # WHY always_on is the default: in a normal meeting the user just talks.
    # ptt_hold needs evdev access (the `input` group), so it is opt-in and
    # never the default on a fresh install.
    p.add_argument("--input-mode", default=InputMode.ALWAYS_ON.value,
                   choices=[m.value for m in InputMode],
                   help="always_on (default), ptt_toggle, or ptt_hold")
    p.add_argument("--ptt-key", default="KEY_F8",
                   help="key to hold in ptt_hold mode (default: KEY_F8). "
                        "Needs the 'evdev' package and membership of the "
                        "'input' group; see README.")
    # D-3 FIX: default None = auto-detect (F-22). The old default was
    # `qwen3-coder:latest`, which is (a) not installed on this machine and (b)
    # the build this repo MEASURED getting dates wrong. Pass a name to pin one.
    p.add_argument("--llm-model", default=None,
                   help="Ollama model for romanised Hinglish and takeover "
                        "answers. Default: auto-detect the best installed one.")
    p.add_argument("--show", action="store_true",
                   help="start with the overlay visible (default: hidden)")
    p.add_argument("--corner", default="bottom-right",
                   choices=["top-left", "top-right", "bottom-left", "bottom-right"])
    p.add_argument("--no-auto-route", dest="auto_route", action="store_false",
                   help="do not move meeting apps onto the virtual mic "
                        "automatically; switch the microphone yourself instead")
    # ---- F-21 takeover, opt-in ----
    p.add_argument("--takeover", action="store_true",
                   help="also listen to the other participants so Vaani can "
                        "answer in your voice when you freeze. Requires "
                        "--remote-input. Starts OFF; arm it separately.")
    p.add_argument("--remote-input",
                   help="monitor source carrying the MEETING's audio (the other "
                        "side). See `vaani devices --all`.")
    p.add_argument("--arm", action="store_true",
                   help="arm takeover immediately instead of waiting for "
                        "`vaani takeover-arm`")
    p.add_argument("--takeover-silence", type=float, default=4.0,
                   help="seconds of silence after a question before Vaani may "
                        "answer (default 4.0)")
    p.set_defaults(func=cmd_meeting, auto_route=True)

    # Standalone takeover entry point (AC-21.7). `vaani meeting --takeover` is
    # the one to use -- it has the overlay, the control socket and auto-routing.
    # This exists because MeetingTakeoverApp is also the Windows console's
    # runtime, and because it is the only way to run takeover headless.
    p = sub.add_parser("takeover",
                       help="takeover-only mode, no overlay (prefer: "
                            "`vaani meeting --takeover`)")
    p.add_argument("--input", help="your physical microphone device name")
    p.add_argument("--remote-input", required=True,
                   help="monitor source carrying the meeting's audio")
    p.add_argument("--output-device",
                   help="Windows virtual-cable playback endpoint")
    p.add_argument("--model", default="small", help="Whisper model size")
    p.add_argument("--llm-model", default=None,
                   help="Ollama model (default: auto-detect)")
    p.add_argument("--voice", default="personal",
                   choices=["personal", "fallback"])
    p.add_argument("--mode", default="balanced",
                   choices=[m.value for m in PerformanceMode])
    p.set_defaults(func=cmd_takeover)

    for name, helptext in (("show", "show the overlay"),
                           ("hide", "hide the overlay (translation keeps running)"),
                           ("toggle", "toggle the overlay"),
                           ("pause", "pause translation"),
                           ("resume", "resume translation"),
                           ("mute", "stop capturing audio"),
                           ("unmute", "resume capturing audio"),
                           ("panic", "emergency stop and exit"),
                           ("status", "show the running session's status"),
                           # F-21 control verbs. Same KDE-shortcut pattern as
                           # the rest -- see ipc/control.py for why a socket and
                           # not a hotkey.
                           ("takeover-arm",
                            "allow Vaani to answer for you when you freeze"),
                           ("takeover-now",
                            "answer the pending question right now"),
                           ("takeover-stop",
                            "stop Vaani speaking for you (always works)"),
                           # Phase 2 meeting controls. Bind these to desktop
                           # shortcuts; they need no special permissions
                           # because they go through the control socket rather
                           # than grabbing the keyboard.
                           ("bypass",
                            "toggle YOUR OWN voice through, untranslated"),
                           ("bypass-on", "pass your real voice through"),
                           ("bypass-off", "go back to translated speech"),
                           ("talk", "push-to-talk: toggle whether you are live"),
                           ("mode-always-on",
                            "translate everything you say (default)"),
                           ("mode-ptt-toggle",
                            "translate only after `vaani talk`"),
                           ("mode-ptt-hold",
                            "translate only while the talk key is held"),
                           # Phase 4 Takeover A.
                           ("continue",
                            "finish YOUR point in your voice, right now"),
                           ("continue-arm",
                            "let Vaani finish your point when you stall"),
                           ("continue-off", "stop automatic continuations"),
                           ("cancel",
                            "STOP Vaani speaking for you, immediately")):
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

    p = sub.add_parser("models",
                       help="show which models Vaani would use, and why")
    p.add_argument("--probe", action="store_true",
                   help="actually run each candidate (slow: loads each model)")
    p.add_argument("--model", default=None,
                   help="check a specific model instead of auto-detecting")
    p.set_defaults(func=cmd_models)

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
    # D-3: same auto-detect default as `meeting`. All three entry points agree.
    p.add_argument("--llm-model", default=None,
                   help="Ollama model used for romanised Hinglish "
                        "(default: auto-detect the best installed one)")
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

    # WHY HERE: system/logging.py has had a RotatingFileHandler (5 MiB x 3)
    # since before v1.0, but setup_logging() was never called from anywhere -
    # so the rotating log was dead code and a 30-minute session left no
    # diagnosable trace behind. This is the one place every subcommand passes
    # through, and it runs AFTER parse_args so that `--help` and usage errors
    # stay instant and do not touch the filesystem.
    # WHERE THE DATA FLOWS NEXT: ~/.local/share/vaani/logs/vaani.log, which is
    # what a user should attach when reporting a mid-meeting failure.
    try:
        setup_logging(level=getattr(args, "log_level", "INFO"))
    except Exception as exc:        # noqa: BLE001
        # Logging must never be the reason Vaani refuses to start: a
        # read-only or full home directory is annoying, not fatal.
        print(f"warning: could not open the log file ({exc}); "
              f"continuing without file logging", file=sys.stderr)

    # E-1, enforced at parse time so the failure arrives instantly rather than
    # after 30 s of model loading. `ap.error` exits 2, which is the conventional
    # code for a usage error and is what the tests assert on.
    if getattr(args, "takeover", False) and not getattr(args, "remote_input", None):
        ap.error("--takeover requires --remote-input (the monitor source "
                 "carrying the meeting's audio). List devices with: "
                 "vaani devices --all")

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
