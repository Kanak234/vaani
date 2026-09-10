"""Guided validation with the real user's voice and speech (Task 1).

Nothing in this project has been tested on the person it was built for. Every STT
figure comes from one public English clip; every voice-cloning figure used a
stand-in speaker. This script closes that gap and writes docs/10-USER-VALIDATION.md
with whatever it actually measures — including failures.

It is deliberately interactive: the measurements that matter here cannot be
synthesised.

Usage: PYTHONPATH=src .venv/bin/python scripts/validate_with_user.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from vaani.ai.stt.whisper import FasterWhisperRecognizer      # noqa: E402
from vaani.ai.translate.nllb import NllbTranslator            # noqa: E402
from vaani.ai.translate.ollama import OllamaTranslator        # noqa: E402
from vaani.ai.translate.router import RoutingTranslator, is_romanised_hindi  # noqa: E402
from vaani.ai.vad.silero import SileroVad                     # noqa: E402
from vaani.audio.backend.pulse_backend import PulseCaptureStream  # noqa: E402
from vaani.audio.segmenter import SegmenterConfig, UtteranceSegmenter  # noqa: E402
from vaani.core.gate import ConfidenceGate, GateConfig        # noqa: E402
from vaani.core.types import PerformanceMode, Transcript      # noqa: E402

SR, FRAME_MS = 16000, 20
FRAME_N = SR * FRAME_MS // 1000

#: Sentences chosen so each one tests something specific and checkable.
PROMPTS = [
    ("Devanagari, a date commitment",
     "हाँ, मेरा कहना ये है कि हम इस project को अगले हफ्ते तक complete कर सकते हैं।",
     ["next week"]),
    ("Code-mixed, natural register",
     "Actually मेरा point ये है कि deadline थोड़ी tight है।",
     ["deadline"]),
    ("Romanised Hinglish — say it the way you normally would",
     "haan bilkul, main kal call kar lunga",
     ["tomorrow", "call"]),
    ("A number that must survive",
     "मुझे तीन दिन चाहिए इस काम के लिए।",
     ["three"]),
    ("A negation that must survive",
     "नहीं, यह मेरी ज़िम्मेदारी नहीं है।",
     ["not"]),
    ("Plain English, for comparison",
     "I will complete the integration by Friday afternoon.",
     ["friday"]),
]


def record_utterance(capture, vad, segmenter, max_seconds=15.0):
    """Capture until the segmenter closes an utterance, or time out."""
    vad.reset()
    segmenter.reset()
    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        frame = capture.read_frame()
        segment = segmenter.push(frame, vad.is_speech(frame))
        if segment is not None:
            return segment
    return segmenter.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="capture device key")
    ap.add_argument("--out", default="docs/10-USER-VALIDATION.md")
    args = ap.parse_args()

    print("=" * 74)
    print("VAANI — VALIDATION WITH YOUR OWN VOICE")
    print("=" * 74)
    print("You will be shown a sentence. Press Enter, say it naturally, then stop.")
    print("Say it the way you actually would in a meeting, not carefully.\n")

    print("Loading models (this takes a moment)...")
    recognizer = FasterWhisperRecognizer.for_mode(PerformanceMode.BALANCED)
    recognizer.warmup()
    translator = RoutingTranslator(fast=NllbTranslator(),
                                   accurate=OllamaTranslator())
    translator.warmup()
    gate = ConfidenceGate(GateConfig())
    vad = SileroVad()
    segmenter = UtteranceSegmenter(config=SegmenterConfig.for_mode(
        PerformanceMode.BALANCED), sample_rate=SR, frame_ms=FRAME_MS)
    capture = PulseCaptureStream(device=args.input, sample_rate=SR,
                                 frame_ms=FRAME_MS, stream_name="validate")
    print(f"  STT: {recognizer.model_size} on {recognizer.device_used}\n")

    rows = []
    try:
        for i, (label, sentence, expect) in enumerate(PROMPTS, 1):
            print("-" * 74)
            print(f"[{i}/{len(PROMPTS)}] {label}")
            print(f"    SAY: {sentence}")
            input("    Press Enter, then speak... ")
            print("    listening...")

            segment = record_utterance(capture, vad, segmenter)
            if segment is None or segment.audio.size == 0:
                print("    no speech detected — skipping")
                rows.append({"label": label, "expected": sentence, "status": "no speech"})
                continue

            t0 = time.perf_counter()
            try:
                transcript = recognizer.transcribe(segment.audio, SR)
            except Exception as exc:
                print(f"    STT failed: {exc}")
                rows.append({"label": label, "expected": sentence,
                             "status": f"stt failed: {exc}"})
                continue
            stt_ms = (time.perf_counter() - t0) * 1000

            decision = gate.check_transcript(transcript)
            t0 = time.perf_counter()
            translated, route = "", "-"
            if decision.allowed:
                engine, route = translator.choose_engine(
                    transcript.text, transcript.dominant_language)
                try:
                    result = translator.translate(
                        transcript.text,
                        source_language=transcript.dominant_language,
                        target_language="en")
                    translated = result.text
                    tdec = gate.check_translation(result, transcript)
                except Exception as exc:
                    translated, tdec = f"<error: {exc}>", None
            else:
                tdec = None
            mt_ms = (time.perf_counter() - t0) * 1000

            hit = [w for w in expect if w.lower() in translated.lower()]
            suppressed = (not decision.allowed) or (tdec is not None and tdec.suppressed)

            print(f"    heard      : {transcript.text}")
            print(f"    confidence : {transcript.confidence:.2f}   "
                  f"languages: {transcript.language_distribution}")
            print(f"    route      : {route}")
            print(f"    English    : {translated or '(suppressed)'}")
            print(f"    key terms  : {len(hit)}/{len(expect)} kept {hit}")
            print(f"    timing     : STT {stt_ms:.0f} ms   translate {mt_ms:.0f} ms")
            if suppressed:
                why = decision.detail or (tdec.detail if tdec else "")
                print(f"    SUPPRESSED : {why}")

            rows.append({
                "label": label, "expected": sentence,
                "heard": transcript.text, "confidence": transcript.confidence,
                "langs": transcript.language_distribution, "route": route,
                "english": translated, "kept": f"{len(hit)}/{len(expect)}",
                "stt_ms": stt_ms, "mt_ms": mt_ms, "suppressed": suppressed,
                "romanised": is_romanised_hindi(transcript.text),
                "status": "suppressed" if suppressed else "ok",
            })
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        capture.close()

    _write_report(Path(args.out), rows, recognizer)
    print("=" * 74)
    print(f"Report written to {args.out}")
    print("=" * 74)
    return 0


def _write_report(path: Path, rows: list[dict], recognizer) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    stt = [r["stt_ms"] for r in ok if "stt_ms" in r]
    confs = [r["confidence"] for r in ok if "confidence" in r]
    romanised = [r for r in ok if r.get("romanised")]

    lines = [
        "# User Validation — measured on the actual user's speech",
        "",
        f"**Date:** {datetime.now(UTC).date().isoformat()}  ",
        f"**STT:** {recognizer.model_size} on {recognizer.device_used}/"
        f"{recognizer.compute_type_used}",
        "",
        "Generated by `scripts/validate_with_user.py`. Every figure below came "
        "from the user speaking into the microphone — nothing here is synthetic.",
        "",
        "## Results",
        "",
        "| # | What it tests | Heard | English | Conf | Route | Kept | STT | MT |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        if r.get("status") in ("no speech",) or "heard" not in r:
            lines.append(f"| {i} | {r['label']} | — | — | — | — | — | — | "
                         f"{r.get('status','')} |")
            continue
        lines.append(
            f"| {i} | {r['label']} | {r['heard'][:40]} | "
            f"{(r['english'] or '(suppressed)')[:40]} | {r['confidence']:.2f} | "
            f"{r['route']} | {r['kept']} | {r['stt_ms']:.0f} ms | {r['mt_ms']:.0f} ms |")

    lines += ["", "## Summary", ""]
    if stt:
        lines += [
            f"- Utterances completed: **{len(ok)}/{len(rows)}**",
            f"- STT latency: p50 **{statistics.median(stt):.0f} ms**, "
            f"max {max(stt):.0f} ms",
            f"- STT confidence: mean **{statistics.mean(confs):.2f}**, "
            f"min {min(confs):.2f}",
            f"- Suppressed: **{sum(1 for r in rows if r.get('suppressed'))}**",
            f"- Detected as romanised Hinglish (routed to the LLM): "
            f"**{len(romanised)}/{len(ok)}**",
        ]
    else:
        lines.append("- No utterances completed. See the console output.")

    lines += [
        "",
        "## What this does and does not establish",
        "",
        "**Establishes:** recognition accuracy on this user's accent, how much of "
        "their natural speech is romanised rather than Devanagari (which decides "
        "how often the slower LLM route is used), and real per-stage latency.",
        "",
        "**Does not establish:** whether the synthesised voice *sounds like them*. "
        "That is NFR-7 and it requires their own judgement after enrolling a voice "
        "profile — run `vaani enroll record`, then `vaani run`, and listen.",
        "",
        "## Open questions to answer next",
        "",
        "1. If confidence is consistently low, the gate threshold (default 0.55) "
        "may need recalibrating for this speaker — see `core/gate.py`.",
        "2. If most utterances route to the LLM, the latency budget needs "
        "rechecking, since that path measured 1906 ms p50 in isolation.",
        "3. If enrollment was rejected, the thresholds in "
        "`voice/enrollment.py` (`EnrollmentCriteria`) need review — this already "
        "happened once during development.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
