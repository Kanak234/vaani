#!/usr/bin/env python
"""Long-session soak test: prove the session does not leak.

WHY THIS EXISTS
---------------
Phase 1 requires a session that runs 30+ minutes without a memory leak. Unit
tests can assert that ``len(session.results)`` is capped, but a cap on one list
is not the same claim as "the process does not grow". Only a long run with real
RSS sampling can support that, so this script makes the measurement and applies
an explicit pass rule instead of leaving the reader to eyeball a graph.

WHAT IT MEASURES
----------------
Resident set size (RSS) of this process, sampled on a fixed interval while a
TranslationSession is driven with synthetic utterances at a realistic meeting
cadence. RSS is the honest number here: it is what the OS says Vaani is
actually holding, and it is what makes a 4 GB laptop start swapping.

THE PASS RULE
-------------
Compare median RSS in the WARMUP-to-settle window against median RSS over the
final window, and require growth <= --max-growth-pct (default 10%).

WHY MEDIANS OF WINDOWS rather than first vs last sample: a single sample
catches whatever GC and the allocator happened to be doing at that instant,
which is noise of tens of MB. WHY SKIP THE FIRST MINUTES: lazy imports, arena
growth and provider warmup all inflate early RSS, so measuring from t=0 would
compare a cold process with a warm one and report growth that is really just
startup. The baseline is therefore taken AFTER --settle-s.

FAKE PROVIDERS BY DEFAULT
-------------------------
The default run uses stub STT/MT and the fallback synthesiser, so it exercises
the session machinery -- queues, the results window, the supervisor, the
threading -- without model download time or a GPU. That is deliberate: this
harness is testing VAANI's memory behaviour, not PyTorch's. A leak in the
session shows up here; a leak inside Whisper would need --real-providers, which
is left as the on-hardware check in docs/TASK_PLAN.md.

CHECKPOINTING
-------------
Samples are appended to the JSON report as they are taken, not held until the
end. The desktop machine this targets is known to crash under long sustained
load, and a soak run that loses all its data on a crash at minute 28 is worse
than useless. A partial report is still evidence.

USAGE
-----
    python scripts/soak_test.py --minutes 30
    python scripts/soak_test.py --minutes 2 --out /tmp/soak.json   # smoke

WHERE THE DATA FLOWS NEXT
-------------------------
The JSON report is the evidence cited by docs/09-ACCEPTANCE-STATUS.md for
NFR-4/NFR-5. Exit code 0 = pass, 1 = fail, 2 = could not run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Make `src/` importable when run straight from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vaani.ai.tts.fallback import FallbackSynthesizer          # noqa: E402
from vaani.core.types import Transcript, Translation, Utterance  # noqa: E402
from vaani.session.engine import SessionConfig, TranslationSession  # noqa: E402

SR = 16000


def rss_mib() -> float:
    """Current resident set size in MiB.

    psutil when present; /proc otherwise. WHY THE FALLBACK: this script must
    run on a bare install.sh venv, and refusing to measure because an optional
    dependency is missing would defeat the point of having the harness.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        # /proc/self/statm field 2 is resident pages.
        with open("/proc/self/statm") as fh:
            resident_pages = int(fh.read().split()[1])
        import resource
        return resident_pages * resource.getpagesize() / (1024 * 1024)


class StubRecognizer:
    """Fixed Hinglish transcript. Costs no model, so the loop measures the
    session rather than Whisper."""
    key = "soak_stt"

    def transcribe(self, audio, sample_rate, *, language_hint=None, context_prompt=None):
        return Transcript(text="मैं कल मीटिंग में आऊंगा",
                          language_distribution={"hi": 1.0}, confidence=0.92)


class StubTranslator:
    key = "soak_mt"

    def translate(self, text, *, source_language, target_language, context=None):
        return Translation(text="I will come to the meeting tomorrow.",
                           source_language=source_language,
                           target_language=target_language, confidence=0.92)


class StubVad:
    def is_speech(self, frame, sample_rate):
        return True

    def reset(self):
        pass


def make_utterance(seq: int, seconds: float) -> Utterance:
    # Non-zero audio: a zero array can be special-cased by allocators and by
    # any future silence check, which would make the measurement dishonest.
    samples = (np.random.default_rng(seq).standard_normal(
        int(SR * seconds)) * 0.05).astype(np.float32)
    return Utterance(session_id="soak", seq=seq, audio=samples, sample_rate=SR,
                     speech_end_time=time.monotonic(), speech_ms=seconds * 1000)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=30.0,
                    help="how long to run (default: 30, the Phase 1 target)")
    ap.add_argument("--utterance-every-s", type=float, default=4.0,
                    help="cadence of synthetic speech (default: 4 s, a talkative meeting)")
    ap.add_argument("--utterance-seconds", type=float, default=3.0)
    ap.add_argument("--sample-every-s", type=float, default=10.0)
    ap.add_argument("--settle-s", type=float, default=120.0,
                    help="ignore RSS before this point; warmup is not a leak")
    ap.add_argument("--max-growth-pct", type=float, default=10.0)
    ap.add_argument("--max-retained-results", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("soak_report.json"))
    args = ap.parse_args()

    total_s = args.minutes * 60.0
    if args.settle_s >= total_s:
        # Refuse rather than silently comparing a window against itself.
        print(f"ERROR: --settle-s ({args.settle_s:.0f}s) must be less than the "
              f"run length ({total_s:.0f}s). For a short smoke run pass a "
              f"smaller --settle-s, e.g. --minutes 2 --settle-s 20.",
              file=sys.stderr)
        return 2

    session = TranslationSession(
        recognizer=StubRecognizer(), translator=StubTranslator(),
        synthesizer=FallbackSynthesizer(), vad=StubVad(),
        config=SessionConfig(use_virtual_mic=False,
                             max_retained_results=args.max_retained_results))

    report: dict = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": vars(args) | {"out": str(args.out)},
        "samples": [],
        "status": "running",
    }

    def checkpoint() -> None:
        # Written on every sample - see the module docstring on why a partial
        # report beats losing 28 minutes of data to a crash.
        args.out.write_text(json.dumps(report, indent=2, default=str))

    print(f"Soak: {args.minutes:.0f} min, utterance every {args.utterance_every_s}s, "
          f"sampling RSS every {args.sample_every_s}s")
    print(f"Pass rule: median RSS after settle vs final window, growth <= "
          f"{args.max_growth_pct}%\n")
    print(f"{'t(s)':>7} {'RSS(MiB)':>10} {'utts':>7} {'retained':>9} {'failed':>7}")

    started = time.monotonic()
    next_utterance = started
    next_sample = started
    seq = 0

    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            if elapsed >= total_s:
                break

            if now >= next_utterance:
                seq += 1
                # Driven through the supervised path, exactly as the worker
                # thread does, so containment and the results window are both
                # exercised rather than bypassed.
                session.supervisor.run(session._process_one,
                                       make_utterance(seq, args.utterance_seconds))
                next_utterance += args.utterance_every_s

            if now >= next_sample:
                sample = {
                    "t_s": round(elapsed, 1),
                    "rss_mib": round(rss_mib(), 2),
                    "utterances": session.total_utterances,
                    "retained": len(session.results),
                    "failed": session.failed_utterances,
                    "dropped_audio_blocks": session.dropped_audio_blocks,
                }
                report["samples"].append(sample)
                checkpoint()
                print(f"{sample['t_s']:>7.0f} {sample['rss_mib']:>10.2f} "
                      f"{sample['utterances']:>7} {sample['retained']:>9} "
                      f"{sample['failed']:>7}")
                next_sample += args.sample_every_s

            # Sleep to the next scheduled event rather than spinning, so the
            # harness itself does not distort what it is measuring.
            time.sleep(max(0.01, min(next_utterance, next_sample) - time.monotonic()))
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        checkpoint()
        print("\nInterrupted; partial report written.", file=sys.stderr)
        return 2

    # ----------------------------------------------------------- the verdict
    settled = [s for s in report["samples"] if s["t_s"] >= args.settle_s]
    if len(settled) < 4:
        report["status"] = "inconclusive"
        report["reason"] = (f"only {len(settled)} samples after settle; need >= 4 "
                            f"to compare two windows")
        checkpoint()
        print(f"\nINCONCLUSIVE: {report['reason']}", file=sys.stderr)
        return 2

    window = max(2, len(settled) // 4)
    baseline = statistics.median(s["rss_mib"] for s in settled[:window])
    final = statistics.median(s["rss_mib"] for s in settled[-window:])
    growth_pct = ((final - baseline) / baseline) * 100.0 if baseline else 0.0
    passed = growth_pct <= args.max_growth_pct

    report |= {
        "status": "pass" if passed else "fail",
        "baseline_rss_mib": round(baseline, 2),
        "final_rss_mib": round(final, 2),
        "growth_pct": round(growth_pct, 2),
        "peak_rss_mib": round(max(s["rss_mib"] for s in report["samples"]), 2),
        "window_samples": window,
        "total_utterances": session.total_utterances,
        "failed_utterances": session.failed_utterances,
        "retained_results": len(session.results),
    }
    checkpoint()

    print(f"\n{'PASS' if passed else 'FAIL'}: RSS {baseline:.1f} -> {final:.1f} MiB "
          f"({growth_pct:+.2f}%, limit {args.max_growth_pct}%)")
    print(f"peak {report['peak_rss_mib']:.1f} MiB over "
          f"{session.total_utterances} utterances; "
          f"retained {len(session.results)} results")
    print(f"report: {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
