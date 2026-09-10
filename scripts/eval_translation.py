"""Evaluate a translation engine against the Hinglish fixtures.

Reports per-register and per-group accuracy. The `commitment` group is called out
separately because those are the failures that cause real professional harm.

Usage: eval_translation.py [--engine nllb|passthrough]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "tests/fixtures")

from hinglish_cases import CASES                                   # noqa: E402
from vaani.core.gate import ConfidenceGate, GateConfig             # noqa: E402
from vaani.core.types import Transcript                            # noqa: E402


def check(case, output: str) -> tuple[bool, str]:
    low = output.lower()
    for token in case.must_contain:
        if token.lower() not in low:
            return False, f"missing {token!r}"
    for token in case.must_not_contain:
        if token.lower() in low:
            return False, f"contains forbidden {token!r}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="nllb",
                    choices=["nllb", "passthrough", "ollama", "routing"])
    ap.add_argument("--model", default=None, help="ollama model override")
    args = ap.parse_args()

    if args.engine == "routing":
        from vaani.ai.translate.nllb import NllbTranslator
        from vaani.ai.translate.ollama import OllamaTranslator
        from vaani.ai.translate.router import RoutingTranslator
        engine = RoutingTranslator(
            fast=NllbTranslator(),
            accurate=OllamaTranslator(model=args.model or "qwen3-coder:latest"))
    elif args.engine == "ollama":
        from vaani.ai.translate.ollama import OllamaTranslator
        engine = (OllamaTranslator(model=args.model) if args.model
                  else OllamaTranslator())
    elif args.engine == "nllb":
        from vaani.ai.translate.nllb import NllbTranslator
        engine = NllbTranslator()
    else:
        from vaani.ai.translate.passthrough import PassthroughTranslator
        engine = PassthroughTranslator()

    t0 = time.perf_counter()
    engine.warmup()
    print(f"engine: {engine.key} (warmup {time.perf_counter()-t0:.1f}s)")
    print("=" * 96)

    gate = ConfidenceGate(GateConfig())
    by_group, by_register = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    latencies, gated = [], 0

    for case in CASES:
        src = "en" if case.register == "english" else "hi"
        t0 = time.perf_counter()
        try:
            result = engine.translate(case.text, source_language=src,
                                      target_language="en",
                                      context=list(case.context) or None)
            out, conf = result.text, result.confidence
        except Exception as exc:
            out, conf = f"<ERROR: {type(exc).__name__}>", 0.0
        latencies.append((time.perf_counter() - t0) * 1000)

        ok, reason = check(case, out)
        decision = gate.check_translation(
            type(result)(text=out, source_language=src, target_language="en",
                         confidence=conf) if "ERROR" not in out else result,
            Transcript(text=case.text, language_distribution={src: 1.0},
                       confidence=0.9),
        ) if "ERROR" not in out else None
        suppressed = bool(decision and decision.suppressed)
        gated += suppressed

        by_group[case.group][1] += 1
        by_register[case.register][1] += 1
        if ok:
            by_group[case.group][0] += 1
            by_register[case.register][0] += 1

        mark = "PASS" if ok else ("GATED" if suppressed else "FAIL ")
        print(f"[{mark}] ({case.register}/{case.group}) conf={conf:.2f}")
        print(f"   in : {case.text[:82]}")
        print(f"   out: {out[:82]}")
        if not ok:
            print(f"   why: {reason}"
                  + ("  -- correctly suppressed by the gate" if suppressed else ""))
        print()

    print("=" * 96)
    print(f"{'BY REGISTER':<22}{'pass/total':>14}")
    for k, (good, total) in sorted(by_register.items()):
        print(f"  {k:<20}{good:>6}/{total:<6}  {good/total*100:5.0f}%")
    print(f"\n{'BY GROUP':<22}{'pass/total':>14}")
    for k, (good, total) in sorted(by_group.items()):
        flag = "  <-- the ones that cause real harm" if k == "commitment" else ""
        print(f"  {k:<20}{good:>6}/{total:<6}  {good/total*100:5.0f}%{flag}")

    total_good = sum(g for g, _ in by_group.values())
    total_all = sum(t for _, t in by_group.values())
    lat = sorted(latencies)
    if hasattr(engine, "route_counts"):
        print(f"\nROUTING: {engine.route_counts}")
    print(f"\nOVERALL {total_good}/{total_all} ({total_good/total_all*100:.0f}%)")
    print(f"suppressed by the gate: {gated}")
    print(f"latency p50 {lat[len(lat)//2]:.0f}ms  max {lat[-1]:.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
