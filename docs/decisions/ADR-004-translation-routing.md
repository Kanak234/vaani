# ADR-004 — Route translation per-utterance instead of picking one engine

**Status:** Accepted · **Date:** 2026-09-09 · **Supersedes:** TRD §8's single-engine assumption

## Context

NLLB-200 600M was measured at 12/14 (86 %) on the Hinglish fixtures with a p50 of
87 ms. Its two failures were both **romanised Hindi** — Latin-script Hindi such as
`"haan bilkul, main kal call kar lunga"`, which it returns verbatim at confidence
0.86 because it looks like English to a model trained on monolingual pairs. The
gate suppresses those, so the user gets silence (ADR-003).

Separately, NLLB is sentence-level and cannot use conversational context, leaving
PRD acceptance criterion **AC-06.2 unmet by construction**.

A cloud LLM was the expected fix. Then `ollama` was found already running on this
machine, which changes the privacy calculus completely: it serves models over
`http://localhost:11434`, so **no data leaves the device** and `requires_network`
stays `False`. LLM-quality translation without weakening the local-only guarantee.

## Measurements

All on the 14 fixtures in `tests/fixtures/hinglish_cases.py`, temperature 0.

| Register | NLLB-200 | phi4-mini (2.5 GB) | qwen3-coder (18 GB) |
|---|---|---|---|
| Devanagari | **4/4** | 2/4 | **4/4** |
| Code-mixed | **6/6** | 4/6 | **6/6** |
| English | **2/2** | 2/2 | **2/2** |
| **Romanised** | **0/2** | **2/2** | **2/2** |
| **Overall** | 12/14 (86 %) | 10/14 (71 %) | **14/14 (100 %)** |
| **Latency p50** | **87 ms** | 595 ms | 1906 ms (max 5458 ms) |

`phi4-mini` was rejected outright, and its failures show *why small models are
dangerous here rather than merely weaker*:

```
नहीं, यह मेरी ज़िम्मेदारी नहीं है।   ->  "I don't know, Meri Jimmereen doesn't exist."
उसमें authentication वाला part अभी बाकी है।  ->  "That is Abhi Bakkie's responsibility..."
Team के साथ meeting Thursday को schedule कर दो।  ->  "...arrange a meeting for tomorrow?"
```

It hallucinated a person's name out of `अभी बाकी`, dropped a negation, and moved a
Thursday meeting to tomorrow — while reporting the same confidence as its correct
answers.

## The decision

Neither single engine is right. `qwen3-coder` is the most accurate but at 1906 ms
it pushes the pipeline to ~4194 ms, **over the 4000 ms balanced budget**. NLLB is
22× faster and already perfect on every register except one.

The two registers diverge on a property that is **detectable from the text before
either engine runs**: whether the input is romanised Hindi. So route per utterance:

```
Devanagari present, or plain English   ->  NLLB          (87 ms, already 100% there)
Latin-script Hindi markers detected    ->  LLM           (the only engine that works)
LLM unavailable                        ->  NLLB, then the gate suppresses
```

## Verification

```
BY REGISTER    devanagari 4/4 · english 2/2 · hinglish 6/6 · romanised 2/2
BY GROUP       commitment 8/8 · general 2/2 · pronoun 1/1 · technical 3/3
ROUTING        {'fast': 12, 'accurate': 2, 'fallback': 0}
OVERALL        14/14 (100%)   suppressed by the gate: 0
latency        p50 424 ms   max 2591 ms
```

**14/14 at a p50 of 424 ms** — the LLM's accuracy at roughly a quarter of its
latency cost, because only 2 of 14 utterances needed it.

Updated budget: `22 + 500 + 578 + 424 + 1188 = 2712 ms` against 4000 ms.

## Consequences

- **AC-06.2 is now satisfiable.** The LLM path receives prior turns and reports a
  truthful `used_context_turns`. NLLB still reports 0, which remains honest.
- **The router inherits the strictest privacy requirement of its engines.** Today
  both are local, so `requires_network` is False and local-only mode hides nothing.
  Pointing Ollama at a remote host flips the flag automatically.
- **Running the LLM costs VRAM that the rest of the pipeline then cannot use.**
  Measured: ollama holds ~2738 MiB of the 4096 MiB card, which pushes Whisper,
  NLLB and XTTS to CPU via `model_budget`. The LLM route is not free — it trades
  GPU headroom for accuracy, and the scheduler absorbs that automatically.
- **The marker list is shared in spirit with the gate's** (ADR-003) and carries the
  same constraint: only words that are *not* common English, or ordinary English
  gets routed to a 1906 ms engine.
- A false positive costs latency; a false negative costs a suppressed utterance.
  Both are acceptable, neither is silent — which is why routing is safe to
  automate.
