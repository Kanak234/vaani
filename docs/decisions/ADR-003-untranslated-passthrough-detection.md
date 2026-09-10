# ADR-003 — The gate must detect untranslated output, not just wrong script

**Status:** Accepted · **Date:** 2026-09-09 · **Found by:** `scripts/eval_translation.py`

## Context

The confidence gate's job is to keep the system silent rather than let it say
something wrong in the user's own voice. It originally checked English output for
Devanagari characters: if Hindi script survived into the "English" translation, the
engine had clearly passed the input through untranslated (AC-06.3).

That check is necessary but nowhere near sufficient.

## The problem, measured

Running NLLB-200 against the Hinglish fixtures produced this:

```
[PASS] (romanised/commitment) conf=0.86
   in : haan bilkul, main kal call kar lunga
   out: Haan bilkul, main kal call kar lunga.
```

The output is the input, verbatim, **at confidence 0.86**, and the gate allowed it.
Romanised Hindi is written in Latin script, so there is no Devanagari to detect.
In a real meeting this would have spoken untranslated Hindi to an English audience,
in the user's own cloned voice, with every health indicator green.

**Worse, the test fixture reported this as a PASS.** Its assertion was
`must_contain=("call",)`, and the untranslated Hindi sentence contains the English
loanword "call". Overall accuracy read **93 %** when the true figure was **86 %**.
A fixture that can pass on completely untranslated output is worse than no fixture,
because it manufactures confidence.

## Decision

Two additional checks in `check_translation`, plus a fixture correction.

**1. Verbatim echo.** If the normalised output equals the normalised input, nothing
was translated.

**2. Romanised-Hindi markers.** A word-boundary regex of Hindi function and
auxiliary words. Suppression requires **two distinct markers AND a density above
15 %**, because one marker alone appears in ordinary English.

**3. Fixture correction.** The romanised cases now carry `must_not_contain`
assertions naming the actual input tokens, so an untranslated echo cannot pass.

## The fix that broke something worse

The first version of the verbatim check applied to all English output. It
immediately suppressed:

```
SUPPRESSED [english] I will complete the integration by Friday afternoon.
   why: output is identical to the input; nothing was translated
```

For English input, identical output is **the correct result** — the pass-through
engine is supposed to return what the user said. As first written, the fix would
have **silenced every English utterance in the product**.

The verbatim check now applies only when `source_language != target_language`.

The marker list had the same shape of bug: it originally included `the`, which is a
Hindi word (*were*) and the most common word in English. It suppressed the
perfectly good sentence "The Ki framework handles this well." The list now
deliberately **excludes** every Hindi function word that collides with a common
English one — `the`, `hi`, `ho`, `main`, `par`, `se`, `ka`, `ke`, `ki`, `to`, `ya`.

**The asymmetry that governs both decisions:** false-suppressing correct speech
costs the user an utterance they said properly and will have to repeat. Missing
some romanised Hindi costs one bad utterance. Both are bad, but the first is worse
because it fires on the common case, so the checks are tuned to be conservative.

## Verification

Final measured results, with the corrected gate and honest fixtures:

| Register | Pass | Behaviour |
|---|---|---|
| Devanagari | 4/4 (100%) | spoken |
| Hinglish (code-mixed) | 6/6 (100%) | spoken |
| English | 2/2 (100%) | spoken, not falsely suppressed |
| Romanised Hindi | 0/2 (0%) | **suppressed, not spoken wrongly** |

Overall 12/14 (86 %), 2 suppressions, both correct. 21 gate unit tests cover both
directions, including the two regressions above.

## Consequences

- Romanised Hindi currently produces **silence**. That is the designed behaviour
  and it is honest, but it is a real capability gap: a user who speaks mostly
  romanised Hinglish will be suppressed often. This is now the strongest argument
  for adding a cloud LLM provider, and a much more specific one than "code-mixing
  is hard".
- The marker list is a heuristic and will need extending. It must only ever be
  extended with words that are **not** common English.
- Every future translation engine inherits these checks for free — they live in the
  gate, not in the provider.
