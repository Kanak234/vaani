"""Translation test cases covering the registers this product must handle.

These are the acceptance fixtures for AC-06.3 and AC-06.6. `must_contain` and
`must_not_contain` are checked case-insensitively; they encode the parts of the
meaning that CANNOT be lost, rather than demanding an exact string, because there
are many valid English renderings of each sentence.

The critical group is `commitment` -- dates, numbers and yes/no. The primary
persona's stated failure mode is the tool mistranslating a commitment and forcing
a public correction, so these are the cases where suppression is strongly
preferred over a wrong answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    text: str
    register: str            # devanagari | hinglish | romanised | english
    group: str               # general | commitment | pronoun | technical
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    note: str = ""
    #: Prior turns, for engines that can use context.
    context: tuple[tuple[str, str], ...] = field(default=())


CASES: list[Case] = [
    # --- pure Devanagari, the easiest register -----------------------------
    Case("मेरा नाम कनक है और मैं एक सॉफ्टवेयर इंजीनियर हूँ।",
         "devanagari", "general",
         must_contain=("name", "engineer")),
    Case("हाँ, मेरा कहना ये है कि हम इस project को अगले हफ्ते तक complete कर सकते हैं।",
         "hinglish", "commitment",
         must_contain=("next week",),
         must_not_contain=("next month", "next year", "tomorrow"),
         note="THE motivating example from the brief. 'अगले हफ्ते' must not drift."),
    Case("यह काम कल तक हो जाएगा।",
         "devanagari", "commitment",
         must_contain=("tomorrow",),
         must_not_contain=("yesterday", "next week"),
         note="कल is both yesterday and tomorrow; tense must disambiguate."),
    Case("मुझे तीन दिन चाहिए इस काम के लिए।",
         "devanagari", "commitment",
         must_contain=("three",),
         must_not_contain=("two", "four", "thirty")),
    Case("नहीं, यह मेरी ज़िम्मेदारी नहीं है।",
         "devanagari", "commitment",
         must_contain=("not",),
         note="Negation must survive. Dropping 'not' inverts a commitment."),

    # --- code-mixed, the actual target register ----------------------------
    Case("Actually मेरा point ये है कि deadline थोड़ी tight है.",
         "hinglish", "general",
         must_contain=("deadline",),
         note="Mixed script mid-sentence; must not be split or half-translated."),
    Case("उसमें authentication वाला part अभी बाकी है।",
         "hinglish", "pronoun",
         must_contain=("authentication",),
         context=(("We completed the backend API yesterday.",
                   "We completed the backend API yesterday."),),
         note="PRD AC-06.2. 'उसमें' needs the prior turn. NLLB cannot do this."),
    Case("मैंने code review कर लिया है, बस deploy करना बाकी है।",
         "hinglish", "technical",
         must_contain=("review",),
         must_not_contain=("समीक्षा",)),
    Case("Team के साथ meeting Thursday को schedule कर दो।",
         "hinglish", "commitment",
         must_contain=("thursday",),
         must_not_contain=("tuesday", "wednesday", "friday")),
    Case("Database migration में कुछ issue आ रहा है।",
         "hinglish", "technical",
         must_contain=("migration",)),

    # --- romanised Hindi, the known weak spot ------------------------------
    Case("mera point ye hai ki hum kal tak finish kar sakte hain",
         "romanised", "commitment",
         must_contain=("tomorrow",),
         must_not_contain=("mera point", "ye hai", "kar sakte"),
         note="All-Latin but Hindi. Documented weakness of NLLB (TRD §8)."),
    Case("haan bilkul, main kal call kar lunga",
         "romanised", "commitment",
         must_contain=("call",),
         # Without these, the case passed while the output was the INPUT verbatim:
         # `must_contain=("call",)` matched the untranslated Hindi word. A fixture
         # that can pass on untranslated output is worse than no fixture.
         must_not_contain=("haan", "bilkul", "main kal", "kar lunga"),
         note="Romanised Hinglish. NLLB returns this verbatim at conf 0.86; "
              "the gate must catch it since there is no Devanagari to detect."),

    # --- already English, must pass through untouched ----------------------
    Case("I will complete the integration by Friday afternoon.",
         "english", "commitment",
         must_contain=("friday",),
         must_not_contain=("monday", "thursday")),
    Case("The authentication module needs more testing before deployment.",
         "english", "technical",
         must_contain=("authentication", "testing")),
]


def by_group(group: str) -> list[Case]:
    return [c for c in CASES if c.group == group]


def by_register(register: str) -> list[Case]:
    return [c for c in CASES if c.register == register]
