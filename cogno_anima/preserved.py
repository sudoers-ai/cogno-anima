"""What a PRESERVED TERM is — one definition, read by the stage that produces the list and by
the stage that judges against it.

The NOUMENO's rewriter lists the terms it kept intact (``preserved_terms``); the SUPEREGO's
judge demands their exact reproduction (criterion #4) and its output backstop flags a mutation.
Three kinds are worth guarding — a FIGURE, an E-MAIL, a URL — because altering one silently
corrupts the answer. That definition used to live in the SUPEREGO alone (``_CRITICAL_TERM_RE``,
#172); the NOUMENO now needs half of it, so it lives here and both stages import it. A second
regex would be the divergence #172 closed, reopened one stage upstream.

Two of the three are EXACT TOKENS. An e-mail or a URL has no legitimate rewrite, so one that is
not in the contact's own words is the MODEL's and not the contact's — measured 3 of 3
(qwen3:8b, temperature=0) on the host's onboarding cassette (#954): the contact's address came
back with one character altered, the judge then rejected every reply carrying the RIGHT address
for "altering" it, and the contact got the failure sentence. A figure is NOT an exact token: the
rewriter normalises format on its way to English ("R$ 1.000,00" may come out "1000"), and a
preserved term is a VALUE, not a spelling (``superego._PRESERVED_IS_A_VALUE``). That is why
:func:`filter_preserved_terms` reads :data:`EXACT_TOKEN_RE` and never :data:`CRITICAL_TERM_RE`.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# The three kinds, as fragments, so the two compiled forms below are ONE definition.
_FIGURE = r"\d"
_EMAIL = r"@"
_URL = r"https?://"

#: A term worth a grounding backstop: a figure, an e-mail, a URL.
CRITICAL_TERM_RE = re.compile("|".join((_FIGURE, _EMAIL, _URL)), re.IGNORECASE)
#: The exact-token half of it: an e-mail, a URL — never a figure.
EXACT_TOKEN_RE = re.compile("|".join((_EMAIL, _URL)), re.IGNORECASE)

# "Typed verbatim" is a WHOLE-ADDRESS match, not a bare substring, and the difference is the
# measured family: at edit distance 1, a character dropped at either END of the local part
# ("na.silva@…" in "ana.silva@…") or of the domain ("…@example.com" in "…@example.com.br")
# leaves a substring of what was typed that is still not the address. So the character before
# the match may not continue an address (a word character or one of the connectors an address is
# made of) and the character after may not either — a ``.`` counts only when a label follows it,
# so the sentence-final period after an address or a URL is not part of it.
_BEFORE = r"(?<![\w.%+-])"
_AFTER = r"(?![\w-]|\.\w)"


def typed_verbatim(term: str, user_input: str) -> bool:
    """``term`` appears in ``user_input`` as a whole address, case included.

    Exact, because the two errors are not symmetric: a false DROP costs only the marking (the
    contact's own words are already in the judge's prompt, verbatim, under ``# User request``),
    while a false KEEP is the measured defect. Case-folding cannot rescue a changed character;
    it could only admit a normalisation the MODEL chose, after which the judge would demand the
    model's spelling over the contact's. There is nothing to buy with it.
    """
    term = term.strip()
    if not term:
        return False
    return re.search(_BEFORE + re.escape(term) + _AFTER, user_input) is not None


def filter_preserved_terms(terms: Iterable[Any], user_input: str) -> tuple[list[Any], int]:
    """Drop every preserved E-MAIL/URL the contact did not type; keep everything else as it came.

    Returns ``(kept, dropped)`` — the list the stage records and the COUNT of what it removed.
    The count is the record; the values never are (an address is PII, and the point of the
    drop is that the value was wrong). A figure, a name, a phrase, a non-string: untouched,
    whatever its spelling — only :data:`EXACT_TOKEN_RE` terms are compared, against the RAW
    input and never the rewrite, which carries the same mutation.
    """
    kept: list[Any] = []   # the model's JSON, as it came — the result model validates it
    dropped = 0
    for term in terms:
        if isinstance(term, str) and EXACT_TOKEN_RE.search(term) \
                and not typed_verbatim(term, user_input):
            dropped += 1
            continue
        kept.append(term)
    return kept, dropped
