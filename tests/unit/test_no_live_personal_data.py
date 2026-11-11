"""No LIVE personal datum may enter this tree — and the allowlists are ENUMERATED here.

This repository is PUBLIC and Apache-licensed. A personal datum committed here is not
untidiness, it is disclosure: it ships to every clone, every mirror and every model that
scrapes GitHub, and it does so under a licence that invites redistribution.

**Why this guard exists.** A sweep on 2026-09-08 found one real person's mobile number used
as fixture input in two files of ``cognobench/`` — the same subscriber that ``cogno-host``
anonymised the same day on the private side. It arrived the way these always arrive: somebody
reproduced a live turn and pasted it, and the paste outlived the debugging session. The values
were anonymised (never deleted — see below); this exists so the next paste is caught by a
machine instead of by the next sweep.

**Anonymise, never delete.** The value of a fixture line is the MEASUREMENT it carries, and a
``git rm`` destroys it while a consistent synthetic preserves it. The form is kept — a phone
still looks like a phone — and the same entity gets the same fake across the ecosystem, so the
number this repo now uses is the one ``cogno-host`` chose for that subscriber. Anonymising a
fixture must not be the same act as breaking it: ``tests/unit/test_pii_detector.py`` and the
``ner_pii_*`` bench cases were re-run against the replacements and the detector's verdict is
byte-identical, which is the only thing that makes the swap safe.

**The allowlists are literal sets in this file, and that is the whole design.** A guard that
answered "is this number one of our contacts?" by querying the customer base would be a SECOND
copy of the problem — the repo would hold, or reach for, the very data it exists to keep out.
So the question it answers is the narrow one it can answer offline: *is this one of the
synthetics this repo has already agreed to use?* Anything else fails, including a real value
that belongs to nobody in particular. Adding an entry is a deliberate act, with a comment.

**What counts as a phone**, and why it is not "ten to thirteen digits": that would fire on
timestamps, ids, hashes and hex. A hit must be a Brazilian number by CONSTRUCTION — an
assigned area code (:data:`_DDD`), then either a 9-digit mobile opening with ``9`` or an
8-digit landline opening with ``2``-``5`` — optionally carrying the ``55`` country code and
the usual separators. The country code is stripped before comparison, so the same fake written
another way is one entry and not two: a value cannot slip through by being reformatted.

**The regex finds a SHAPE; the construction rules DECIDE.** This is inherited as a lesson, not
as a style. ``cogno-host``'s first cut spelled the mobile ``9`` into the pattern itself
(``9?[0-9]{4}``), which made the subscriber check unreachable — a mutation deleting that check
stayed green, because no input could ever arrive at it. Dead code inside a guard is a claim
the guard does not check. The pattern here therefore matches any 4-5 digit block and
:func:`phone_numbers_in` alone decides what is a number; both halves of that decision have a
mutation twin below.

**Enumerated with ``git ls-files``, not a walk of the disk.** An uncommitted file is not a
contract, and a scratch copy under ``tests/`` is not something this repo ever published. The
walk would also have to learn about ``.venv``, ``htmlcov`` and ``__pycache__``; the index
already knows.

────────────────────────────────────────────────────────────────────────────────────────────
**WHAT THIS DOES NOT CATCH.** Stated here so nobody reads more into a green than is in it.

* **A person's NAME.** This is measured, not assumed. ``cogno-host`` ran its own 115-name
  lexicon over its tree: **732 firings across 90 of 422 files**, and of the **5** real names
  that had to be removed it would have caught **1**. A first name has no shape that separates
  it from a persona, a fixture or half the prose in a Portuguese repo, and the only way to
  raise the recall is to LIST THE REAL NAMES in the repo — which is re-leaking them. That
  class is a human review rule, and it is written where reviewers look (``README.md``): *if
  you quote a turn, anonymise the NAME and keep the SENTENCE.* Changing the words of a
  MEASURED quotation turns it into a FABRICATED one, which is why the rule is anonymise, not
  rewrite.
* **CPF/CNPJ.** A document number has exactly one structural property — its check digits — and
  every generator emits valid ones, so a real CPF and a synthetic CPF are indistinguishable by
  construction. A guard here could only be an allowlist with no filter in front of it, firing
  on every new legitimate fixture until somebody switched it off. The sweep found only
  canonical public test documents (``111.444.777-35``, ``11.222.333/0001-81`` and sequential
  placeholders) and no real one.
* **Addresses and postcodes.** Same reason, plus the sweep found only ``Rua das Flores, 123``
  (the Brazilian equivalent of Elm Street) and CEP ``01310-100``, which is Avenida Paulista —
  a landmark, not a residence.
* **Tenant/session ids.** No live id is present (the sweep checked the host's own tenant UUID
  explicitly); the three UUIDs here are the canonical RFC-4122 examples. An opaque id has no
  shape to key on, and enumerating live ids here would import them.
* **Test payment instruments.** ``4111 1111 1111 1111`` is Visa's published test number and
  belongs to nobody. A guard that flagged it would be a guard somebody switches off in week
  one, which costs more than it saves.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Area codes actually assigned in Brazil. A number whose DDD is not one of these is not a
# number, whatever its length.
_DDD = frozenset(
    str(d) for d in (
        *range(11, 20), 21, 22, 24, 27, 28, *range(31, 36), 37, 38,
        *range(41, 50), 51, 53, 54, 55, *range(61, 70),
        71, 73, 74, 75, 77, 79, *range(81, 90), *range(91, 100),
    )
)

# Shape only. Boundaries exclude an adjacent hyphen on both sides: separators are legal INSIDE
# a written number (``97000-1111``) and never at its edges, so the exclusion costs nothing and
# removes the one false positive worth removing — the digit groups of a UUID.
_RUN = re.compile(
    r"(?<![0-9A-Za-z-])"
    r"((?:\+?55)?[\s\-.()]*[1-9][0-9][\s\-.()]*[0-9]{4,5}[\s\-.()]*[0-9]{4})"
    r"(?![0-9A-Za-z-])"
)

# The synthetics this repo has agreed to use, in NATIONAL form (DDD + subscriber, no country
# code). Every entry is a number nobody answers. Adding one is a deliberate act — say which
# fixture needs it and why the value is safe.
ALLOWED_SYNTHETIC = frozenset({
    "11970001111",    # the ecosystem's example contact — the value cogno-host chose for this
                      # subscriber, reused here so one entity has one fake everywhere
    "11987654321",    # canonical fake, PII detector fixtures (also written +55 11 98765-4321)
    "11988887777",    # patterned placeholder
    "11999998888",    # patterned placeholder, redaction docstrings
})

# Mail domains this repo has agreed to use. RFC 2606 reserves example.com/.net/.org and the
# .test/.invalid/.example TLDs precisely so fixtures need not borrow somebody's real domain;
# the rest are enumerated one by one.
ALLOWED_EMAIL_DOMAINS = frozenset({
    "example.com", "aexample.com",   # RFC 2606, and the near-miss twin a detector test needs
    "exemplo.com",                   # its Portuguese spelling, NER/PII fixtures
    "empresa.com.br",                # "company" in Portuguese — safety bench input
    "x.com", "y.com", "b.com", "y.zz",   # one-letter stubs in parser/detector unit tests
})

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

# Binary/large blobs the index tracks but nobody pastes a phone number into.
_SKIP_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff",
                            ".woff2", ".ttf", ".zip", ".gz", ".whl", ".so", ".lock"})

# ──────────────────────────────────────────────────────────────────────────────────────
# OPAQUE LIVE IDENTIFIERS. The 2026-09-08 sweep anonymised phone numbers; this half catches
# the OTHER shape a pasted production turn leaves behind — the opaque ids that point AT a
# live record: a tenant/session scope, a persisted turn id, a long bare account/document run.
# They have no internal structure to validate the way a phone does, so the guard keys on the
# ways the repo WRITES them down, and every id it must tolerate is ENUMERATED here with a
# reason, exactly as the phone and e-mail allowlists are — the allowlist is the whole design.

# A TRUNCATED scope/tenant id: a hex run of 8+ digits closed by the ellipsis the prose uses
# to abbreviate it (an eight-hex prefix written hard against a trailing ellipsis — not spelled
# out here, because this very scan would then flag its own comment). A full opaque id has no
# shape to key on — the docstring says so above — but the ABBREVIATION does: nothing legitimate
# in this tree writes a hex run against that ellipsis. There is deliberately no allowlist; a
# truncated LIVE id is never something a fixture needs, so any hit is a leak to anonymise.
_SCOPE_PREFIX = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{8,}…")

# A persisted turn id, as the production tables are cited in design prose: the ``turn_traces``
# table NAME followed closely by the row number — ``turn_traces 1795``, ``turn_traces id=1440``,
# ``turn_traces` (1514 rows``. A bare number elsewhere has no shape and is not keyed on; the
# TABLE NAME is the shape. Every id the prose keeps as measurement provenance is enumerated.
_TURN_ID = re.compile(r"turn_traces[`\s(]*(?:id=)?([0-9]{3,})")

# Turn/row numbers the design prose cites as the evidence behind a rule. Each is a row of the
# rehearsal/demo box with NO tenant scope beside it (a scope IS caught, by ``_SCOPE_PREFIX``),
# kept so the measurement stays legible. A NEW id cannot slip in unreviewed; adding one is a
# deliberate act, with a reason, like every list here.
ALLOWED_TURN_IDS = frozenset({
    "1440",   # the canonical rejected read-only turn, cited across the judge-grounding files
    "1795",   # the derived-figure turn ("show your work"); grounding-of-derivations prose
    "1970",   # the exhaustion/November-reconstruction turn, turn 107
    "1969",   # its predecessor (turn 106), the October list that WAS read
    "1198",   # the "nothing was tried" expense specimen, execution-verdict prose
    "1691",   # the two-sanctioned-sources turn
    "448",    # a refused-write specimen (ok=False, side_effect=False) in the failure prose
    "1514",   # NOT an id but a row COUNT ("1514 rows"); same table-name shape, so enumerated
})

# Long bare digit runs of 11–13 that are NOT phones (the phone guard above owns those): a CPF,
# a GitHub Actions run id, a UUID node, a sequential placeholder. Hex-adjacent runs are
# excluded by construction (the boundaries below), so a digit slice of a sha256 never fires.
_LONG_NUMBER = re.compile(r"(?<![0-9A-Za-z])[0-9]{11,13}(?![0-9A-Za-z])")

ALLOWED_LONG_NUMBERS = frozenset({
    # Synthetic CPFs — document numbers nobody owns; redaction/PII fixtures.
    "52998224725",     # 529.982.247-25, redaction docstrings + PII provenance fixtures
    "11144477735",     # 111.444.777-35, canonical public test CPF, PII detector fixture
    # Synthetic phones, in the forms a long-digit scan sees them (national and +55).
    "11970001111", "11987654321", "11988887777", "11999998888",  # ALLOWED_SYNTHETIC, bare
    "5511999998888", "5511987654321",                             # same numbers, +55 prefix
    # GitHub Actions run ids — public CI identifiers, named in canary comments and CHANGELOG.
    "35207468672",     # nightly canary run of 2026-09-17 (superego.py + integration/unit tests)
    "35285170724",     # PR #167 canary run (superego.py + a grounding test)
    "37459138949",     # nightly run in the CHANGELOG xfail entry
    "33494160177",     # nightly run id behind the CI shard-split timing (.github/workflows)
    # Other public test constants.
    "426614174000",    # the node of RFC-4122 example UUID 123e4567-…-426614174000 (PIX fixtures)
    "123456789012",    # sequential placeholder integer, pre-judge digest test
})


def _national(digits: str) -> "str | None":
    """Country code stripped, so one number is one entry however it was written."""
    if digits.startswith("55") and len(digits) in (12, 13):
        return digits[2:]
    if len(digits) in (10, 11):
        return digits
    return None


def phone_numbers_in(text: str) -> "set[str]":
    """Every Brazilian phone number in ``text``, in national form.

    Pure, and the mutation twins call it directly — a detector that scanned nothing would
    otherwise pass the sweep by finding nothing."""
    out: set[str] = set()
    for match in _RUN.finditer(text):
        digits = re.sub(r"[^0-9]", "", match.group(1))
        national = _national(digits)
        if national is None:
            continue
        ddd, subscriber = national[:2], national[2:]
        if ddd not in _DDD:
            continue
        if len(subscriber) == 9 and not subscriber.startswith("9"):
            continue
        if len(subscriber) == 8 and subscriber[0] not in "2345":
            continue
        out.add(national)
    return out


def email_domains_in(text: str) -> "set[str]":
    """Every mail domain in ``text``, lowercased. Pure, for the same reason."""
    return {m.group(1).lower().rstrip(".") for m in _EMAIL.finditer(text)}


def truncated_scope_ids_in(text: str) -> "set[str]":
    """Every ``<hex>…`` truncated opaque id in ``text``. Pure."""
    return {m.group(0) for m in _SCOPE_PREFIX.finditer(text)}


def turn_ids_in(text: str) -> "set[str]":
    """Every ``turn_traces``-qualified row number in ``text``. Pure."""
    return {m.group(1) for m in _TURN_ID.finditer(text)}


def long_numbers_in(text: str) -> "set[str]":
    """Every bare 11–13 digit run in ``text``, hex-adjacent runs excluded. Pure."""
    return {m.group(0) for m in _LONG_NUMBER.finditer(text)}


def _tracked_files() -> "list[Path]":
    """The INDEX, not the disk: an uncommitted file is not a contract."""
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
                         capture_output=True, text=True, check=True)
    return [REPO_ROOT / rel for rel in out.stdout.split("\0")
            if rel and Path(rel).suffix.lower() not in _SKIP_SUFFIXES]


def _readable() -> "list[tuple[Path, str]]":
    found: list[tuple[Path, str]] = []
    for path in _tracked_files():
        try:
            found.append((path, path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue          # binary or unreadable: nothing to read a datum in
    return found


def test_the_sweep_actually_reads_the_tree():
    """The denominator, pinned. A guard that swept an empty tree would be green forever."""
    files = _tracked_files()
    assert len(files) > 120, f"only {len(files)} files swept — the enumeration is broken"
    names = {p.name for p in files}
    assert {"ner_cases.py", "safety_cases.py"} <= names, "the bench fixtures are not being read"


def test_no_phone_number_outside_the_enumerated_synthetics():
    offenders: dict[str, list[str]] = {}
    for path, text in _readable():
        for number in phone_numbers_in(text) - ALLOWED_SYNTHETIC:
            offenders.setdefault(number, []).append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "phone number(s) outside ALLOWED_SYNTHETIC — anonymise the VALUE (do not delete the "
        "line: it carries a measurement), keep the form, then add the synthetic to the list "
        f"with a comment: { {k: v[:3] for k, v in offenders.items()} }")


def test_no_email_outside_the_enumerated_domains():
    offenders: dict[str, list[str]] = {}
    for path, text in _readable():
        for domain in email_domains_in(text) - ALLOWED_EMAIL_DOMAINS:
            offenders.setdefault(domain, []).append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "mail domain(s) outside ALLOWED_EMAIL_DOMAINS — prefer an RFC 2606 reserved domain "
        "(example.com, or a .test/.invalid TLD) for any new fixture: "
        f"{ {k: v[:3] for k, v in offenders.items()} }")


def test_the_detector_finds_a_number_in_every_shape_it_is_written():
    """Mutation twin for the FORM half: the same number, five ways, is one entry."""
    for written in ("5511987654321", "+55 11 98765-4321", "(11) 98765-4321",
                    "11 98765-4321", "11987654321"):
        assert phone_numbers_in(f"liga no {written} depois") == {"11987654321"}, written


def test_the_detector_does_not_fire_on_things_that_are_not_numbers():
    """The other half of the bar: a detector that over-matches gets switched off."""
    assert phone_numbers_in("123e4567-e89b-42d3-a456-426614174000") == set()   # uuid
    assert phone_numbers_in("sha 5511970001111abcdef") == set()                # inside a hash
    assert phone_numbers_in("(01) 98765-4321") == set()                        # no such DDD
    # Both halves of the construction rule, each REACHABLE — this is the mutation that
    # survived in cogno-host, and it survived because the pattern decided instead of the rule.
    assert phone_numbers_in("(11) 68765-4321") == set()   # 9 digits not opening with 9
    assert phone_numbers_in("(11) 8888-7777") == set()    # 8 digits outside 2-5


def test_the_email_detector_reads_the_domain_and_not_the_mailbox():
    assert email_domains_in("escreve para joao.silva@example.com hoje") == {"example.com"}
    assert email_domains_in("a@x.com e b@y.zz") == {"x.com", "y.zz"}
    assert email_domains_in("sem correio nenhum aqui") == set()


def test_no_truncated_scope_or_tenant_id_anywhere():
    offenders: dict[str, list[str]] = {}
    for path, text in _readable():
        for ident in truncated_scope_ids_in(text):
            offenders.setdefault(ident, []).append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "truncated scope/tenant id(s) in the tree — a ``<hex>…`` abbreviation points at a live "
        "record and must not ship in a public clone; anonymise the line (keep the measurement, "
        f"drop the id): { {k: v[:3] for k, v in offenders.items()} }")


def test_no_turn_id_outside_the_enumerated_provenance():
    offenders: dict[str, list[str]] = {}
    for path, text in _readable():
        for turn_id in turn_ids_in(text) - ALLOWED_TURN_IDS:
            offenders.setdefault(turn_id, []).append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "``turn_traces`` row id(s) outside ALLOWED_TURN_IDS — a bare record id points at a live "
        "turn; either anonymise it to 'a real turn of <date>' (no number) or, if it is kept as "
        "measurement provenance, add it to the list with a reason: "
        f"{ {k: v[:3] for k, v in offenders.items()} }")


def test_no_long_number_outside_the_enumerated_set():
    offenders: dict[str, list[str]] = {}
    for path, text in _readable():
        for number in long_numbers_in(text) - ALLOWED_LONG_NUMBERS - ALLOWED_SYNTHETIC:
            offenders.setdefault(number, []).append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "bare 11–13 digit run(s) outside the enumerated sets — a CPF, an account or a session "
        "id slips in as a plain number; anonymise the value or enumerate it with a reason: "
        f"{ {k: v[:3] for k, v in offenders.items()} }")


def test_the_scope_detector_fires_on_the_abbreviation_and_not_a_bare_sha():
    """Mutation twin: the ellipsis is the shape, a plain commit sha is not an offender.

    The ellipsis is a variable and the hex is synthetic, so this test's OWN source never
    writes a hex run against the ellipsis — otherwise the tree-wide scan would flag it."""
    ell = "…"
    assert truncated_scope_ids_in(f"scope deadbeef{ell} then cafe9999{ell} x") == {
        f"deadbeef{ell}", f"cafe9999{ell}"}
    assert truncated_scope_ids_in("host f251c29, anima d34822a5") == set()       # bare shas
    assert truncated_scope_ids_in(f"the 1094-row corpus{ell}") == set()          # no hex run


def test_the_turn_id_detector_reads_the_table_name_and_not_a_bare_numeral():
    """Mutation twin: the TABLE NAME keys it; ``id=`` and bare forms both count."""
    assert turn_ids_in("on `turn_traces` 1795 the voice") == {"1795"}
    assert turn_ids_in("MEASURED (`turn_traces` id=1440, host") == {"1440"}
    assert turn_ids_in("the contact asked at turn 107 of the day") == set()      # no table name


def test_the_long_number_detector_ignores_a_digit_slice_of_a_hash():
    """Mutation twin: a hex-adjacent run is part of a digest, not a standalone number."""
    assert long_numbers_in("sha 9e5909516563765e here") == set()                 # inside hex
    assert long_numbers_in("cpf 52998224725 fim") == {"52998224725"}             # standalone
    assert long_numbers_in("run 35207468672 of 2026-09-17") == {"35207468672"}
