"""Suite versioning (plan 0.6) — the mechanics behind "absolute counts, and only
within the same suite version".

Every ``*_cases.py`` declares a manual ``SUITE_ID`` (``"ner-v1"``); this module
computes a ``suite_hash`` over the canonical serialization of the case data
(case ids + every expected value). The pair is:

* recorded into every persisted run's metadata (``BenchReport.suites``), so a
  results JSON is attributable to an exact case universe;
* pinned by ``cognobench/suite_hashes.json`` + a unit test — ANY case
  addition/removal/edit changes the hash, and the test fails until the human
  bumps the ``SUITE_ID`` and re-records (the same pattern as the
  domains↔prompt alignment test). Published tables cite the suite id; numbers
  from different suite versions never share a table.

Deliberately NOT hashed: prompts and models (they are the object under
measurement, not the instrument) and check *predicates* (a scorer change is a
code review concern; hashing code churns on refactors).

Regenerate the pin file after a deliberate suite change + SUITE_ID bump:

    python -m cognobench.suites --update
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

_PIN_FILE = Path(__file__).resolve().parent / "suite_hashes.json"


def canonical_suite_hash(cases: list) -> str:
    """sha256 (16 hex chars) of the canonical JSON of a case list.

    ``dataclasses.asdict`` walks nested case dataclasses (multi-turn cases);
    ``default=str`` covers tuples/None; ``sort_keys`` makes field order
    irrelevant. Case ORDER matters on purpose — ``--limit`` slices by position,
    so reordering changes which cases a capped run sees."""
    payload = [dataclasses.asdict(c) for c in cases]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def registry() -> dict[str, dict]:
    """dimension → {suite_id, suite_hash, cases (count), case_ids}.

    Imported lazily so ``--stub`` runs and unit tests pay the import once.
    The superego suite covers its three scored sub-dimensions
    (superego_scope/judge/voice) — one case file, one version."""
    from cognobench.ner_cases import NER_CASES, SUITE_ID as NER_SUITE
    from cognobench.drift_cases import DRIFT_CASES, SUITE_ID as DRIFT_SUITE
    from cognobench.noumeno_cases import NOUMENO_CASES, SUITE_ID as NOUMENO_SUITE
    from cognobench.id_cases import ID_CASES, SUITE_ID as ID_SUITE
    from cognobench.ego_cases import EGO_CASES, SUITE_ID as EGO_SUITE
    from cognobench.superego_cases import SUPEREGO_CASES, SUITE_ID as SUPEREGO_SUITE
    from cognobench.conversation_cases import CONVERSATION_CASES, SUITE_ID as CONV_SUITE
    from cognobench.safety_cases import SAFETY_CASES, SUITE_ID as SAFETY_SUITE

    table = {
        "noumeno": (NOUMENO_SUITE, NOUMENO_CASES),
        "ner": (NER_SUITE, NER_CASES),
        "id": (ID_SUITE, ID_CASES),
        "safety": (SAFETY_SUITE, SAFETY_CASES),
        "ego": (EGO_SUITE, EGO_CASES),
        "superego": (SUPEREGO_SUITE, SUPEREGO_CASES),
        "drift": (DRIFT_SUITE, DRIFT_CASES),
        "conversations": (CONV_SUITE, CONVERSATION_CASES),
    }
    return {
        dim: {
            "suite_id": suite_id,
            "suite_hash": canonical_suite_hash(cases),
            "cases": len(cases),
            "case_ids": [c.id for c in cases],
        }
        for dim, (suite_id, cases) in table.items()
    }


def recorded() -> dict[str, dict]:
    """The pinned {dimension: {suite_id, suite_hash}} from suite_hashes.json."""
    if not _PIN_FILE.exists():
        return {}
    return json.loads(_PIN_FILE.read_text(encoding="utf-8"))


def update_pins() -> dict[str, dict]:
    live = {dim: {"suite_id": info["suite_id"], "suite_hash": info["suite_hash"],
                  "cases": info["cases"]}
            for dim, info in registry().items()}
    _PIN_FILE.write_text(json.dumps(live, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    return live


if __name__ == "__main__":
    import sys
    if "--update" in sys.argv:
        pins = update_pins()
        for dim, info in pins.items():
            print(f"  {dim:<14} {info['suite_id']:<16} {info['suite_hash']}  "
                  f"({info['cases']} cases)")
        print(f"\nrecorded → {_PIN_FILE}")
    else:
        live, pinned = registry(), recorded()
        drift = [d for d in live
                 if pinned.get(d, {}).get("suite_hash") != live[d]["suite_hash"]]
        for dim, info in live.items():
            mark = "≠ PIN" if dim in drift else "ok"
            print(f"  {dim:<14} {info['suite_id']:<16} {info['suite_hash']}  {mark}")
        raise SystemExit(1 if drift else 0)
