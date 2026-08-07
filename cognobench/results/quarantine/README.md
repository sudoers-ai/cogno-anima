Pre-reboot artifacts from a dying machine (2026-08-07 02:43/02:46 UTC): every
pipeline dimension failed with EMPTY-response StageParseError in series (119
cases), was scored as per-case model fault, and the runs stayed "valid" at
448/476 and 121/241 — the incident that motivated the systemic-failure breaker
in `_systemic_guard`. Kept as evidence; `compare.py` does not read this
subdirectory (non-recursive glob).
