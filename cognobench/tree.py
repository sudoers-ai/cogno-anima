"""Which TREE produced this number — the cognobench half of a rule this house
already wrote once, for the host (`hostbench/tree.py`, cogno-host #801).

A score without the tree that produced it is not comparable, and the failure is not
hypothetical: on 2026-09-08 two safety numbers measured on two different corpora were
read as one number moving. The persisted run JSON carried the model, the embedder and
the suite pin — everything except the code.

Three facts travel with the sha, and each exists because its absence reads as a pass:

* **dirty** — a sha over a dirty tree is a stamp that lies. It is recorded and PRINTED,
  never used to refuse: refusing would push every working-tree measurement out of the
  instrument, and a labelled measurement beats an unmade one.
* **ruler** — the git tree object of `cognobench/` itself. Two trees only produce
  comparable scores when the BENCH is the same code; `git` already hashes a directory
  into one id, so the check is exact and free. It is None when `cognobench/` has
  uncommitted changes, because a tree id read off HEAD would then name code that is not
  what ran.
* **root** — the tree is located from THIS MODULE's own path, not from the cwd. The cwd
  is not evidence of anything: `python3 /abs/path/script.py` puts the script's directory
  on `sys.path[0]` and the cwd nowhere, so a run can import one tree while standing in
  another. The honest answer to "which code produced this" is where the code that ran
  came from.

Not knowing is a result and is printed like one: no git, no repository, no sha — the
stamp says the question was not answered instead of quietly dropping the line.

**Why this is a second module and not an import.** `hostbench/tree.py` lives in
`cogno-host`, which DEPENDS on this package; an import the other way inverts the
dependency, and `cognobench/` is deliberately excluded from the wheel
(`[tool.setuptools.packages.find] include = ["cogno_anima*"]`), so a host import would
rest on a checkout layout the packaging refuses to promise. What is shared is the RULE,
and the two modules answer about different repositories: this one has no served process
and no lib pins to read, and hostbench's has no suite versions. If the pair ever needs
one definition, the move is to a SHIPPED package and hostbench importing it — a decision
above a bench module.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

# The tree that produced a number is the one the imported code came from — see the
# module docstring on why this is not the cwd.
BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent


def _git(args: list[str], cwd: str | Path) -> str | None:
    """`git <args>` in `cwd`, stripped — or None on any failure.

    Never raises and never inherits stdin: this runs inside a report, and a diagnostic
    that can hang is worse than no diagnostic."""
    try:
        out = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=10,
                             stdin=subprocess.DEVNULL)
    except Exception:      # noqa: BLE001 — no git binary, no diagnosis, no crash
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@dataclass(frozen=True)
class TreeId:
    """One tree, as it must appear in any result that came out of it."""

    path: str
    sha: str | None
    subject: str = ""
    committed: str = ""
    dirty: bool = False
    ruler: str | None = None      # the `cognobench/` tree object — see `bench_ruler`

    @property
    def short(self) -> str:
        return self.sha[:7] if self.sha else "NO-SHA"

    @property
    def join_key(self) -> str:
        """What two runs must share to be poolable. A DIRTY tree gets its own key: the
        sha does not describe what ran, so two dirty runs at the same sha are not
        evidence of the same code."""
        if not self.sha:
            return "unknown"
        return f"{self.sha}+dirty" if self.dirty else self.sha

    def stamp(self) -> str:
        """The one line that makes a number citable a week later."""
        if not self.sha:
            return (f"tree {self.path}  ⚠ SHA UNKNOWN (not a git tree) — this number "
                    f"does not say which code produced it")
        dirty = "  ⚠ DIRTY TREE (the sha does not describe what was measured)" \
            if self.dirty else ""
        when = f"  {self.committed}" if self.committed else ""
        ruler = "" if self.ruler else "  ⚠ cognobench/ is dirty — the RULER is unpinned"
        return f"tree {self.short}{when}  {self.subject}{dirty}{ruler}"

    def as_dict(self) -> dict:
        return asdict(self)


def bench_ruler(path: str | Path = REPO_ROOT) -> str | None:
    """The git tree object of `cognobench/` — the identity of the RULER, not of what it
    measured.

    None when `cognobench/` has uncommitted changes: a tree id read off HEAD would name
    code that is not what ran, which is the lie this module exists to refuse."""
    if _git(["status", "--porcelain", "--", "cognobench"], path):
        return None
    return _git(["rev-parse", "HEAD:cognobench"], path)


def identify(path: str | Path = REPO_ROOT) -> TreeId:
    """Stamp a directory. Degrades — loudly, via `TreeId.stamp` — when it is not a git
    tree."""
    resolved = str(Path(path).resolve())
    sha = _git(["rev-parse", "HEAD"], resolved)
    if not sha:
        return TreeId(path=resolved, sha=None)
    return TreeId(
        path=resolved,
        sha=sha,
        subject=_git(["log", "-1", "--pretty=%s", sha], resolved) or "",
        committed=_git(["log", "-1", "--date=iso-strict", "--pretty=%cd", sha],
                       resolved) or "",
        dirty=bool(_git(["status", "--porcelain"], resolved)),
        ruler=bench_ruler(resolved),
    )
