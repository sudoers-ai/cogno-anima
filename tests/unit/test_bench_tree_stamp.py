"""The number carries the tree that produced it — and two trees never pool as two runs.

Every test here is the twin of a way the stamp can be born inert: a stamp that is only
written and never read, a refusal that fires on absence, a dirty tree that stamps as if
it were clean, a ruler nobody compares.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from cognobench import compare
from cognobench.tree import TreeId, bench_ruler, identify


def _repo(path, *, subject="first"):
    """A throwaway git repo with one commit — no network, no shared state."""
    path.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True,  # noqa: E731
                                    capture_output=True)
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "T")
    (path / "cognobench").mkdir(exist_ok=True)
    (path / "cognobench" / "cases.py").write_text("CASES = [1]\n")
    run("add", "-A")
    run("commit", "-qm", subject)
    return path


# ── identify: the three facts, and the one it refuses to invent ──────────────────────

def test_identify_reports_sha_subject_and_clean_tree(tmp_path):
    t = identify(_repo(tmp_path / "r", subject="the subject line"))
    assert t.sha and len(t.sha) == 40
    assert t.subject == "the subject line"
    assert t.dirty is False
    assert t.short == t.sha[:7]
    assert t.short in t.stamp() and "the subject line" in t.stamp()


def test_a_dirty_tree_is_stamped_dirty_and_not_refused(tmp_path):
    """A sha over a dirty tree is a stamp that lies — so it says so, in the line a human
    reads. It is NOT refused: pushing every working-tree measurement out of the
    instrument buys nothing, and a labelled measurement beats an unmade one."""
    repo = _repo(tmp_path / "r")
    (repo / "cognobench" / "cases.py").write_text("CASES = [1, 2]\n")
    t = identify(repo)
    assert t.sha, "a dirty tree still has a sha — it is the DESCRIPTION that is void"
    assert t.dirty is True
    assert "DIRTY" in t.stamp()


def test_not_a_git_tree_says_the_question_was_not_answered(tmp_path):
    """A blank where a verdict should be reads as a pass."""
    plain = tmp_path / "plain"
    plain.mkdir()
    t = identify(plain)
    assert t.sha is None
    assert t.short == "NO-SHA"
    assert "UNKNOWN" in t.stamp()


def test_ruler_is_the_bench_directory_and_unpins_when_it_is_dirty(tmp_path):
    """Two trees only produce comparable scores when the BENCH is the same code."""
    repo = _repo(tmp_path / "r")
    clean = bench_ruler(repo)
    assert clean, "a clean tree must yield the cognobench/ tree object"
    assert bench_ruler(repo) == clean, "the same tree must hash to the same ruler"
    (repo / "cognobench" / "cases.py").write_text("CASES = [9]\n")
    assert bench_ruler(repo) is None, (
        "with cognobench/ dirty, a tree id read off HEAD names code that did not run"
    )


def test_two_trees_that_differ_have_different_rulers(tmp_path):
    """The discrimination check: a ruler that never changes measures nothing."""
    a = _repo(tmp_path / "a")
    b = _repo(tmp_path / "b")
    assert bench_ruler(a) == bench_ruler(b), "identical content → identical tree object"
    (b / "cognobench" / "cases.py").write_text("CASES = [1, 2, 3]\n")
    subprocess.run(["git", "-C", str(b), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(b), "commit", "-qm", "edit"], check=True,
                   capture_output=True)
    assert bench_ruler(a) != bench_ruler(b)


def test_dirty_gets_its_own_join_key(tmp_path):
    """Two DIRTY runs at the same sha are not evidence of the same code."""
    clean = TreeId(path="/x", sha="a" * 40)
    dirty = TreeId(path="/x", sha="a" * 40, dirty=True)
    assert clean.join_key != dirty.join_key
    assert TreeId(path="/x", sha=None).join_key == "unknown"


# ── the runner actually writes it (a stamp nobody records is decoration) ─────────────

def test_a_stub_run_records_the_tree_in_its_config():
    """A stamp the runner never writes is decoration. Exercised through the real
    persistence path (`--out`), because the JSON is what a comparison a week later
    reads — not the in-memory report."""
    from cognobench.runner import main

    out_dir = pytest.importorskip("pathlib").Path(
        __import__("tempfile").mkdtemp(prefix="cognobench-tree-"))
    assert main(["--stub", "--only", "safety", "--limit", "1",
                 "--out", str(out_dir)]) == 0
    written = list(out_dir.glob("*.json"))
    assert len(written) == 1, written
    tree = json.loads(written[0].read_text())["config"]["tree"]
    assert set(tree) >= {"sha", "dirty", "ruler", "path", "subject", "committed"}
    # Round-trips out of the artifact and back into the renderer, unchanged.
    assert TreeId(**tree).stamp()


def test_the_rendered_report_carries_the_stamp():
    from cognobench.report import render
    from cognobench.types import BenchReport

    report = BenchReport(model="m")
    report.config["tree"] = TreeId(path="/x", sha="b" * 40, subject="a subject").as_dict()
    assert "b" * 7 in render(report) and "a subject" in render(report)


# ── compare: two trees are not two runs ──────────────────────────────────────────────

def _run(tmp_path, name, *, sha, correct=True, dirty=False):
    payload = {
        "model": "m", "config": {"limit": None, "embed_model": "e"},
        "suites": {"ner": {"suite_id": "ner-v1", "suite_hash": "abc"}},
        "dimensions": [{"dimension": "ner", "valid": True, "checks": [
            {"case_id": "c1", "field": "f", "expected": "X", "actual": "X",
             "correct": correct}]}],
    }
    if sha is not None:
        payload["config"]["tree"] = {"path": "/x", "sha": sha, "subject": "s",
                                     "committed": "", "dirty": dirty, "ruler": "r"}
    (tmp_path / name).write_text(json.dumps(payload))


def test_runs_from_two_trees_do_not_pool(tmp_path, capsys):
    _run(tmp_path, "a.json", sha="a" * 40, correct=True)
    _run(tmp_path, "b.json", sha="b" * 40, correct=False)
    by_model = compare._index(compare.load_runs([tmp_path]))
    votes = by_model["m +e"]["dims"]["ner"][("c1", "f")]
    assert votes == [True], (
        "the second tree must be dropped, not averaged in — pooling them turns a code "
        f"difference into run-to-run noise; got {votes}"
    )
    err = capsys.readouterr().err
    assert "mixed code trees" in err and "aaaaaaa" in err and "bbbbbbb" in err


def test_allow_tree_drift_pools_them_and_says_so(tmp_path, capsys):
    _run(tmp_path, "a.json", sha="a" * 40, correct=True)
    _run(tmp_path, "b.json", sha="b" * 40, correct=False)
    by_model = compare._index(compare.load_runs([tmp_path]), allow_tree_drift=True)
    assert sorted(by_model["m +e"]["dims"]["ner"][("c1", "f")]) == [False, True]
    assert "--allow-tree-drift" in capsys.readouterr().err


def test_an_unstamped_run_is_kept_and_named_not_treated_as_drift(tmp_path, capsys):
    """Absence is not a change of side — every artifact written before the stamp
    existed carries none, and dropping them would delete the entire history."""
    _run(tmp_path, "a.json", sha="a" * 40, correct=True)
    _run(tmp_path, "b.json", sha=None, correct=False)
    by_model = compare._index(compare.load_runs([tmp_path]))
    assert sorted(by_model["m +e"]["dims"]["ner"][("c1", "f")]) == [False, True]
    err = capsys.readouterr().err
    assert "NO tree stamp" in err
    assert "mixed code trees" not in err


def test_the_same_tree_twice_is_one_tree(tmp_path, capsys):
    """The other half of discrimination: the guard must not fire on agreement."""
    _run(tmp_path, "a.json", sha="a" * 40, correct=True)
    _run(tmp_path, "b.json", sha="a" * 40, correct=False)
    by_model = compare._index(compare.load_runs([tmp_path]))
    assert sorted(by_model["m +e"]["dims"]["ner"][("c1", "f")]) == [False, True]
    assert "mixed code trees" not in capsys.readouterr().err


def test_a_dirty_run_is_announced(tmp_path, capsys):
    _run(tmp_path, "a.json", sha="a" * 40, correct=True, dirty=True)
    compare._index(compare.load_runs([tmp_path]))
    assert "DIRTY tree" in capsys.readouterr().err


def test_the_cli_exposes_the_override(tmp_path):
    _run(tmp_path, "a.json", sha="a" * 40)
    _run(tmp_path, "b.json", sha="b" * 40)
    assert compare.main([str(tmp_path), "--pair", "m +e", "m +e"]) == 0
    with pytest.raises(SystemExit):
        compare.main([str(tmp_path), "--allow-tree-drift", "--nonsense"])
