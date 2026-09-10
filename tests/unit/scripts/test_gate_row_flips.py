"""-CLEANUP Item 3 — the flip harness's provenance evidence.

`scripts/gate_row_flips.py` is the tool that decides whether an A/B measurement
may be believed. Its `_source_digest` used to glob `*.py` only, so a **data-only**
naming change (`src/orthonym/data/*.json`) left the digest identical and `diff`
answered a legitimate A/B with

    FATAL: cannot prove the two snapshots ran different code...
           the engines are byte-identical (rc 4)

which is factually false. That class is historically real: and
 are naming-behaviour commits whose only engine change is a `.json`.

These tests pin the two properties the fix must have — data sensitivity and
bytecode insensitivity — plus the guard-4 OR that fixed in the
fallback branch only.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def flips():
    """Import `scripts/gate_row_flips.py` by path (it is a script, not a package)."""
    path = PROJECT_ROOT / "scripts" / "gate_row_flips.py"
    spec = importlib.util.spec_from_file_location("gate_row_flips_undertest", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tree(root: Path, files: dict) -> Path:
    """Build a fake engine tree: root/src/orthonym/<relpath> = <content>."""
    pkg = root / "src" / "orthonym"
    for rel, content in files.items():
        p = pkg / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


BASE_FILES = {
    "namer.py": "def name(): return 'ethanol'\n",
    "data/__init__.py": "",
    "data/iupac_2013_pin_list.json": '{"pins": ["ethanol"]}',
}


def test_data_only_change_moves_the_digest(flips, tmp_path):
    """THE regression this item exists for.

    Two trees whose ONLY difference is a runtime `.json` must NOT hash equal.
    Before the fix these produced the same digest, which is what turned a real
    A/B into a false FATAL.
    """
    a = _tree(tmp_path / "a", BASE_FILES)
    b = _tree(tmp_path / "b", {**BASE_FILES,
                               "data/iupac_2013_pin_list.json":
                                   '{"pins": ["ethanol", "propan-1-ol"]}'})
    assert flips._source_digest(a) != flips._source_digest(b), (
        "a data-only engine change must be visible to the provenance digest"
    )


def test_code_only_change_still_moves_the_digest(flips, tmp_path):
    """The original `.py` sensitivity must survive the widening."""
    a = _tree(tmp_path / "a", BASE_FILES)
    b = _tree(tmp_path / "b", {**BASE_FILES,
                               "namer.py": "def name(): return 'methanol'\n"})
    assert flips._source_digest(a) != flips._source_digest(b)


def test_identical_trees_hash_equal(flips, tmp_path):
    """Positive control: without it, a digest that simply always differs would
    pass both tests above while proving nothing."""
    a = _tree(tmp_path / "a", BASE_FILES)
    b = _tree(tmp_path / "b", BASE_FILES)
    assert flips._source_digest(a) == flips._source_digest(b)


def test_bytecode_caches_do_not_move_the_digest(flips, tmp_path):
    """Widening `*.py` to `*` must not make the digest unstable.

    `__pycache__` is build output that varies with interpreter and mtime; if it
    counted, the SAME source would hash differently between runs and guard 4
    would flip from false-FATAL to false-PASS — a strictly worse failure.
    """
    a = _tree(tmp_path / "a", BASE_FILES)
    b = _tree(tmp_path / "b", BASE_FILES)
    before = flips._source_digest(b)
    cache = b / "src" / "orthonym" / "__pycache__"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "namer.cpython-311.pyc").write_bytes(b"\x00\x01garbage")
    (b / "src" / "orthonym" / "data" / "stray.pyc").write_bytes(b"\x02\x03")
    assert flips._source_digest(b) == before
    assert flips._source_digest(a) == flips._source_digest(b)


def test_a_new_data_extension_needs_no_harness_edit(flips, tmp_path):
    """The point of globbing `*` rather than listing extensions: the next runtime
    data format is covered without anyone remembering to add it here."""
    a = _tree(tmp_path / "a", BASE_FILES)
    b = _tree(tmp_path / "b", {**BASE_FILES, "data/ring_table.csv": "a,b\n1,2\n"})
    assert flips._source_digest(a) != flips._source_digest(b)


# ---------------------------------------------------------------------------
# guard 4 — the OR that fixed in the fallback branch only
# ---------------------------------------------------------------------------

def _meta(digest, head, fps):
    return {"source_digest": digest, "git_head": head, "fingerprints": fps}


def _proven_different(bm, cm):
    """Mirror of the guard-4 predicate in `diff`, extracted so it can be tested
    without building two full 1981-row snapshots.

    NOTE this duplicates the expression under test rather than importing it; the
    accompanying assertion is that the source still reads this way (below), so a
    divergence is caught rather than silently tolerated."""
    b_dig, c_dig = bm.get("source_digest"), cm.get("source_digest")
    if b_dig and c_dig:
        return (b_dig != c_dig
                or bm["git_head"] != cm["git_head"]
                or bm["fingerprints"] != cm["fingerprints"])
    return (bm["git_head"] != cm["git_head"]
            or bm["fingerprints"] != cm["fingerprints"])


def test_primary_branch_ors_in_git_head(flips):
    """Equal digests + different commit must still count as proven-different."""
    bm = _meta("aaaa", "commit1", {"f": "1"})
    cm = _meta("aaaa", "commit2", {"f": "1"})
    assert _proven_different(bm, cm) is True


def test_primary_branch_still_accepts_a_moved_digest(flips):
    bm = _meta("aaaa", "commit1", {"f": "1"})
    cm = _meta("bbbb", "commit1", {"f": "1"})
    assert _proven_different(bm, cm) is True


def test_truly_identical_snapshots_are_refused(flips):
    """The guard must keep failing CLOSED on a genuine non-A/B."""
    bm = _meta("aaaa", "commit1", {"f": "1"})
    cm = _meta("aaaa", "commit1", {"f": "1"})
    assert _proven_different(bm, cm) is False


def test_guard4_source_still_matches_the_mirrored_predicate():
    """Tripwire for the mirror above: if `diff`'s primary branch stops ORing in
    `git_head`, this fails even though the mirrored copy would keep passing."""
    src = (PROJECT_ROOT / "scripts" / "gate_row_flips.py").read_text()
    i = src.index("b_dig, c_dig = bm.get(\"source_digest\")")
    window = src[i:i + 900]
    assert "proven_different = (b_dig != c_dig" in window
    assert 'or bm["git_head"] != cm["git_head"]' in window
    assert 'or bm["fingerprints"] != cm["fingerprints"]' in window


# ---------------------------------------------------------------------------
# MINOR 6 — a gold that moved between snapshots must be NAMED, not left a mystery
# ---------------------------------------------------------------------------

def _snapshot(digest, head, rows):
    return {
        "_meta": {
            "harness": "scripts/gate_row_flips.py",
            "git_head": head,
            "git_describe": f"{head[:7]} test",
            "engine_orthonym_path": f"/fake/{head}/orthonym/__init__.py",
            "packs_source": "/fake/packs",
            "legacy_source": [],
            "fingerprints": {"src/orthonym/namer.py": digest},
            "source_digest": digest,
            "rows": len(rows),
            "target_rows": 1,
            "expect_targets": 1,
            "target_passes": sum(1 for r in rows.values()
                                 if r["category"] == "target"
                                 and r["verdict"] == "MATCH"),
            "target_total": 1,
            "unevaluable": 0,
        },
        "rows": rows,
    }


def _row(expected, shipped, category="target"):
    return {
        "pack": "p", "def_id": "D1", "smiles": "CCO", "category": category,
        "expected_pin": expected, "shipped_name": shipped,
        "verdict": "MATCH" if expected == shipped else "MISMATCH",
    }


def test_a_moved_gold_is_reported_separately(flips, tmp_path, capsys):
    """The engine did not move; the GOLD did. Before this, the row surfaced under
    'REGRESSED ROWS' with two identical shipped names and no explanation."""
    base = tmp_path / "b.json"
    cur = tmp_path / "c.json"
    base.write_text(json.dumps(_snapshot(
        "aaaa", "commit1", {"k1": _row("ethanol", "ethanol")})))
    cur.write_text(json.dumps(_snapshot(
        "bbbb", "commit2", {"k1": _row("L-ethanol", "ethanol")})))

    rc = flips.diff(base, cur, allow_zero_flips=True, targets_only=False)
    out = capsys.readouterr().out
    assert "GOLD moved (not code): 1" in out, out
    assert "ROWS WHOSE GOLD MOVED BETWEEN SNAPSHOTS" in out, out
    assert "'ethanol' -> 'ethanol'" in out, out
    assert rc == 0


def test_an_unmoved_gold_reports_zero(flips, tmp_path, capsys):
    """Positive control — otherwise a bucket that always fires would pass above."""
    base = tmp_path / "b2.json"
    cur = tmp_path / "c2.json"
    base.write_text(json.dumps(_snapshot(
        "aaaa", "commit1", {"k1": _row("ethanol", "methanol")})))
    cur.write_text(json.dumps(_snapshot(
        "bbbb", "commit2", {"k1": _row("ethanol", "ethanol")})))

    flips.diff(base, cur, allow_zero_flips=False, targets_only=False)
    out = capsys.readouterr().out
    assert "GOLD moved (not code): 0" in out, out
    assert "ROWS WHOSE GOLD MOVED" not in out, out


def test_source_digest_does_not_enumerate_extensions():
    """Tripwire: re-narrowing the glob to an extension allowlist is the exact
    regression this item fixed."""
    src = (PROJECT_ROOT / "scripts" / "gate_row_flips.py").read_text()
    i = src.index("def _source_digest")
    body = src[i:i + 2400]
    assert 'src.rglob("*.py")' not in body, (
        "_source_digest re-narrowed to *.py — a data-only naming change would "
        "again produce a false FATAL"
    )
    assert 'src.rglob("*")' in body
