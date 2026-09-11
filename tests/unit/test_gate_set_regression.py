"""v47 — the set-based PIN anti-regression gate.

Root cause under test: v22_gate.py checked `target_passes >= baseline` (a COUNT).
Between 2026-09-05 and 09-10 six target rows were fixed while four regressed
underneath, so the count rose 1653->1655 and every gate reported PASS while four
PINs silently regressed (N-vs-N1 locants, enclosing marks, stereo). The row
identity is (def_id, smiles) — a def_id spans many SMILES, so a def_id-level set
would have missed W4-S1 (only one of its 8 rows flipped).

`target_row_regressions` compares the current run against a committed floor of
passing (def_id, smiles) rows and reports every MATCH->non-MATCH flip that the
count criterion masks. Governing finding:
internal notes
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pin_conformance_eval import target_row_regressions, load_baseline_target_rows


def _rec(def_id, smiles, verdict, category="target", **kw):
    return {"def_id": def_id, "smiles": smiles, "verdict": verdict,
            "category": category, "expected_pin": kw.get("expected"),
            "shipped_name": kw.get("shipped")}


def test_count_rises_but_a_row_flips_is_caught():
    """The exact masking shape: one row fixed (count +1), one regressed (count -1),
    net count flat/up, but the SET check flags the regressed row."""
    baseline_rows = {("A", "CCO"), ("B", "CC(=O)O")}  # both passed at baseline
    report = {"records": [
        _rec("A", "CCO", "MATCH"),        # A still passes
        _rec("B", "CC(=O)O", "MISMATCH", expected="acetic acid", shipped="ethanoic acid"),  # B regressed
        _rec("C", "c1ccccc1", "MATCH"),   # C newly passes (was failing) -> count masks B
    ]}
    regs = target_row_regressions(report, baseline_rows)
    assert [r["def_id"] for r in regs] == ["B"]
    assert regs[0]["shipped_name"] == "ethanoic acid"


def test_def_id_identity_is_def_id_plus_smiles():
    """A def_id with several SMILES: only the flipped SMILES is a regression."""
    baseline_rows = {("W4-S1", "s1"), ("W4-S1", "s2")}
    report = {"records": [
        _rec("W4-S1", "s1", "MATCH"),
        _rec("W4-S1", "s2", "MISMATCH", expected="x", shipped="y"),
    ]}
    regs = target_row_regressions(report, baseline_rows)
    assert len(regs) == 1 and regs[0]["smiles"] == "s2"


def test_row_dropped_from_packs_is_not_a_regression():
    """A baseline row absent from the current run is an intentional pack edit,
    not a PASS->FAIL flip, so it is NOT flagged (fail-safe against pack churn)."""
    baseline_rows = {("A", "CCO"), ("GONE", "removed")}
    report = {"records": [_rec("A", "CCO", "MATCH")]}
    assert target_row_regressions(report, baseline_rows) == []


def test_empty_floor_disables_the_check():
    """No floor file -> empty set -> no regressions (falls back to count only)."""
    assert target_row_regressions({"records": [_rec("A", "x", "MISMATCH")]}, set()) == []


def test_committed_floor_loads_and_is_nonempty():
    """The shipped floor file parses and carries the 1653-row baseline."""
    rows = load_baseline_target_rows()
    assert len(rows) >= 1653
    assert all(isinstance(k, tuple) and len(k) == 2 for k in rows)
