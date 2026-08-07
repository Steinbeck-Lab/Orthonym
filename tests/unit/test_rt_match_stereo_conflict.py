"""RISK 3 (fable v30) — the recovery lane's stereo-flagged RT compare.

`_stereo_emit_decision` flags an emission `stereo_unexpressed` whenever the name is
stereo-INCOMPLETE, which includes a name asserting PARTIAL (possibly wrong) stereo. The
flagged branch of `_rt_match` compares stereo-STRIPPED constitution only, so any stereo the
name DOES assert is unverified — a wrong stereoisomer can ship at T3 (0 dev500 instances,
by luck not architecture).

A first fix (ca6bb3de, routing through C6's RegistrationHash verdict) was REVERTED: a flagged
emission is by construction stereo-count-mismatched, and C6's conflict check only fires on
EQUAL counts (so it can never catch a flagged partial-conflict), and its TAUTOMER_HASH is
E/Z-blind on conjugated systems; it also LOOSENED constitution matching (accepted charge/
tautomer differences). The correct fix is a per-element atom-mapped stereo compare — NOT a
whole-molecule hash. These tests assert the current CORRECT behavior and mark the two real
holes xfail (they will XPASS when the per-element fix lands — remove the markers then).

`_rt_match` has one live caller (namer.py, `_try_general_engine_recovery`, best-effort/
complete-gated) -> PIN default byte-identical.
"""
import pytest
from orthonym import namer as NM

L = "C[C@H](N)C(=O)O"   # L-alanine
D = "C[C@@H](N)C(=O)O"  # D-alanine — same constitution, OPPOSITE stereo
FLAT = "CC(N)C(=O)O"    # alanine, no stereo specified (omission)


# --- correct current behavior (must hold) ------------------------------------

def test_flagged_tolerates_stereo_omission():
    # a flagged constitution-only name (less/no stereo) is the sanctioned case -> ship.
    assert NM.Orthonym._rt_match(L, FLAT, True) is True


def test_flagged_accepts_exact_same():
    assert NM.Orthonym._rt_match(L, L, True) is True


def test_flagged_rejects_constitution_difference():
    assert NM.Orthonym._rt_match(L, "c1ccccc1", True) is False


def test_flagged_does_not_loosen_charge():
    # the reverted (constitution-only) compare keeps charge -> neutral != anion.
    # (the ca6bb3de C6 route wrongly returned True here — a loosening we reverted.)
    assert NM.Orthonym._rt_match("C[C@H](N)C(=O)O", "CC(N)C(=O)[O-]", True) is False


def test_unflagged_isomeric_exact_unchanged():
    assert NM.Orthonym._rt_match(L, D, False) is False
    assert NM.Orthonym._rt_match(L, L, False) is True


# --- KNOWN UNCLOSED HOLES (RISK 3) — xfail until the per-element fix lands ----

@pytest.mark.xfail(reason="RISK 3: flagged partial-conflict not caught; needs "
                          "per-element atom-mapped stereo compare", strict=True)
def test_flagged_partial_stereo_conflict_rejected():
    # 2-center input; name asserts 1 center WRONG (the literal RISK-3 shape).
    assert NM.Orthonym._rt_match(
        "C[C@H](O)[C@H](N)C(=O)O", "C[C@@H](O)C(N)C(=O)O", True) is False


@pytest.mark.xfail(reason="RISK 3: conjugated E/Z conflict not caught (stripped "
                          "compare drops bond stereo)", strict=True)
def test_flagged_conjugated_ez_conflict_rejected():
    E = "c1ccccc1/C=C/c1ccccc1"
    Z = "c1ccccc1/C=C\\c1ccccc1"
    assert NM.Orthonym._rt_match(E, Z, True) is False
