"""RISK 3 (a review) — the recovery lane's stereo-flagged RT compare.

`_stereo_emit_decision` flags an emission `stereo_unexpressed` whenever the name is
stereo-INCOMPLETE, which includes a name asserting PARTIAL (possibly wrong) stereo. The old
flagged branch of `_rt_match` compared stereo-STRIPPED constitution only, so any stereo the
name DID assert was unverified — a wrong stereoisomer could ship at T3.

A first fix (ca6bb3de, routing through C6's RegistrationHash verdict) was REVERTED: a flagged
emission is by construction stereo-count-mismatched, C6's conflict check only fires on EQUAL
counts (so it can never catch a flagged partial-conflict), its TAUTOMER_HASH is E/Z-blind on
conjugated systems, and it LOOSENED constitution matching (accepted charge/tautomer diffs).

The correct fix (this suite pins it) is a **per-element atom-mapped stereo compare**: the
name's asserted structure (OPSIN parse) is matched as a chiral substructure QUERY into the
input TARGET, gated by stereo-stripped constitution equality. A query element WITHOUT stereo
matches anything (tolerate OMISSION); a query element WITH stereo forces the input to agree
(catch CONFLICT and FABRICATION). Covers tetrahedral R/S AND double-bond E/Z (RDKit
`useChirality=True` enforces both on the pinned 2026.03.1). Existence over valid mappings is
the right criterion, so molecular symmetry (meso / C2 / enantiomer pairs) is handled correctly.

`_rt_match` has one live caller (namer.py, `_try_general_engine_recovery`, best-effort/
complete-gated) -> PIN default byte-identical.
"""
from orthonym import namer as NM

L = "C[C@H](N)C(=O)O"   # L-alanine
D = "C[C@@H](N)C(=O)O"  # D-alanine — same constitution, OPPOSITE stereo
FLAT = "CC(N)C(=O)O"    # alanine, no stereo specified (omission)


# --- correct behavior: tolerate omission, exact match, reject constitution/charge ----------

def test_flagged_tolerates_stereo_omission():
    # a flagged constitution-only name (less/no stereo) is the sanctioned case -> ship.
    assert NM.Orthonym._rt_match(L, FLAT, True) is True


def test_flagged_accepts_exact_same():
    assert NM.Orthonym._rt_match(L, L, True) is True


def test_flagged_rejects_constitution_difference():
    assert NM.Orthonym._rt_match(L, "c1ccccc1", True) is False


def test_flagged_does_not_loosen_charge():
    # the constitution guard keeps charge -> neutral != anion.
    # (the reverted ca6bb3de C6 route wrongly returned True here — a loosening.)
    assert NM.Orthonym._rt_match("C[C@H](N)C(=O)O", "CC(N)C(=O)[O-]", True) is False


def test_flagged_rejects_isotope_difference():
    # a plain name for a 13C-labeled chiral input is the WRONG isotopologue -> reject.
    # (a review RISK 1: the old `isomericSmiles=False` guard stripped isotopes too and accepted it.)
    assert NM.Orthonym._rt_match("[13CH3][C@H](O)CC", "CCC(C)O", True) is False


def test_flagged_isotope_kept_stereo_omitted_accepted():
    # same isotope, name omits only the stereo -> tolerate the stereo omission, keep the label.
    assert NM.Orthonym._rt_match("[13CH3][C@H](O)CC", "[13CH3]C(O)CC", True) is True


def test_unflagged_isomeric_exact_unchanged():
    assert NM.Orthonym._rt_match(L, D, False) is False
    assert NM.Orthonym._rt_match(L, L, False) is True


# --- RISK 3 holes now CLOSED: per-element stereo conflict must be rejected -----------------

def test_flagged_partial_stereo_conflict_rejected():
    # 2-center input; name asserts 1 center WRONG (the literal shape).
    assert NM.Orthonym._rt_match(
        "C[C@H](O)[C@H](N)C(=O)O", "C[C@@H](O)C(N)C(=O)O", True) is False


def test_flagged_partial_stereo_correct_accepted():
    # 2-center input; name asserts 1 center CORRECTLY, omits the other -> ship (breadth).
    assert NM.Orthonym._rt_match(
        "C[C@H](O)[C@H](N)C(=O)O", "C[C@H](O)C(N)C(=O)O", True) is True


def test_flagged_conjugated_ez_conflict_rejected():
    E = "c1ccccc1/C=C/c1ccccc1"
    Z = "c1ccccc1/C=C\\c1ccccc1"
    assert NM.Orthonym._rt_match(E, Z, True) is False


def test_flagged_conjugated_ez_omission_accepted():
    E = "c1ccccc1/C=C/c1ccccc1"
    flat = "c1ccccc1C=Cc1ccccc1"
    assert NM.Orthonym._rt_match(E, flat, True) is True


def test_flagged_aliphatic_ez_conflict_rejected():
    assert NM.Orthonym._rt_match("C/C=C/C", "C/C=C\\C", True) is False


def test_flagged_cn_oxime_ez_conflict_rejected():
    assert NM.Orthonym._rt_match("C/C=N/O", "C/C=N\\O", True) is False


def test_flagged_fabricated_stereo_rejected():
    # name asserts a stereocenter the input does NOT have -> reject fabrication.
    assert NM.Orthonym._rt_match(FLAT, L, True) is False


# --- symmetry traps: existence-over-mappings must NOT rescue a wrong isomer ----------------

RR = "O[C@@H]([C@H](O)C(=O)O)C(=O)O"    # (R,R)-tartaric
SS = "O[C@H]([C@@H](O)C(=O)O)C(=O)O"    # (S,S)-tartaric — enantiomer of RR
MESO = "O[C@@H]([C@@H](O)C(=O)O)C(=O)O"  # meso-tartaric


def test_flagged_enantiomer_not_rescued_by_symmetry():
    # C2-symmetric molecule: no automorphism maps (S,S) onto (R,R) -> reject.
    assert NM.Orthonym._rt_match(RR, SS, True) is False


def test_flagged_diastereomer_rejected():
    assert NM.Orthonym._rt_match(RR, MESO, True) is False


def test_flagged_meso_self_accepted():
    assert NM.Orthonym._rt_match(MESO, MESO, True) is True


def test_flagged_meso_tolerates_full_omission():
    assert NM.Orthonym._rt_match(MESO, "OC(C(O)C(=O)O)C(=O)O", True) is True
