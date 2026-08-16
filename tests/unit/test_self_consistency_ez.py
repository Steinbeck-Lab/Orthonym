"""C6 (v30) — `_self_consistency_verdict` must catch an EQUAL-count stereo CONFLICT even on a
conjugated/aromatic double bond, where the RegistrationHash TAUTOMER_HASH layer is E/Z-blind
(fumarate vs maleate hash equal). The fix decides an equal-count, same-EXACT-constitution case
by a sound per-element atom-mapped compare, while preserving tautomer/mobile-H tolerance (the
gate is deliberately tautomer-insensitive).

v27 Phase 0 ABANDONED stereo-omission tolerance: a name whose OPSIN re-perception specifies
FEWER stereo features than the input (nb < na) describes a less-specific WRONG molecule, so
`_name_omits_input_stereo` (namer.py) makes it a "mismatch", not an "ok". This module's tests
reflect that current contract — see `test_ez_omission_is_mismatch` /
`test_tetrahedral_omission_is_mismatch` below (this docstring previously claimed the opposite,
"a less-specific name is valid", which was already dead by the time it was written here).

v33 Phase 1 (C3): the na==0 case (stereo-unspecified input vs a stereo-implying name) is
likewise NOT tolerated — it FABRICATES configuration the input does not define (P-92 "CIP
Priority and Sequence Rules" / P-93 "Configuration Specification", BlueBookV2.md:44525/:44527)
and is now a "mismatch". See tests/unit/test_gate_stereo_fabrication.py.
`_self_consistency_verdict` is the SELF-01 stereo decision on the DEFAULT (PIN) path.
"""
from orthonym import namer as NM

V = NM._self_consistency_verdict


# --- the fix: EQUAL-count E/Z conflict on a conjugated bond must be a mismatch --------------

def test_fumarate_vs_maleate_ez_conflict_is_mismatch():
    # but-2-enedioic acid: (E)=fumaric vs (Z)=maleic — DISTINCT compounds, same constitution.
    assert V("OC(=O)/C=C/C(=O)O", "OC(=O)/C=C\\C(=O)O") == "mismatch"


def test_crotonic_ez_conflict_is_mismatch():
    assert V("C/C=C/C(=O)O", "C/C=C\\C(=O)O") == "mismatch"


def test_cinnamic_ez_conflict_is_mismatch():
    assert V("c1ccccc1/C=C/C(=O)O", "c1ccccc1/C=C\\C(=O)O") == "mismatch"


# --- preserved behaviors --------------------------------------------------------------------

def test_ez_omission_is_mismatch():
    # v27 Phase 0: name leaves the double-bond geometry unspecified -> the OPSIN
    # re-perception describes a LESS-SPECIFIC (wrong) molecule -> rejected, not
    # tolerated.
    assert V("OC(=O)/C=C/C(=O)O", "OC(=O)C=CC(=O)O") == "mismatch"


def test_ez_exact_is_ok():
    assert V("OC(=O)/C=C/C(=O)O", "OC(=O)/C=C/C(=O)O") == "ok"


def test_tetrahedral_conflict_is_mismatch():
    assert V("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O") == "mismatch"


def test_tetrahedral_exact_is_ok():
    assert V("C[C@H](N)C(=O)O", "C[C@H](N)C(=O)O") == "ok"


def test_tetrahedral_omission_is_mismatch():
    # v27 Phase 0: omission is rejected, not tolerated (this test used to encode
    # the pre-v27 "ok" contract, which was already dead here).
    assert V("C[C@H](N)C(=O)O", "CC(N)C(=O)O") == "mismatch"


def test_unspecified_input_fabricated_stereo_is_mismatch():
    # v33 Phase 1 (C3): na==0 (flat input) vs a stereo-asserting name FABRICATES
    # configuration -> mismatch (was tolerated pre-C3).
    assert V("CC(C)CC([NH3+])C(=O)O", "[NH3+][C@@H](CC(C)C)C(=O)O") == "mismatch"


def test_constitution_difference_is_mismatch():
    assert V("C[C@H](N)C(=O)O", "c1ccccc1") == "mismatch"


def test_tautomer_same_stereo_is_ok():
    # 2-hydroxypyridine vs 2-pyridone — mobile-H tautomers, must stay tolerated.
    assert V("Oc1ccccn1", "O=c1cccc[nH]1") == "ok"


def test_ignore_stereo_carveout_unaffected():
    # BBR-GATE stereo carve-out compares constitution only -> E/Z conflict is still "ok".
    assert V("OC(=O)/C=C/C(=O)O", "OC(=O)/C=C\\C(=O)O", ignore_stereo=True) == "ok"
