"""RISK 3 (fable v30) — the recovery lane's stereo-flagged RT compare must CATCH a
stereo CONFLICT, not just tolerate omission.

`_stereo_emit_decision` flags an emission `stereo_unexpressed` whenever the name is
stereo-INCOMPLETE — which includes a name that asserts PARTIAL (possibly wrong) stereo.
The old `_rt_match(..., stereo_flagged=True)` stripped stereo from BOTH sides, so any
stereo the name asserted was verified by nobody: a wrong stereoisomer could ship at T3
(inside the T1-T3 0-wrong scope). The fix routes the flagged compare through C6's
RegistrationHash verdict — tolerate OMISSION (count differs), catch CONFLICT (same count,
different config) and any constitution difference.

`_rt_match` has one live caller (namer.py:3185, the best-effort/complete recovery lane),
so this is PIN-default byte-identical.
"""
from orthonym import namer as NM

L = "C[C@H](N)C(=O)O"   # L-alanine
D = "C[C@@H](N)C(=O)O"  # D-alanine — same constitution, OPPOSITE stereo
FLAT = "CC(N)C(=O)O"    # alanine, no stereo specified (omission)


def test_flagged_rejects_stereo_conflict():
    # the hole: a flagged name asserting D-stereo for an L input must NOT ship.
    assert NM.Orthonym._rt_match(L, D, True) is False


def test_flagged_tolerates_stereo_omission():
    # a flagged constitution-only name (no/less stereo) is the sanctioned case -> ship.
    assert NM.Orthonym._rt_match(L, FLAT, True) is True


def test_flagged_accepts_exact_same():
    assert NM.Orthonym._rt_match(L, L, True) is True


def test_flagged_rejects_constitution_difference():
    assert NM.Orthonym._rt_match(L, "c1ccccc1", True) is False


def test_unflagged_isomeric_exact_unchanged():
    # the non-flagged branch stays an EXACT isomeric compare (byte-identical).
    assert NM.Orthonym._rt_match(L, D, False) is False
    assert NM.Orthonym._rt_match(L, L, False) is True
