"""The short-name guard shadows the chars/HA guards. Pin it (Task Z2).

`_name_quality_is_acceptable` rejects on a character count in several places.
The FIRST of them,

    if heavy_atoms > 15 and len(name) < heavy_atoms // 2: # chars/HA < ~0.5

is the strictest, and it is written without a division -- so a grep for
`len(name) /` does not find it, and an audit that greps for the formula
attributes its rejections to whichever ratio test happens to be nearby.

A `heavy_atoms > 25 and len(name)/heavy_atoms < 0.45` test used to sit later in
the same function. It could never fire: every name short enough to trip it was
short enough to trip the `// 2` guard first. It was deleted as dead code.

These tests exist so that a future change to the `// 2` threshold cannot
silently resurrect that dead branch's behaviour without someone noticing, and
so the shadowing itself stays documented in an executable form.

UPDATED BY TASK Z3. Two tests here pinned the guards REJECTING names that were
each correct for their own molecule -- deliberately, as a record of the defect
rather than an endorsement of it. Z3 fixed that defect: the character counts
are unchanged and still match, but a name they would discard is now measured
against the real oracle and kept when the measurement proves it denotes the
molecule. Those two tests now assert the fixed behaviour AND, via the rollback
lever, that turning the oracle off restores the old rejection exactly -- so
they still pin the character count itself, which is what this file is for.
"""
import math

import pytest
from rdkit import Chem

from orthonym.decomposition import engine
from orthonym.decomposition.engine import (
    _name_covers_molecule,
    _name_is_proven_complete,
    _name_quality_is_acceptable,
)

CHOLESTEROL = "CC(C)CCCC(C)C1CCC2C1(C)CCC1C2CC=C2CC(O)CCC12C"  # 28 heavy
COA_FRAGMENT = ("CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)"
                "C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCS")           # 48 heavy


def test_deleted_045_test_was_unreachable_for_every_plausible_size():
    """The exhaustive proof, as an assertion rather than a claim in a report.

    For every heavy-atom count the deleted guard applied to (HA > 25), the
    longest name it would have rejected is still short enough for the `// 2`
    guard to have rejected it first. One counterexample would mean the
    deletion changed behaviour.
    """
    for ha in range(26, 4001):
        longest_name_rejected_by_045 = math.ceil(0.45 * ha) - 1
        assert longest_name_rejected_by_045 < ha // 2, (
            f"HA={ha}: a name of {longest_name_rejected_by_045} chars would "
            f"have been rejected by the 0.45 test but NOT by the // 2 test, "
            f"so the deleted branch was reachable after all"
        )


def test_cholesterol_survives_the_short_name_guard(monkeypatch):
    """Task Z3 FIXED the defect this test used to pin.

    Z2 asserted ``is False`` here and said so explicitly -- "pinning it keeps
    the behaviour visible rather than asserting it is desirable". The character
    count is unchanged and still MATCHES; what changed is that it no longer
    gets to be the verdict, because the real oracle
    (``validation/atom_coverage.py``, constitution by InChIKey skeleton)
    overrules it.

    Both directions are asserted, so the new expectation has teeth: with the
    rollback lever off, the old rejection comes straight back, which proves the
    oracle -- and not some unrelated drift -- is what changed the answer.

    SCOPE: ``is True`` here is a claim about COVERAGE, not about preference.
    The fixture SMILES carries no stereocentres, so ``cholesterol`` denotes it
    up to constitution, which is the only question this predicate asks.
    """
    mol = Chem.MolFromSmiles(CHOLESTEROL)
    ha = mol.GetNumHeavyAtoms()
    assert ha == 28

    # The character count still fires -- it was not retuned or removed.
    assert len("cholesterol") < ha // 2, "the // 2 guard is what fires"
    assert len("cholesterol") / ha < 0.45, "the deleted guard would also match"

    # Independent of the predicate under test: OPSIN parse-back + InChIKey
    # skeleton says this name denotes exactly this molecule.
    assert _name_is_proven_complete("cholesterol", mol) is True

    assert _name_quality_is_acceptable("cholesterol", mol) is True

    # Mutation: revert the fix, and the pre-Z3 rejection returns.
    monkeypatch.setenv("ORTHONYM_DECOMP_COVERAGE_ORACLE", "0")
    engine._PROVEN_COMPLETE_CACHE.clear()
    assert _name_quality_is_acceptable("cholesterol", mol) is False


def test_retained_core_name_on_a_large_molecule_still_falls_through():
    """`adenine` naming a 48-heavy-atom cofactor is not accepted.

    Exercises the retained-name ratio branch (0.146 < 0.25 -> fall through),
    which the 40-molecule pipeline trace never reached. Kept so that branch has
    at least one executable witness.
    """
    mol = Chem.MolFromSmiles(COA_FRAGMENT)
    assert mol.GetNumHeavyAtoms() > 40
    assert _name_quality_is_acceptable("adenine", mol) is False


def test_name_covers_molecule_accepts_cholesterol():
    """The two guards disagree, and that disagreement is load-bearing.

    `_name_covers_molecule` ACCEPTS the same name `_name_quality_is_acceptable`
    rejects, because cholesterol has no cleavable bonds -- there is nothing to
    decompose into, so refusing the name would gain nothing. Anyone unifying
    these two guards must preserve this.
    """
    mol = Chem.MolFromSmiles(CHOLESTEROL)
    assert _name_covers_molecule("cholesterol", mol) is True


@pytest.mark.parametrize("name,smiles", [
    # The one chars/HA guard measured to reject on the naming path. Z2 recorded
    # it rejecting this name for being 14 characters long (0.636 < 0.65).
    #
    # ⚠ `ethyl stearate` is NOT the preferred name: the Blue Book puts
    # `(PIN)` on `octadecanoic acid`, making `stearic` the non-PIN alternative.
    # It is used here ONLY as a constitution witness -- it denotes exactly this
    # molecule, so a COVERAGE predicate must not call it partial. Do not read
    # this test as endorsing the name; the PIN question is decided elsewhere.
    ("ethyl stearate", "CCCCCCCCCCCCCCCCCC(=O)OCC"),
])
def test_d04_no_longer_rejects_a_covering_name_for_being_short(name, smiles,
                                                              monkeypatch):
    mol = Chem.MolFromSmiles(smiles)
    ha = mol.GetNumHeavyAtoms()
    assert 15 < ha <= 30, f"D-04 only applies to 15<HA<=30, got {ha}"
    assert len(name) >= ha // 2, "must survive the // 2 guard to reach D-04"
    assert len(name) / ha < 0.65, "and must be under the D-04 threshold"

    assert _name_is_proven_complete(name, mol) is True
    assert _name_quality_is_acceptable(name, mol) is True

    monkeypatch.setenv("ORTHONYM_DECOMP_COVERAGE_ORACLE", "0")
    engine._PROVEN_COMPLETE_CACHE.clear()
    assert _name_quality_is_acceptable(name, mol) is False
