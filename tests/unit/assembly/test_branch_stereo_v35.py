""" Track A: recursive branch-stereo emission on the universal floor.

A stereocentre that sits INSIDE a substituent branch (not on the top-level
parent spine) must be expressed so the emitted best-effort name round-trips to
the FULL input InChIKey (constitution AND stereo), not merely block-1. Before
 the floor prepended a stereodescriptor block for the top spine only, so a
branch stereocentre was silently dropped and the name failed full-InChIKey RT.

Each test asserts the 0-wrong-safe outcome directly: the floor's own emitted
name, parsed by OPSIN, yields the SAME full InChIKey as the input -- i.e.
``verify_or_none`` (the exact gate the producer uses) returns non-None. The
production change that makes these fail before: nothing downstream of the
per-branch recursion ever emits the branch's own stereodescriptors.
"""
from rdkit import Chem

from orthonym.assembly.universal_substituent import name_universal_substitutive
from orthonym.validation.reconstruct import verify_or_none


def _floor_full_rt(smiles: str):
    """(verified_name_or_None, emitted_name) for the universal-floor producer."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparseable test SMILES: {smiles}"
    res = name_universal_substitutive(mol, 400)
    assert res is not None and res.name, f"floor produced no name for {smiles}"
    return verify_or_none(res.name, Chem.MolToSmiles(mol)), res.name


def test_neutral_branch_rs_centre_full_roundtrips():
    # one R/S centre inside the oxapentenyl branch, off the aromatic parent spine
    verified, name = _floor_full_rt("C=C(C)[C@@H](O)COc1ccc(CC(=O)OC)cc1")
    assert verified is not None, f"branch R/S dropped: {name}"


def test_zwitterion_branch_rs_centre_full_roundtrips():
    # dopa zwitterion: R/S centre inside the charged aminopropanoate branch
    verified, name = _floor_full_rt("[NH3+][C@H](Cc1ccc(O)c(O)c1)C(=O)[O-]")
    assert verified is not None, f"charged branch R/S dropped: {name}"


def test_branch_ez_double_bond_full_roundtrips():
    # E/Z double bond stereo inside the butenyl branch, off the oxazole parent
    verified, name = _floor_full_rt(r"C[C@H](O)/C(C=O)=C\c1cocn1")
    assert verified is not None, f"branch E/Z dropped: {name}"


def test_stereo_rebuild_never_abstains_at_tight_budget():
    # The stereo cascade rebuilds the component tree once more; that rebuild
    # must NOT consume the plain build's already-spent budget and abstain.
    # A stereo molecule whose CONSTITUTION is nameable within the budget must
    # still return SOME name (degrade to plain), never None. (Adversarial
    # reviewer repro: this molecule's constitution needs budget ~14; the stereo
    # rebuild on the shared depleted budget raised _BudgetExceeded at 21.)
    mol = Chem.MolFromSmiles("C[C@H](Cl)[C@@H](C)[C@H](C)[C@@H](C)Cl")
    res = name_universal_substitutive(mol, 21)
    assert res is not None and res.name, "stereo rebuild abstained at tight budget"


def test_stereo_captured_at_default_budget():
    # sanity: at the default budget the same molecule's stereo IS expressed
    # (full-InChIKey round-trip), proving the fresh-budget rebuild still works.
    verified, name = _floor_full_rt("C[C@H](Cl)[C@@H](C)[C@H](C)[C@@H](C)Cl")
    assert verified is not None, f"stereo not captured at default budget: {name}"
