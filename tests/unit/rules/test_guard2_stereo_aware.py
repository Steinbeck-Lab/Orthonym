""" #36 (a review F6 finding 3) — guard 2 of ``_acid_stem_unsaturated_oxide_prefix``
re-anchors the acid sub-namer's output by CONSTITUTIONAL skeleton only (InChIKey
first block, stereo-free), so a WRONG CIP descriptor (E/Z, R/S) from the
gate-disabled sub-namer would ship a wrong-stereo ``...sulfinyl/sulfonyl`` prefix.
Guard 2 now also requires the C6 RegistrationHash stereo layer to match.

There is no live witness (the sub-namer routes stereo through the standard CIP
path), so these tests pin the defence-in-depth: correct stereo still emits, and a
synthetic wrong descriptor fails closed.
"""
import pytest
from rdkit import Chem

from orthonym.rules.sulfur import _acid_stem_unsaturated_oxide_prefix
from orthonym.namer import Orthonym

pytestmark = pytest.mark.unit


def _arm(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    s = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'S'][0]
    c = [n.GetIdx() for n in mol.GetAtomWithIdx(s).GetNeighbors()
         if n.GetSymbol() == 'C' and n.GetDegree() > 1][0]
    return mol, c, s


def test_correct_ez_prefix_still_emits():
    """The stereo-aware guard must NOT fail-close a correct E/Z arm."""
    molE, cE, sE = _arm("C/C=C/S(=O)(=O)C")
    molZ, cZ, sZ = _arm("C/C=C\\S(=O)(=O)C")
    assert _acid_stem_unsaturated_oxide_prefix(molE, cE, sE, "sulfonyl") == "(1E)-prop-1-ene-1-sulfonyl"
    assert _acid_stem_unsaturated_oxide_prefix(molZ, cZ, sZ, "sulfonyl") == "(1Z)-prop-1-ene-1-sulfonyl"


def test_injected_wrong_stereo_fails_closed(monkeypatch):
    """If the acid sub-namer emits a WRONG descriptor, guard 2 rejects it."""
    mol, c, s = _arm("C/C=C/S(=O)(=O)C")
    orig = Orthonym.name

    def wrong(self, smi):
        n = orig(self, smi)
        if isinstance(n, str) and n.endswith("sulfonic acid") and "1E" in n:
            return n.replace("1E", "1Z")  # flip the descriptor
        return n

    monkeypatch.setattr(Orthonym, "name", wrong)
    assert _acid_stem_unsaturated_oxide_prefix(mol, c, s, "sulfonyl") is None
