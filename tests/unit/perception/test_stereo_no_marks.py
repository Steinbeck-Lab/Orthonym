"""``assign_stereochemistry`` makes no call to the centres labeller for a molecule that carries no stereo.

centres labels what the stereo marks of its input SMILES say, and a molecule with no chiral tag, no bond stereo
flag (``STEREOANY`` included) and no bond direction writes none: it would label nothing. The rest of the
centres branch runs as after an empty label map, so the labels are those of the call; only the JVM call is not
made (and, in a new process, the first one, which loads the centres classes).
"""
import pytest
from rdkit import Chem

from orthonym.perception import centres_bridge, stereo

pytestmark = pytest.mark.unit


@pytest.fixture
def calls(monkeypatch):
    made = []
    real = centres_bridge.centres_label_mol

    def spy(mol):
        made.append(Chem.MolToSmiles(mol))
        return real(mol)
    monkeypatch.setattr(centres_bridge, "centres_label_mol", spy)
    monkeypatch.setattr(stereo, "_USE_CENTRES_CIP", True)
    return made


def _labels(mol):
    return ([(a.GetIdx(), a.GetProp("_CIPCode")) for a in mol.GetAtoms() if a.HasProp("_CIPCode")],
            [(b.GetIdx(), b.GetProp("_CIPCode")) for b in mol.GetBonds() if b.HasProp("_CIPCode")])


@pytest.mark.parametrize("smiles", ["CCO", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CC=CC", "OC1CCC(O)CC1"])
def test_no_stereo_marks_no_centres_call(calls, smiles):
    mol = Chem.MolFromSmiles(smiles)
    stereo.assign_stereochemistry(mol)
    assert calls == []
    assert mol.HasProp(stereo._CIP_ASSIGNED_PROP)
    assert _labels(mol) == ([], [])


@pytest.mark.parametrize("smiles", ["C[C@H](N)C(=O)O", "C/C=C/C", "O[C@H]1CC[C@@H](O)CC1"])
def test_stereo_marks_go_to_centres(calls, smiles):
    stereo.assign_stereochemistry(Chem.MolFromSmiles(smiles))
    assert len(calls) == 1


def test_a_bond_direction_alone_or_stereoany_goes_to_centres(calls):
    mol = Chem.RWMol(Chem.MolFromSmiles("CC=CC"))
    mol.GetBondWithIdx(0).SetBondDir(Chem.BondDir.ENDUPRIGHT)
    stereo.assign_stereochemistry(mol)
    mol = Chem.RWMol(Chem.MolFromSmiles("CC=CC"))
    mol.GetBondWithIdx(1).SetStereo(Chem.BondStereo.STEREOANY)
    stereo.assign_stereochemistry(mol)
    assert len(calls) == 2
