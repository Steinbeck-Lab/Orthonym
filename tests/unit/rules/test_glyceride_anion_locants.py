"""Different anions on glycerol: the anion cited first takes the lowest locants.

 (the Blue Book): "names of anions are cited in alphanumerical order preceded
by a locant" -- method (1), the PIN -- with the exact molecule printed at:31840 'propane-1,2,3-triyl
1,2-diacetate 3-propanoate (PIN)'. Both numberings of the glycerol give the ester set {1,2,3}; the
tie goes to the anion cited first, as for a prefix cited first (g),:3307), and the book's
three-anion row agrees: '... 2-acetate 1-hexadecanoate 3-[(9Z)-octadec-9-enoate] (PIN)' (:31846).
The engine numbered by atom order ('2,3-diacetate 1-propanoate'). OPSIN 2.9.0 parses no multi-anion
locant ester name, so the label stays below pin_verified; the assertion is on the producer's
string, whatever the input atom order.
"""
import pytest
from rdkit import Chem

from orthonym.perception.lipids import detect_lipid_backbone
from orthonym.rules.lipids import _assemble_glyceride


def _name(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return _assemble_glyceride(mol, detect_lipid_backbone(mol), "pin")


@pytest.mark.parametrize("smiles", [
    "CCC(=O)OCC(COC(C)=O)OC(C)=O",
    "CC(=O)OCC(OC(C)=O)COC(=O)CC",
    "CC(=O)OC(COC(=O)CC)COC(C)=O",
])
def test_first_cited_anion_takes_the_lowest_locants(smiles):
    assert _name(smiles) == "propane-1,2,3-triyl 1,2-diacetate 3-propanoate"


@pytest.mark.parametrize("smiles", [
    "CCCCCCCCCCCCCCCC(=O)OCC(OC(C)=O)COC(=O)CCCCCCC/C=C\\CCCCCCCC",
    "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(OC(C)=O)COC(=O)CCCCCCCCCCCCCCC",
])
def test_the_three_anion_row(smiles):
    assert _name(smiles) == "propane-1,2,3-triyl 2-acetate 1-hexadecanoate 3-[(9Z)-octadec-9-enoate]"


def test_identical_anions_unchanged():
    assert _name("CC(=O)OCC(COC(C)=O)OC(C)=O") == "propane-1,2,3-triyl triacetate"
