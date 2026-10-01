import pytest
from rdkit import Chem

from orthonym.perception.automorphisms import skeleton_automorphisms


def _all(smiles, blind=True, cap=1000):
    m = Chem.MolFromSmiles(smiles)
    atoms = {a.GetIdx() for a in m.GetAtoms()}
    return skeleton_automorphisms(m, atoms, element_blind=blind, cap=cap)


@pytest.mark.unit
@pytest.mark.parametrize("smiles,n", [("C12C3C4C1C1C2C3C41", 48),   # cubane |Oh| = 48
                                      ("C1C2CC3CC1CC(C2)C3", 24),   # adamantane |Td| = 24
                                      ("c1ccc2ccccc2c1", 4)])       # naphthalene skeleton D2h -> 4
def test_automorphism_counts(smiles, n):
    auts, exhaustive = _all(smiles)
    assert exhaustive and len(auts) == n


@pytest.mark.unit
def test_cap_marks_non_exhaustive():
    auts, exhaustive = _all("C12C3C4C1C1C2C3C41", cap=10)
    assert len(auts) == 10 and exhaustive is False


@pytest.mark.unit
def test_element_aware_breaks_symmetry():
    auts, _ = _all("C1COCCN1", blind=False)      # morpholine: O and N fixed -> 2 (reflection)
    assert len(auts) == 2
