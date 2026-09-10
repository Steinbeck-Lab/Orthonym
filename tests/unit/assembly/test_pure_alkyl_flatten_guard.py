import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import _fragment_is_saturated_acyclic


@pytest.mark.unit
def test_saturated_acyclic_is_flattenable():
    mol = Chem.MolFromSmiles("CCCC")  # butane
    assert _fragment_is_saturated_acyclic(mol, [0, 1, 2, 3]) is True


@pytest.mark.unit
def test_unsaturated_is_not_flattenable():
    mol = Chem.MolFromSmiles("C=CCC")  # but-1-ene: a C=C in the fragment
    assert _fragment_is_saturated_acyclic(mol, [0, 1, 2, 3]) is False


@pytest.mark.unit
def test_cyclic_is_not_flattenable():
    mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane: ring atoms
    assert _fragment_is_saturated_acyclic(mol, [0, 1, 2, 3, 4, 5]) is False
