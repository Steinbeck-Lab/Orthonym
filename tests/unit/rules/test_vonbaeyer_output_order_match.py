"""The canonical copy of a molecule is mapped onto it by the SMILES output order, not by a substructure search.

The search ran for minutes on a large symmetric molecule in native code that the time limit cannot interrupt; the
output order gives the same mapping in linear time, and the search stays as the fallback when the copy differs.
"""

import pytest
from rdkit import Chem

from orthonym.rules.vonbaeyer_universal import _output_order_match, _stereo_aware_match

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles", ["C[C@H](O)CC[C@@H](N)C", "C1CC2CCC1C2", "c1ccc2ccccc2c1", "C/C=C/C=C\\C"])
def test_output_order_is_an_isomorphism(smiles):
    mol = Chem.MolFromSmiles(smiles)
    canon = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    order = _output_order_match(mol, canon)
    assert order is not None and sorted(order) == list(range(mol.GetNumAtoms()))
    # an isomorphism: every bond of the copy is a bond of the molecule under the mapping (for a symmetric molecule it
    # may be another automorphism than the one the search finds)
    for b in canon.GetBonds():
        mb = mol.GetBondBetweenAtoms(order[b.GetBeginAtomIdx()], order[b.GetEndAtomIdx()])
        assert mb is not None and mb.GetBondType() == b.GetBondType()
    assert _stereo_aware_match(mol, canon) == order


def test_no_output_order_falls_back_to_the_search():
    mol = Chem.MolFromSmiles("CCO")
    canon = Chem.MolFromSmiles("OCC")
    assert _output_order_match(mol, canon) is None
    assert sorted(_stereo_aware_match(mol, canon)) == [0, 1, 2]
