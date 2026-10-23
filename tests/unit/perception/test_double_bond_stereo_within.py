"""``get_double_bond_stereo(mol, within=atoms)`` returns the entries of the whole molecule's list whose two atoms
are both in ``atoms``, in the same order, found from those atoms' own bonds instead of every bond of the molecule."""
import random

import pytest
from rdkit import Chem

from orthonym.perception.stereo import assign_stereochemistry, get_double_bond_stereo

SMILES = [
    "C/C=C/C(=O)N[C@@H](C)C(=O)O",
    "C/C=C\\C=C\\C=C/C",
    "OC(=O)/C=C/c1ccccc1/C=C\\C(N)=O",
    "CC(C)=CC(=O)NC1CCCCC1",           # a double bond with no geometry
    "C/C=C/C.C/C=C\\C",                # two molecules
]


@pytest.mark.parametrize("smiles", SMILES)
def test_within_equals_the_whole_list_filtered(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assign_stereochemistry(mol)
    full = get_double_bond_stereo(mol)
    rng = random.Random(7)
    n = mol.GetNumAtoms()
    seen_nonempty = False
    for _ in range(60):
        atoms = {i for i in range(n) if rng.random() < 0.7}
        want = [sb for sb in full if sb['atoms'][0] in atoms and sb['atoms'][1] in atoms]
        assert get_double_bond_stereo(mol, within=atoms) == want
        assert get_double_bond_stereo(mol, within=sorted(atoms)) == want   # any collection with `in`
        seen_nonempty |= bool(want)
    assert get_double_bond_stereo(mol, within=set(range(n))) == full
    assert get_double_bond_stereo(mol, within=set()) == []
    if full:
        assert seen_nonempty


def test_a_dict_of_locants_is_a_valid_within():
    # ``_located_stereo_block`` passes its atom -> locant map
    mol = Chem.MolFromSmiles("C/C=C/C(=O)O")
    assign_stereochemistry(mol)
    locant_of = {0: 4, 1: 3, 2: 2, 3: 1}
    assert [sb['stereo'] for sb in get_double_bond_stereo(mol, within=locant_of)] == ['E']
    assert get_double_bond_stereo(mol, within={0: 4, 1: 3}) == []
