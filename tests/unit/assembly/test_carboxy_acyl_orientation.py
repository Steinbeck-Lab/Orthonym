"""a review review 2026-09-08: a bare carboxyl -C(=O)OH attached at its OWN carbon
is the 'carboxy' prefix, NOT 'formyl'. The Step-4/5 recursion named the capped
fragment 'formic acid' and parent_to_prefix applied the acid->acyl table
('formic acid'->'formyl'), which has no free-valence orientation and DROPS the
-OH -- a wrong constitution (-CHO). Cite: the Blue Book (carboxy prefix).
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_naming import name_substituent_fragment
from orthonym import name_compound


@pytest.mark.unit
def test_bare_carboxyl_is_carboxy_not_formyl():
    mol = Chem.MolFromSmiles("CC(=O)O")  # atom1 = carboxyl C, 2 = =O, 3 = -OH
    got = name_substituent_fragment(mol, [1, 2, 3], 1, [0])
    assert got == "carboxy", f"expected carboxy, got {got!r} (formyl drops the -OH)"


@pytest.mark.unit
def test_aldehyde_still_formyl():
    """-CHO must still be formyl -- the guard is scoped to the 3-atom carboxyl."""
    mol = Chem.MolFromSmiles("O=CC")  # atom1 = CHO carbon, 0 = =O
    assert name_substituent_fragment(mol, [0, 1], 1, [2]) == "formyl"


@pytest.mark.unit
@pytest.mark.parametrize(
    "smi,expected",
    [
        ("CC(=O)O", "acetic acid"),
        ("OC(=O)Cc1ccccc1", "phenylacetic acid"),
        ("CC(=O)c1ccccc1", "1-phenylethan-1-one"),
    ],
)
def test_whole_molecule_unaffected(smi, expected):
    assert name_compound(smi) == expected
