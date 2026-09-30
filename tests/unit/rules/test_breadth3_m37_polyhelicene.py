"""Breadth job 3, class M37: the polyhelicene series, the Blue Book
"... named by citing a numerical prefix ('hexa', 'hepta', etc.) denoting the total
number of benzene rings forming a helical arrangement followed by the term
'helicene'"; 'hexahelicene (PIN)':11483, numbering. Only hexahelicene
was catalogued, so the dev2000 [9]helicene row abstained at every tier.

Every name is read back by a FRESH OPSIN call that does not go through the engine
(tests.support.rt_assert._independent_parse) and compared by full InChIKey.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.data.polycyclic_data import POLYCYCLIC_DATA
from tests.support.rt_assert import _independent_parse

pytestmark = [pytest.mark.opsin_gate]


def _key(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToInchiKey(mol) if mol is not None else ""


@pytest.mark.parametrize("smiles,expected", [
    ("c1ccc2c(c1)ccc1ccc3ccc4ccc5ccccc5c4c3c12", "hexahelicene"),     # unchanged
    ("c1ccc2c(c1)ccc1ccc3ccc4ccc5ccc6ccccc6c5c4c3c12", "heptahelicene"),
    # the dev2000 M37 row
    ("c1ccc2c(c1)ccc1ccc3ccc4ccc5ccc6ccc7ccc8ccccc8c7c6c5c4c3c12", "nonahelicene"),
    ("Cc1ccc2c(c1)ccc1ccc3ccc4ccc5ccc6ccccc6c5c4c3c12", "3-methylheptahelicene"),
])
def test_polyhelicene(smiles, expected):
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified", row
    assert _key(_independent_parse(expected)) == _key(smiles)


def test_generated_hexahelicene_numbering_is_the_catalogued_one():
    # the generator reproduces the hand-verified hexahelicene entry atom for atom
    from orthonym.data.polycyclic_data import _helicene_entry, _HELICENE_PREFIX
    _HELICENE_PREFIX[6] = 'hexa'
    try:
        name, data = _helicene_entry(6)
    finally:
        del _HELICENE_PREFIX[6]
    assert name == 'hexahelicene'
    cat = POLYCYCLIC_DATA['hexahelicene']
    assert data['canonical_smiles'] == cat['canonical_smiles']
    assert data['iupac_numbering'] == cat['iupac_numbering']
