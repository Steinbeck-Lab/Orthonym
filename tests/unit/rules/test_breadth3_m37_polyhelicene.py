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
from orthonym.tiers import emit_tier_flags
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
    row = Orthonym(**emit_tier_flags("pin")).name_tiered(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified", row
    assert _key(_independent_parse(expected)) == _key(smiles)


def test_class_numbering_is_the_catalogued_hexahelicene_one():
    # the class of rules/hydrocarbon_series.py numbers hexahelicene as the
    # hand-entered row does, atom for atom up to the symmetry of the skeleton; heptahelicene
    # and the longer members have no row, the class serves them
    from orthonym.rules.hydrocarbon_series import series_entry
    cat = POLYCYCLIC_DATA['hexahelicene']
    mine = series_entry('hexahelicene')
    a, b = Chem.MolFromSmiles(cat['canonical_smiles']), Chem.MolFromSmiles(mine['canonical_smiles'])
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    query = Chem.AdjustQueryProperties(a, params)
    la = {i: str(v) for i, v in cat['iupac_numbering'].items()}
    lb = {i: str(v) for i, v in mine['iupac_numbering'].items()}
    assert any(all(la[i] == lb[m[i]] for i in la)
               for m in b.GetSubstructMatches(query, uniquify=False, maxMatches=1000))
    assert all(f'{n}helicene' not in POLYCYCLIC_DATA for n in ('hepta', 'octa', 'nona', 'deca', 'undeca', 'dodeca'))
