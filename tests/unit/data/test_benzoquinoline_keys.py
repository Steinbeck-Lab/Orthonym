"""benzo[f]quinoline and benzo[h]quinoline in the retained table carry the structures their
names denote.

A fusion name denotes one structure, the Blue Book, the fusion principles):
OPSIN 2.9.0 reads 'benzo[f]quinoline' as c1ccc2c(c1)ccc1ncccc12 (benzo on the quinoline 5,6
bond) and 'benzo[h]quinoline' as c1ccc2c(c1)ccc1cccnc12 (benzo on the 7,8 bond next to N1).
The two SMILES keys of `data/retained_names.py` were swapped: the benzo[h]quinoline input got
the benzo[f] name (a different molecule, refused by the read-back), so the PIN tier abstained.
"""
import pytest
from rdkit import Chem

from orthonym.data.retained_names import get_retained_name
from tests.support.pin_tiers import assert_pin_at_both_tiers
from tests.support.rt_assert import _independent_parse

pytestmark = pytest.mark.opsin_gate

ROWS = [
    ("c1ccc2c(c1)ccc1ncccc12", "benzo[f]quinoline"),
    ("c1ccc2c(c1)ccc1cccnc12", "benzo[h]quinoline"),
]


@pytest.mark.parametrize("smiles,name", ROWS)
def test_retained_key_is_the_structure_of_the_name(smiles, name):
    assert get_retained_name(Chem.CanonSmiles(smiles)) == name
    parsed = _independent_parse(name)
    assert parsed and Chem.CanonSmiles(parsed) == Chem.CanonSmiles(smiles)


@pytest.mark.parametrize("smiles,name", ROWS + [
    ("Cc1ccc2c(ccc3ccccc32)n1", "3-methylbenzo[f]quinoline"),
])
def test_named_at_both_tiers(smiles, name):
    assert_pin_at_both_tiers(smiles, name)
