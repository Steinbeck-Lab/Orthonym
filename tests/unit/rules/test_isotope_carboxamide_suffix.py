"""A nuclide of a carboxamide group goes before the 'carboxamide' suffix, as for the other 'carbo' suffixes.

 (heading 'NAMES', the Blue Book): the nuclide symbol goes "before the part of the compound
that is isotopically substituted"; (:44256): 'benzene(13C)carbonitrile (PIN)', 'benzene(13C)carboxylic
acid (PIN)'. The carbocyclic path used to put the descriptor in front of the whole name
('(15N)naphthalene-2-carboxamide', pin_verified) while the heterocyclic one wrote 'quinoline-6-(15N)carboxamide'.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("[15NH2]C(=O)c1ccc2ccccc2c1", "naphthalene-2-(15N)carboxamide"),
    ("C[15NH]C(=O)c1ccc2ccccc2c1", "N-methylnaphthalene-2-(15N)carboxamide"),
    ("[15NH2]C(=O)c1ccc2ncccc2c1", "quinoline-6-(15N)carboxamide"),
    ("[15NH2]C(=O)C1CCCCC1", "cyclohexane(15N)carboxamide"),
    ("[15NH2][13C](=O)C1CCCCC1", "cyclohexane(13C,15N)carboxamide"),
])
def test_carboxamide_nuclide_before_the_suffix(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_a_ring_carbon_label_keeps_its_locant_in_front():
    row = tier_row("NC(=O)c1ccc2cc[13cH]cc2c1", "pin")
    assert row["name"] == "(7-13C)naphthalene-2-carboxamide", row
