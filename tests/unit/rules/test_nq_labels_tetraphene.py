"""The retained PAH 'benz[a]anthracene' is named tetraphene by every route (lane W, item 25).

 "Polyaphenes" (the Blue Book, under "Systematically named hydrocarbon
parent components"): four ortho-fused benzene rings that form two straight arrangements with a
common ring and a formal angle of 120 degrees are 'tetraphene'; the book's own example
'2,6-di(tetraphen-1-yl)pyridine (PIN)' "Structures containing heterocycles",:25762)
replaces '2,6-bis(benzo[a]anthracen-1-yl)pyridine'. The substituted forms and the prefixes
already gave tetraphene (the bridged fused builder); the bare parent and the substituted forms the
polycyclic PAH table named still said benz[a]anthracene."""
import pytest

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

ROWS = [
    ("c1ccc2cc3c(ccc4ccccc43)cc2c1", "tetraphene"),
    ("C1=CC2=CC3=C(C=CC4=CC=CC=C43)C=C2C=C1", "tetraphene"),
    ("Oc1ccc2cc3c(ccc4ccccc43)cc2c1", "tetraphen-9-ol"),
    ("Nc1cc2cc3c(ccc4ccccc43)cc2cc1", "tetraphen-10-amine"),
    ("OC(=O)c1ccc2cc3c(ccc4ccccc43)cc2c1", "tetraphene-9-carboxylic acid"),
    ("O=C1c2ccccc2C(=O)c2c1ccc1ccccc21", "tetraphene-7,12-dione"),
    ("Cc1c2ccccc2c(C)c2c1ccc1ccccc12", "7,12-dimethyltetraphene"),
    ("C1CCc2cc3c(ccc4ccccc43)cc2C1", "8,9,10,11-tetrahydrotetraphene"),
    ("OC(=O)Cc1ccc2cc3c(ccc4ccccc43)cc2c1", "(tetraphen-9-yl)acetic acid"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_tetraphene_at_both_tiers(smiles, expected):
    row = default_tier_row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert name_best_effort(smiles)["name"] == expected
    assert_full_rt(expected, smiles)


def test_the_polycyclic_table_has_one_entry_for_the_ring_system():
    from orthonym.data.polycyclic_data import POLYCYCLIC_DATA
    assert "tetraphene" in POLYCYCLIC_DATA and "benz[a]anthracene" not in POLYCYCLIC_DATA
    assert POLYCYCLIC_DATA["tetraphene"]["canonical_smiles"] == "c1ccc2cc3c(ccc4ccccc43)cc2c1"


def test_benzo_a_pyrene_is_benzo_pqr_tetraphene():
    """ (the Blue Book): 'benzo[pqr]tetraphene (PIN) [tetraphene (3 rings in
    horizontal row) preferred to chrysene or pyrene (2 rings in horizontal row)]'."""
    from orthonym.data.polycyclic_data import POLYCYCLIC_DATA
    assert "benzo[pqr]tetraphene" in POLYCYCLIC_DATA and "benzo[a]pyrene" not in POLYCYCLIC_DATA
    smiles = "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"
    row = default_tier_row(smiles)
    assert (row["name"], row["tier"]) == ("benzo[pqr]tetraphene", "pin_verified"), row
    assert_full_rt("benzo[pqr]tetraphene", smiles)
