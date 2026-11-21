"""More ring suffixes on a partly saturated fused carbocycle: '5,6,7,8-tetrahydronaphthalene-2-carbonitrile'.

 (the Blue Book): a functional suffix represents "the principal characteristic group";
(:34720) '-carbonitrile' for a nitrile on a ring; (:24890) the suffix takes the lowest locants before the
hydro prefixes. The partly saturated path spelled only acid, alcohol, hydroperoxide and amine suffixes
(rules.partial_sat_group_suffix).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("N#Cc1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalene-2-carbonitrile"),
    ("O=Cc1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalene-2-carbaldehyde"),
    ("NC(=O)c1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalene-2-carboxamide"),
    ("NS(=O)(=O)c1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalene-2-sulfonamide"),
    ("OS(=O)(=O)c1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalene-2-sulfonic acid"),
    ("OS(=O)c1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalene-2-sulfinic acid"),
    ("O=S(=O)(Cl)c1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalene-2-sulfonyl chloride"),
    ("N#Cc1cc(O)c2c(c1)CCCC2", "4-hydroxy-5,6,7,8-tetrahydronaphthalene-2-carbonitrile"),
    # unchanged
    ("OC(=O)c1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalene-2-carboxylic acid"),
    ("Oc1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalen-2-ol"),
    ("N#CCc1ccc2c(c1)CCCC2", "(5,6,7,8-tetrahydronaphthalen-2-yl)acetonitrile"),
])
def test_partial_sat_group_suffix_pins(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
