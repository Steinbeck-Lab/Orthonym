"""A partly saturated fused carbocycle never cites its principal characteristic group as a prefix.

 'Definitions' (the Blue Book): functional suffixes "are exclusive because only one can be placed at
the end of a name to represent the principal characteristic group". The partly saturated carbocycle producer
writes only some suffixes (rules.partial_saturation._PARTIAL_SAT_PCG_SUFFIX) and cited any other principal group
as a prefix: '6-(chlorosulfonyl)-1,2,3,4-tetrahydronaphthalene' was labelled PIN once 'chlorosulfonyl' became
nameable as a prefix. The PIN is 5,6,7,8-tetrahydronaphthalene-2-sulfonyl chloride.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles", [
    "O=S(=O)(Cl)c1ccc2c(c1)CCCC2",
    "OS(=O)(=O)c1ccc2c(c1)CCCC2",
    "NS(=O)(=O)c1ccc2c(c1)CCCC2",
])
def test_no_principal_group_as_a_prefix(smiles):
    row = tier_row(smiles, "pin")
    name = row["name"] or ""
    assert row["tier"] != "pin_verified" or not name.endswith("naphthalene"), row


@pytest.mark.parametrize("smiles,name", [
    ("OC(=O)c1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalene-2-carboxylic acid"),
    ("Oc1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalen-2-ol"),
    ("COc1ccc2c(c1)CCCC2", "6-methoxy-1,2,3,4-tetrahydronaphthalene"),
    ("O=C(Cl)c1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalene-2-carbonyl chloride"),
    ("O=C1CCCc2ccccc21", "3,4-dihydronaphthalen-1(2H)-one"),
])
def test_names_kept(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
