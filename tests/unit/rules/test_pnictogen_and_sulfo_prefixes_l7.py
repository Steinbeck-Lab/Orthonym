"""P and As groups and a junior sulfonic acid as prefixes beside a carboxylic acid, on chains, benzene and naphthalene.

'4-(dihydroxyarsanyl)benzoic acid (PIN) [-COOH is senior to -As(OH)2]' (the Blue Book); 'dihydroxyphosphanyl
(preselected prefix)',:36247); '3-(hydroxyphosphanyl)propanoic acid (PIN)' (:27272). (:18170): a
carboxylic acid is senior to a sulfonic acid, which is then the prefix 'sulfo'.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("OP(O)c1ccc(C(=O)O)cc1", "4-(dihydroxyphosphanyl)benzoic acid"),
    ("O[As](O)c1ccc(C(=O)O)cc1", "4-(dihydroxyarsanyl)benzoic acid"),
    ("OC(=O)CCP(O)O", "3-(dihydroxyphosphanyl)propanoic acid"),
    ("OC(=O)CCPO", "3-(hydroxyphosphanyl)propanoic acid"),
    ("OC(=O)CC[As](O)O", "3-(dihydroxyarsanyl)propanoic acid"),
    ("OP(O)c1ccc2ccccc2c1C(=O)O", "2-(dihydroxyphosphanyl)naphthalene-1-carboxylic acid"),
    ("O[As](O)c1ccc2ccccc2c1C(=O)O", "2-(dihydroxyarsanyl)naphthalene-1-carboxylic acid"),
    ("OS(=O)(=O)c1ccc2ccccc2c1C(=O)O", "2-sulfonaphthalene-1-carboxylic acid"),
    ("OS(=O)(=O)c1ccc2ccccc2c1", "naphthalene-2-sulfonic acid"),
], ids=lambda v: v)
def test_pin_names(smiles, name):
    row = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row
