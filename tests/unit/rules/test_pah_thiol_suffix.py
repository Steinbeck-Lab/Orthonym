"""A ring -SH of a polycyclic arene parent: the '-thiol' suffix when it is the principal group, else 'sulfanyl'.

 class 17 (the Blue Book): hydroxy compounds 'and chalcogen analogues', after -OH and before amines (class 19,:18192).
These rows had no PIN-tier name: the arene writer named no ring -SH at all.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("Sc1ccc2ccccc2c1", "naphthalene-2-thiol"),
    ("Sc1c2ccccc2cc2ccccc12", "anthracene-9-thiol"),
    ("Sc1ccc2ccc(S)cc2c1", "naphthalene-2,7-dithiol"),
    ("Cc1ccc2ccc(S)cc2c1", "7-methylnaphthalene-2-thiol"),
    ("Nc1ccc2ccc(S)cc2c1", "7-aminonaphthalene-2-thiol"),
    ("Oc1ccc2ccc(S)cc2c1", "7-sulfanylnaphthalen-2-ol"),
    ("OC(=O)c1ccc2ccc(S)cc2c1", "7-sulfanylnaphthalene-2-carboxylic acid"),
])
def test_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row
