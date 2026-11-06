"""The prefixes of a fused heterocycle's substituents name the shape of the group, not its carbon count.

``rules/fused_rings.py`` named an -O-R, -S-R, -NH-R or -R group of 3 to 6 carbons from a table keyed by the number of
carbons, so a branched group took the straight-chain name ('7-butoxy-...' for a tert-butoxy group) and the round trip
withdrew the PIN. The count names an unbranched chain bound by its end only (1), the Blue Book;
``assembly.straight_alkyl``); every other group is named by the substituent namer: 'tert-butoxy' and '(propan-2-yl)oxy'
, the Blue Book,:27683: "(propan-2-yl)oxy (preferred prefix)"), '(propan-2-yl)sulfanyl'.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("CC(C)(C)Oc1ccc2[nH]ccc2c1", "5-tert-butoxy-1H-indole"),
    ("CC(C)(C)Oc1ccc2ncccc2c1", "6-tert-butoxyquinoline"),
    ("Cc1nccc2c1[nH]c1cc(OC(C)(C)C)ccc12", "7-tert-butoxy-1-methyl-9H-beta-carboline"),
    ("CC(C)Oc1ccc2[nH]ccc2c1", "5-[(propan-2-yl)oxy]-1H-indole"),
    ("CC(C)Sc1ccc2[nH]ccc2c1", "5-[(propan-2-yl)sulfanyl]-1H-indole"),
    ("CC(C)(C)Sc1ccc2ncccc2c1", "6-(tert-butylsulfanyl)quinoline"),
    # an unbranched chain keeps its count name
    ("CCCOc1ccc2[nH]ccc2c1", "5-propoxy-1H-indole"),
    ("CCSc1ccc2[nH]ccc2c1", "5-(ethylsulfanyl)-1H-indole"),
    ("Cc1nccc2c1[nH]c1cc(OC)ccc12", "7-methoxy-1-methyl-9H-beta-carboline"),
])
def test_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row
    assert name_is_rt_exact(expected, smiles)
