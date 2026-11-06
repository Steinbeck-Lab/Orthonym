"""A Group 13 atom joined to a parent by a multiple bond is a prefix by skeletal replacement (lane l18 item 4).

The capped fragment ``C=[Al]`` is named as a coordination compound ('methylidenealuminium'), which has no '-yl' form:
'3-(methylidenealuminiumyl)-5-methyl-1,2,4-oxadiazole' reads as no structure in OPSIN. Admitting the metal to
``BONDED_NAMES`` (lanes B1 and B2, 2026-10-30) let that prefix reach the substituent cascade first and void the later
tiers, so every such drawing lost the name it had had ('3-(1-aluminaeth-1-en-1-yl)-5-methyl-1,2,4-oxadiazole').

 ORGANOMETALLIC COMPOUNDS INVOLVING THE ELEMENTS OF GROUPS 13 THROUGH 16 (the Blue Book): "named
substitutively by prefixing the appropriate substituent names to the name of a parent hydride"; the metal joined to a
parent takes a prefix of its parent hydride or of skeletal replacement ('a' prefixes,, never the
coordination name with '-yl'. No PIN: PREFERRED IUPAC NAMES (the Blue Book) lists no Group 13 element.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

ROWS = [
    ("CC1=NC(=NO1)[Al]=C", "3-(1-aluminaeth-1-en-1-yl)-5-methyl-1,2,4-oxadiazole"),
    ("CCCCCC(CCCN(C)C)[Al]=O", "1-(dimethylamino)-4-(2-oxa-1-aluminaeth-1-en-1-yl)nonane"),
    ("C1=CC(=CC=C1O)S(=O)(=O)CC[Tl]", "4-[(3λ1-thallapropyl)sulfonyl]benzen-1-ol"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_metal_prefix_by_skeletal_replacement(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected and res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res
