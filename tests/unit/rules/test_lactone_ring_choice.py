"""A lactone ring is a pseudoketone: beside a ring ketone in another ring system, picks the parent.

 (the Blue Book) puts lactones with the pseudoketones (class 16); (:29628): "There is no
seniority order difference between ketones and pseudoketones... between rings and ring systems, are considered"
('4-(4-oxocyclohexyl)oxolan-2-one (PIN)',:29648). With nitrogen in both rings, (e) (more skeletal atoms,
:19415) decides, so the azepanone is the parent (rules.lactones.lactone_outranked in classify_parts).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("O=C1CCCNCC1C1NCCOC1=O", "3-(2-oxomorpholin-3-yl)azepan-4-one"),
    ("O=C1CCCN(C2NCCOC2=O)CCC1", "1-(2-oxomorpholin-3-yl)azocan-5-one"),
    # a lactam C=O is a pseudoketone too class 16,:18189)
    ("O=C1CCCC(C2NCCOC2=O)CN1", "6-(2-oxomorpholin-3-yl)azepan-2-one"),
    # unchanged: the heterocycle is senior to the carbocycle (a))
    ("O=C1CCC(C2CCOC2=O)CC1", "3-(4-oxocyclohexyl)oxolan-2-one"),
    ("O=C1OCCC1Cc1ccccc1", "3-benzyloxolan-2-one"),
])
def test_lactone_beside_ring_ketone(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
