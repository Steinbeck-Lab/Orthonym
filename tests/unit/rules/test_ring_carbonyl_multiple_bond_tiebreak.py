"""Two ring carbonyl systems with one skeleton: the one with more multiple bonds is senior.

 (the Blue Book): "The senior ring, ring system or principal chain has the greater number of
multiple bonds; for the purpose of this criterion, mancude rings or ring systems are considered as consisting of
noncumulative double bonds". cannot tell a 4H-1-benzopyran-4-one from a 2,3-dihydro-4H-1-benzopyran-4-one
(one skeleton), so a name built on the flavanone beside a flavone is not the PIN
(rules.lactones._more_multiple_bonds_on_same_skeleton in another_ring_ketone_system_is_senior).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("O=C1CC(Cc2cc(=O)c3ccccc3o2)Oc2ccccc21",
     "2-[(4-oxo-4H-1-benzopyran-2-yl)methyl]-2,3-dihydro-4H-1-benzopyran-4-one"),
    ("O=C1CC(c2ccc(-c3cc(=O)c4ccccc4o3)cc2)Oc2ccccc21",
     "2-[4-(4-oxo-4H-1-benzopyran-2-yl)phenyl]-2,3-dihydro-4H-1-benzopyran-4-one"),
    # monocycles: a 4H-pyran-4-one outranks an oxan-2-one (one skeleton, more multiple bonds)
    ("O=C1OCCCC1Cc1coccc1=O", "3-[(4-oxo-4H-pyran-3-yl)methyl]oxan-2-one"),
    ("O=C1CCCC(Cc2cc(=O)cco2)O1", "6-[(4-oxo-4H-pyran-2-yl)methyl]oxan-2-one"),
])
def test_flavanone_parent_beside_flavone_is_not_pin(smiles, name):
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("O=c1cc(-c2ccccc2)oc2ccccc12", "2-phenyl-4H-1-benzopyran-4-one"),
    ("O=C1CC(c2ccccc2)Oc2ccccc21", "2-phenyl-2,3-dihydro-4H-1-benzopyran-4-one"),
])
def test_one_ring_carbonyl_system_keeps_pin(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
