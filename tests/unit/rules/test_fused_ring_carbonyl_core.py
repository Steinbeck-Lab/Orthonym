"""A fused-ring name built on a ring carbonyl core that another ring carbonyl system outranks is not the PIN.

 (the Blue Book): "There is no seniority order difference between ketones and pseudoketones...
between rings and ring systems, are considered". (d) (:19415): the system with more rings is senior.
The fused-core match takes the first catalog core it finds (a coumarin beside a naphthopyranone);
rules.lactones.ring_carbonyl_core_checked labels that name below the PIN (best-effort keeps it).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("O=C1Oc2ccccc2C=C1c1cc(=O)c2ccc3ccccc3c2o1", "3-(4-oxo-4H-naphtho[1,2-b]pyran-2-yl)-2H-1-benzopyran-2-one"),
    ("O=C1Oc2ccccc2C=C1C1CC(=O)c2cc3ccccc3cc2O1",
     "3-(4-oxo-3,4-dihydro-2H-naphtho[2,3-b]pyran-2-yl)-2H-1-benzopyran-2-one"),
    ("O=C1Oc2ccccc2C=C1[C@@H]1CC(=O)c2cc3ccccc3cc2O1",
     "3-[(2S)-4-oxo-3,4-dihydro-2H-naphtho[2,3-b]pyran-2-yl]-2H-1-benzopyran-2-one"),
    ("O=C1Oc2ccccc2C=C1C1CC(=O)c2c(O1)ccc1c2ccc2ccccc21",
     "3-(4-oxo-3,4-dihydro-2H-phenanthro[2,1-b]pyran-2-yl)-2H-1-benzopyran-2-one"),
])
def test_core_outranked_by_senior_ring_carbonyl_system_is_not_pin(smiles, name):
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # the senior naphthopyranone is the parent (another producer, once the coumarin name is labelled)
    ("O=C1Oc2ccccc2C=C1Cc1coc2c(c1=O)ccc1ccccc12",
     "3-[(2-oxo-2H-1-benzopyran-3-yl)methyl]-4H-naphtho[1,2-b]pyran-4-one"),
    # a heterocycle outranks a carbocycle (a),:19412): the coumarin stays the parent
    ("O=C1Oc2ccccc2C=C1C1CCC(=O)CC1", "3-(4-oxocyclohexyl)-2H-1-benzopyran-2-one"),
    ("O=C1CCC(c2ccc3ccc(=O)oc3c2)CC1", "7-(4-oxocyclohexyl)-2H-1-benzopyran-2-one"),
    ("O=C1Oc2ccccc2C=C1C1CCC(=O)c2ccccc21", "3-(4-oxo-1,2,3,4-tetrahydronaphthalen-1-yl)-2H-1-benzopyran-2-one"),
    ("O=C1Oc2ccccc2C=C1c1ccc2ccccc2c1", "3-(naphthalen-2-yl)-2H-1-benzopyran-2-one"),
])
def test_senior_or_only_ring_carbonyl_core_keeps_pin(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
