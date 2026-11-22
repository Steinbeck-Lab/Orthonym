"""A branch rooted at a hydrazine nitrogen is named on 'hydrazinyl' at the wider tiers too.

 (the Blue Book): '2-[(propan-2-ylidene)hydrazinyl]benzoic acid (PIN)'. The functional-group namer of
the wider tiers wrote the branch as amino on amino ('2-{[(propan-2-ylidene)amino]amino}benzoic acid', labelled
pin_unverified; '4-(aminoamino)benzoic acid'). assembly.n_branch_prefixes._hydrazine_root_prefix names it by the PIN
tier's route; where that route gives no name, the amino form stands below the PIN.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("CC(C)=NNc1ccccc1C(=O)O", "2-[(propan-2-ylidene)hydrazinyl]benzoic acid"),
    ("NNc1ccc(C(=O)O)cc1", "4-hydrazinylbenzoic acid"),
])
def test_hydrazine_branch_is_hydrazinyl_at_best_effort(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [
    "O=C(O)c1ccc(NN=Cc2ccccc2)cc1",
    "CC(C)=NN(C)c1ccc(C(=O)O)cc1",
])
def test_amino_on_amino_is_below_the_pin(smiles):
    row = tier_row(smiles, "best-effort")
    assert "amino" in row["name"] and "hydrazin" not in row["name"] and row["tier"] == "systematic_verified", row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("O=C(O)c1ccc(NN2CCCCC2)cc1", "4-[(piperidin-1-yl)amino]benzoic acid"),
    ("O=C(O)c1ccc(N=Nc2ccccc2)cc1", "4-(phenyldiazenyl)benzoic acid"),
])
def test_ring_nitrogen_and_azo_keep_their_names(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
