"""A bis(azo) compound without a suffix group has a 'diazene' PIN, which Orthonym does not build yet.

 (the Blue Book), method (1) generates PINs: '{7-[(anthracen-2-yl)diazenyl]naphthalen-2-yl}
(phenyl)diazene (PIN)' (:38829). The ring parent name is below the PIN (rules.bis_azo.note_bis_azo); a monoazo compound
keeps its diazene PIN and a bis(azo) compound with a suffix group keeps its ring parent PIN.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles", [
    "c1ccc(N=Nc2ccc(N=Nc3ccccc3)cc2)cc1",
    "c1ccc(N=Nc2ccc3ccc(N=Nc4ccc5cc6ccccc6cc5c4)cc3c2)cc1",
])
def test_bis_azo_without_suffix_group_is_below_the_pin(smiles):
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, "best-effort")
    assert row["tier"] == "systematic_verified", row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("c1ccc(N=Nc2ccc3ccccc3c2)cc1", "(naphthalen-2-yl)(phenyl)diazene"),
    ("Oc1ccc2ccccc2c1N=Nc1ccc(N=Nc2ccccc2)cc1", "1-{[4-(phenyldiazenyl)phenyl]diazenyl}naphthalen-2-ol"),
])
def test_monoazo_and_suffix_bearing_bis_azo_keep_the_pin(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("smiles", [
    "c1ccc(N=Nc2ccc(N=Nc3ccccn3)cc2)cc1",       # a ring nitrogen is no characteristic group
    "c1ccc(N=Nc2ccc(N=Nc3ccc[nH]3)cc2)cc1",
])
def test_bis_azo_with_a_ring_heteroatom_is_below_the_pin(smiles):
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    assert tier_row(smiles, "best-effort")["tier"] == "systematic_verified"
