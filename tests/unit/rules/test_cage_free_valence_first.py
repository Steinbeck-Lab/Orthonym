"""The free valence of an unsaturated von Baeyer substituent takes its low locant before the 'ene' ending.

 (free valences) comes before ('ene' and 'yne' endings); (the Blue Book):
'2-[bicyclo[6.6.1]pentadeca-8(15)-en-1-yl]ethan-1-ol (PIN)'. The 'a' of that line is the book's exception: 'hexadec-
1(15)-ene' (:16495) and 'oct-1(8)-ene' (:16643) elide it. rules.cage_free_valence.free_valence_first_enes.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("OCCC12C=C(CCCCCC1)CCCCCC2", "2-[bicyclo[6.6.1]pentadec-8(15)-en-1-yl]ethan-1-ol"),
    ("OCCC12C=C(CCC1)CCC2", "2-[bicyclo[3.3.1]non-5(9)-en-1-yl]ethan-1-ol"),
])
def test_free_valence_before_ene(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("OCCC12CCC(CC1)C=C2", "2-(bicyclo[2.2.2]oct-2-en-1-yl)ethan-1-ol"),
    ("OCCC1CC2C=CC1C2", "2-(bicyclo[2.2.1]hept-5-en-2-yl)ethan-1-ol"),
])
def test_names_already_right_stay(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
