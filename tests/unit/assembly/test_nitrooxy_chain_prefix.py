"""A nitric ester group on a chain substituent is the prefix '(nitrooxy)' at the PIN tier.

 (the Blue Book): "-O-NO2 nitrooxy (preselected prefix)"; (:25963):
'2-(tert-butylimino)-3-methyl-3-(nitrooxy)butanoic acid (PIN)'. A carbamate is an ester of carbamic acid, a carbon
acid senior to nitric acid, Table 4.1), so the nitrate is the prefix.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("CC(C)(C)OC(=O)NCCO[N+](=O)[O-]", "tert-butyl [2-(nitrooxy)ethyl]carbamate"),
    ("CC(C)(C)OC(=O)NCCCO[N+](=O)[O-]", "tert-butyl [3-(nitrooxy)propyl]carbamate"),
    ("CC(C)(C)OC(=O)NCC(O[N+](=O)[O-])CO[N+](=O)[O-]", "tert-butyl [2,3-bis(nitrooxy)propyl]carbamate"),
])
def test_a_nitrooxy_chain_on_a_carbamate_is_the_pin(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    assert name_is_rt_exact(name, smiles)


def test_a_nitrate_principal_ester_keeps_its_name():
    r = name_best_effort("OCCO[N+](=O)[O-]")
    assert r["name"] == "2-hydroxyethyl nitrate" and r["tier"] == "pin_verified", r


@pytest.mark.parametrize("smiles,name", [
    # method (2) of "When anions are different, two methods are used" (the Blue Book): the polyol
    # with two different anions is no PIN this way ('methylene acetate formate (PIN)', not '(formyloxy)methyl acetate')
    ("NC(=O)OCCO[N+](=O)[O-]", "2-(nitrooxy)ethyl carbamate"),
    ("CS(=O)(=O)OCCO[N+](=O)[O-]", "2-(nitrooxy)ethyl methanesulfonate"),
])
def test_a_polyol_with_two_different_anions_is_below_the_pin(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified", r


@pytest.mark.parametrize("smiles,name", [
    # a urea is an amide class 11), junior to the ester of nitric or nitrous acid (class 9)
    ("NC(=O)NCCO[N+](=O)[O-]", "2-(carbamoylamino)ethyl nitrate"),
    ("CNC(=O)NCCO[N+](=O)[O-]", "2-[(methylcarbamoyl)amino]ethyl nitrate"),
    ("NC(=O)NCCON=O", "2-(carbamoylamino)ethyl nitrite"),
])
def test_a_nitrate_or_nitrite_ester_is_senior_to_a_urea(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
