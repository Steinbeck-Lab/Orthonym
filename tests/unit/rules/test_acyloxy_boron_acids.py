"""Anhydrides of boric acid with carboxylic acids (``rules/acyloxy_boron_acids.py``).

, the acid is senior to the anhydride: '(propanoyloxy)boronic acid (PIN)' (the Blue Book, not 'boric
propanoic monoanhydride'), 'bis(acetyloxy)borinic acid (PIN; acid is senior to anhydride)' (:32382).
"Anhydrides of mononuclear noncarbon oxoacids" (:35982): 'acetic boric trianhydride (PIN)' (:35994).
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("CCC(=O)OB(O)O", "(propanoyloxy)boronic acid"),                    #:37003
    ("CC(=O)OB(O)O", "(acetyloxy)boronic acid"),
    ("CC(=O)OB(O)OC(C)=O", "bis(acetyloxy)borinic acid"),               #:32382
    ("CC(=O)OB(O)OC(=O)CC", "(acetyloxy)(propanoyloxy)borinic acid"),
    ("O=C(OB(O)O)c1ccc(O)cc1", "[(4-hydroxybenzoyl)oxy]boronic acid"),
    ("CC(=O)OB(OC(C)=O)OC(C)=O", "acetic boric trianhydride"),          #:35994
    ("CC(=O)OB(C)C", "acetic dimethylborinic anhydride"),               #:35992, the monobasic acid, unchanged
])
def test_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles", [
    # takes "the senior acid as parent" (:35990): a sulfo, phosphono or carboxy group outranks the boron acid
    "OS(=O)(=O)CC(=O)OB(O)O",
    "OS(=O)(=O)c1ccc(cc1)C(=O)OB(O)O",
    "OP(=O)(O)CC(=O)OB(O)O",
    "OC(=O)CCC(=O)OB(O)O",
    # a carbamic or carbonic acyl group is no carboxylic one ('[(carbamoyl)oxy]' would be a misspelt prefix)
    "NC(=O)OB(O)O",
    "COC(=O)OB(O)O",
])
def test_no_pin(smiles):
    assert tier_row(smiles, "pin")["tier"] != "pin_verified"


def test_the_acyloxy_prefixes_are_in_alphanumerical_order():
    row = tier_row("CC(=O)OB(O)OC(=O)c1ccccc1Cl", "pin")
    assert row["name"] == "(acetyloxy)[(2-chlorobenzoyl)oxy]borinic acid", row


def test_a_substituted_acid_on_the_neutral_trianhydride_keeps_its_ester_name():
    # no name of this form in the book: the PIN tier declines, best-effort keeps the ester name
    smiles = "ClCC(=O)OB(OC(=O)CCl)OC(=O)CCl"
    assert tier_row(smiles, "pin")["tier"] != "pin_verified"
    assert tier_row(smiles, "best-effort")["name"] == "boranetriyl tris(chloroacetate)"
