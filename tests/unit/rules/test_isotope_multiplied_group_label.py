"""A label on one of two groups cited by one multiplied suffix is not the PIN spelling.

 (heading 'Isotopically modified compounds', the Blue Book): "the isotopically modified
characteristic group with the greater number of modifications is chosen as the principal characteristic group to be
cited as a suffix; the other characteristic is then cited as a prefix" ('1-carboxycyclohexane-1-(13C,2H)carboxylic
acid (PIN)'). '(15N)benzene-1,4-dicarboxamide' reads back to the drawn structure but cites the labelled group in the
multiplied suffix: it ships below the PIN (rules.isotope_suffix_split.mark_label_on_one_of_multiplied_groups).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles", ["[15NH2]C(=O)c1ccc(C(N)=O)cc1", "[15NH2]C(=O)C1(C(N)=O)CCCCC1"])
def test_label_on_one_of_two_multiplied_groups_is_no_pin(smiles):
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, "best-effort")
    assert row["name"] and row["tier"] != "pin_verified", row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("[15NH2]C(=O)c1ccc(C([15NH2])=O)cc1", "(15N2)benzene-1,4-dicarboxamide"),   # both groups alike
    ("NC(=O)c1ccc([13cH]c1)C(N)=O", "(2-13C)benzene-1,4-dicarboxamide"),        # the label on the ring
    ("OC(=O)c1ccc(C(=O)O)cc1[2H]", "(2-2H)benzene-1,4-dicarboxylic acid"),
    ("[15NH2]C(=O)c1ccccc1", "(15N)benzamide"),
])
def test_alike_groups_keep_the_pin(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
