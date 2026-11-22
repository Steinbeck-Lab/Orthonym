"""A labelled molecule named on an unlabelled name that is below the PIN is below the PIN too.

 (the Blue Book, "The name of an isotopically substituted compound is formed by adding or inserting the
nuclide symbol(s) enclosed in parentheses... before the part of the compound that is isotopically substituted"): the
name of the labelled compound is the name of the unlabelled one with the descriptor. A flavanone beside a flavone is not
the PIN parent,:21033, more multiple bonds), and its label-only record does not survive the descriptor
('...-2,3-dihydro(3,3-2H2)-4H-1-benzopyran-4-one'); rules.isotope_memos.carry_skeleton_labels records the labelled
name from the skeleton names the decorator built.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("O=C1CC([2H])(c2ccc(-c3cc(=O)c4ccccc4o3)cc2)Oc2ccccc21",
     "2-[4-(4-oxo-4H-1-benzopyran-2-yl)phenyl]-2,3-dihydro(2-2H)-4H-1-benzopyran-4-one"),
    ("O=C1CC(c2ccc(-c3cc(=O)c4ccccc4o3)cc2)Oc2cc([13CH3])ccc21",
     "7-(13C)methyl-2-[4-(4-oxo-4H-1-benzopyran-2-yl)phenyl]-2,3-dihydro-4H-1-benzopyran-4-one"),
    ("O=C1C([2H])([2H])C(Cc2cc(=O)c3ccccc3o2)Oc2ccccc21",
     "2-[(4-oxo-4H-1-benzopyran-2-yl)methyl]-2,3-dihydro(3,3-2H2)-4H-1-benzopyran-4-one"),
])
def test_labelled_name_on_a_non_pin_skeleton_is_not_pin(smiles, name):
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # the flavone and the flavanone alone are the PIN parents: the labelled names keep the PIN
    ("O=c1cc(-c2ccccc2)oc2cc([13CH3])ccc12", "7-(13C)methyl-2-phenyl-4H-1-benzopyran-4-one"),
    ("O=C1CC([2H])(c2ccccc2)Oc2ccccc21", "2-phenyl-2,3-dihydro(2-2H)-4H-1-benzopyran-4-one"),
])
def test_labelled_name_on_a_pin_skeleton_keeps_pin(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_labelled_name_on_an_uncertified_skeleton_is_not_pin():
    # the unlabelled phosphatidylglycerol has no PIN (the PIN tier abstains); its labelled form must not get one
    smiles = ("[2H]C([2H])(OC(=O)CCCCCCCCCCCCCCCC)[C@@]([2H])(OC(=O)CCCCCCC/C=C\\CCCCCCCC)C([2H])([2H])OP(=O)(O)OCC(O)CO")
    assert tier_row(smiles, "pin")["tier"] == "abstain"

