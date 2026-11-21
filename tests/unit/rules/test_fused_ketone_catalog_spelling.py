"""The fused-core catalog spells acridone, xanthone and thioxanthone systematically, also with substituents.

 (the Blue Book-28334) lists the retained ketone names, and acridone, xanthone and thioxanthone are not
among them rejects the like-built 'quinolone' and 'isoquinolone'), so the PIN is the ketone of the ring
parent: 'acridin-9(10H)-one', '9H-xanthen-9-one', '9H-thioxanthen-9-one'. The substituted names come from
the fused-core match (rules.fused_rings.name_fused_heterocycle), which used to splice the trivial word.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name,tier", [
    ("O=C1Oc2ccccc2C=C1Cc1ccc2[nH]c3ccccc3c(=O)c2c1",
     "2-[(2-oxo-2H-1-benzopyran-3-yl)methyl]acridin-9(10H)-one", "pin_verified"),
    ("O=C1Oc2ccccc2C=C1Cc1ccc2oc3ccccc3c(=O)c2c1",
     "2-[(2-oxo-2H-1-benzopyran-3-yl)methyl]-9H-xanthen-9-one", "pin_verified"),
    # the coumarin is senior here (c), O before S,:19414): below the PIN
    ("O=C1Oc2ccccc2C=C1Cc1ccc2sc3ccccc3c(=O)c2c1",
     "2-[(2-oxo-2H-1-benzopyran-3-yl)methyl]-9H-thioxanthen-9-one", "systematic_verified"),
])
def test_fused_ketone_core_systematic_spelling(smiles, name, tier):
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, tier), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # (g) (the Blue Book): with one locant set, the prefix cited first takes the lower locant;
    # (:3477) alphabetizes the complete prefix name ('oxobutyl' after 'ethyl' and 'methyl')
    ("CCc1cccc2oc3cccc(CCC(C)=O)c3c(=O)c12", "1-ethyl-8-(3-oxobutyl)-9H-xanthen-9-one"),
    ("Cc1cccc2oc3cccc(CCC(C)=O)c3c(=O)c12", "1-methyl-8-(3-oxobutyl)-9H-xanthen-9-one"),
    ("CCc1cccc2[nH]c3cccc(CCC(C)=O)c3c(=O)c12", "1-ethyl-8-(3-oxobutyl)acridin-9(10H)-one"),
    ("CCc1cccc2sc3cccc(CCC(C)=O)c3c(=O)c12", "1-ethyl-8-(3-oxobutyl)-9H-thioxanthen-9-one"),
    ("CCc1cccc2oc3cccc(CC(C)=O)c3c(=O)c12", "1-ethyl-8-(2-oxopropyl)-9H-xanthen-9-one"),
])
def test_mirror_numbering_by_complete_prefix_name(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
