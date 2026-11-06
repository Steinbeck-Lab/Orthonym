"""The sultone and the sultam of naphthalene: heterones of a peri-fused catalogue parent (``rules/catalogue_heterones``).

 (the Blue Book, sultones and sultines; "Method (1) gives preferred IUPAC names"):
'2H-2λ6-naphtho[1,8-cd][1,2]oxathiole-2,2-dione (PIN)' (:32160); the sultams (:33248, "Method (1) leads to preferred
IUPAC names"): '1λ6-naphtho[1,8-cd][1,2]thiazole-1,1(2H)-dione (PIN)' (:33254). The parents are catalogue entries
with OPSIN-derived numbering, peri fusion across naphthalene 1,8).
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import assert_full_rt
from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.unit

BOOK = [
    ("O=S1(=O)Oc2cccc3cccc1c23", "2H-2λ6-naphtho[1,8-cd][1,2]oxathiole-2,2-dione"),
    ("O=S1(=O)Nc2cccc3cccc1c23", "1λ6-naphtho[1,8-cd][1,2]thiazole-1,1(2H)-dione"),
]
PARENTS = [
    ("c1cc2cccc3c2c(c1)OS3", "naphtho[1,8-cd][1,2]oxathiole"),
    ("c1cc2cccc3c2c(c1)NS3", "2H-naphtho[1,8-cd][1,2]thiazole"),
    ("c1cc2cccc3c2c(c1)SS3", "naphtho[1,8-cd][1,2]dithiole"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", BOOK + PARENTS, ids=lambda v: v)
def test_book_names(smiles, name):
    row = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row
    assert_full_rt(name, smiles)


@pytest.mark.opsin_gate
def test_one_oxo_elides_the_final_e():
    """The sultine: one =O, 'one' after the elided 'e' ('1H-1λ4-thiophen-1-one (PIN)', :29440)."""
    row = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered("O=S1Oc2cccc3cccc1c23")
    assert row["name"] == "2H-2λ4-naphtho[1,8-cd][1,2]oxathiol-2-one", row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", [
    ("O=S1OCc2ccccc21", "3H-2,1λ4-benzoxathiol-1-one"),
    ("O=S1(=O)OCc2ccccc21", "3H-2,1λ6-benzoxathiole-1,1-dione"),
    ("O=S1(=O)Cc2ccccc2O1", "3H-1,2λ6-benzoxathiole-2,2-dione"),
], ids=lambda v: v)
def test_a_parent_with_hydrogen_on_a_ring_carbon_is_named_below_the_pin(smiles, name):
    """The certified parent keeps its indicated hydrogen ('3H-2,1-benzoxathiole') and the lambda token joins its
    heteroatom locants; no book name of this kind is printed, so the name is valid, not the PIN."""
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] == "systematic_verified", row
    assert_full_rt(name, smiles)
