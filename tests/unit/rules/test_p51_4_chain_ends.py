"""The chain of a skeletal replacement ('a') name ends in carbon, and an O, N or S that ends the longest path is a substituent.

 (the Blue Book, section "Skeletal replacement ('a') nomenclature in acyclic chains"):
"The chain must be terminated by C or one of the following heteroatoms: P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga,
In, or Tl." (:23348): the 'a' name is used "when four or more heterounits are present in a unbranched
chain containing at least one carbon atom". (:18917): the parent with the senior atom (O above C) is the
senior one "whether cyclic or acyclic", so the chain of oxa units is the parent and the ring a prefix. A chain whose
longest path ends on an ether oxygen bonded to a ring was named from that oxygen
('1-cyclohexyl-1,4,7,10,13-pentaoxatetradecane'), a chain terminated by O.
"""
import pytest

from orthonym import Orthonym
from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

# (SMILES, the PIN): the oxygens in the chain that ends in carbon at both ends, 3 and 6 and 9 and 12...
ROWS = [
    ("ICCOCCOCCOCCOCCOC1CCCCC1", "1-(cyclohexyloxy)-14-iodo-3,6,9,12-tetraoxatetradecane"),
    ("ICCOCCOCCOCCOCCOCCOC1CCCCC1", "1-(cyclohexyloxy)-17-iodo-3,6,9,12,15-pentaoxaheptadecane"),
    ("ICCOCCOCCOCCOCCOc1ccccc1", "1-iodo-14-phenoxy-3,6,9,12-tetraoxatetradecane"),
    ("CCOCCOCCOCCOCCOC1CCCCC1", "1-(cyclohexyloxy)-3,6,9,12-tetraoxatetradecane"),
    ("ClCCOCCOCCOCCOCCOC1CCCCC1", "1-chloro-14-(cyclohexyloxy)-3,6,9,12-tetraoxatetradecane"),
    ("CC(C)OCCOCCOCCOCCOC1CCCCC1", "1-(cyclohexyloxy)-13-methyl-3,6,9,12-tetraoxatetradecane"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_the_chain_ends_in_carbon_and_the_ring_oxygen_is_a_prefix(smiles, expected):
    row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert_full_rt(row["name"], smiles)


def test_three_oxygens_inside_the_chain_stay_a_substitutive_name():
    """Only three heterounits are inside the chain that ends in carbon needs four)."""
    smiles = "ICCOCCOCCOCCOC1CCCCC1"
    row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == "(2-{2-[2-(2-iodoethoxy)ethoxy]ethoxy}ethoxy)cyclohexane", row


def test_a_chain_with_carbon_at_both_ends_is_unchanged():
    row = Orthonym(style="pin").name_tiered("COCCOCCOCCOCCOC")
    assert row["name"] == "2,5,8,11,14-pentaoxapentadecane" and row["tier"] == "pin_verified", row


def test_no_oxa_block_starts_at_locant_one_in_the_class():
    """The old names '1-cyclohexyl-15-iodo-1,4,7,10,13-pentaoxapentadecane' began the chain with an oxygen."""
    for smiles, _expected in ROWS:
        name = Orthonym(style="pin").name_tiered(smiles)["name"]
        assert "1,4,7,10,13-penta" not in name and "1-cyclohexyl" not in name, name


def test_an_unsaturated_chain_is_below_the_pin_now_and_still_named_at_the_wider_tier():
    """The chain of 'oxa' units with a C=C is not built by this producer: the PIN tier gives no name (it gave the
    O-terminated chain '1-cyclohexyl-1,4,7,10,13-pentaoxahexadec-15-ene' before) and the wider tier names it."""
    smiles = "C=CCOCCOCCOCCOCCOC1CCCCC1"
    assert default_tier_row(smiles)["is_pin"] is False
    be = name_best_effort(smiles)
    assert be["tier"] != "pin_verified" and be["is_pin"] is False, be
    assert_full_rt(be["name"], smiles, what="best-effort: ")
