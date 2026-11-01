"""A chain of four or more heterounits that hangs from a ring has an 'a' name, so a substitutive name is no PIN.

 (the Blue Book, section "Skeletal replacement ('a') nomenclature in acyclic chains"):
"Skeletal replacement ('a') nomenclature rather than substitutive or multiplicative names must be used to generate
preferred IUPAC names for acyclic structures when four or more heterounits are present in a unbranched chain
containing at least one carbon atom". (:23430): the chain ends in carbon, so the ether oxygen bonded to
the ring is outside it and a PEG chain of three oxygens beyond it is not an 'a' chain. (:23405), the
N-substituent example: "since four heteroatoms are also present in the N-substituent group, it must also be named
by skeletal replacement ('a') nomenclature", against "since only three heteroatoms are present in the N-substituent
group, it must be named substitutively". The 'a' prefix of such a group ('2,5,8,11-tetraoxatridecan-13-yl') is not
built; the substitutive name of the wider tiers is below the PIN (spelling check ``checks_skeletal_chain``).
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_skeletal_chain import (
    a_substituent_chain_applies, a_substituent_chain_name_check)
from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

# four oxygens beyond the atom bonded to the ring
FOUR = [
    "COCCOCCOCCOCCOc1ccc(C(=O)O)cc1",
    "COCCOCCOCCOCCOC1CCCO1",
    "COCCOCCOCCOCCn1ccnc1",              # attached by a carbon
    "COCCOCCOCCOCCCc1ccc(C(=O)O)cc1",    # three oxygens beyond a propyl: C-C-C-O... four heterounits
]
# three, or a shape the reading does not know
NOT_FOUR = [
    "COCCOCCOCCOc1ccc(C(=O)O)cc1",
    "ICCOCCOCCOCCOC1CCCCC1",
    "c1ccccc1OCCOCCOCCOCCOc1ccccc1",     # a chain between two rings
    "COCCOCCOCCOCCOCC(=O)N",             # no ring
    "OCCOCCOCCOCCOc1ccc(C(=O)O)cc1",     # the free end is an OH
    "CCCCCCCCOc1ccc(C(=O)O)cc1",
]


@pytest.mark.parametrize("smiles", FOUR)
def test_the_reading_applies_to_four_heterounits_beyond_the_ring(smiles):
    assert a_substituent_chain_applies(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("smiles", NOT_FOUR)
def test_the_reading_does_not_apply_to_three_or_to_an_unread_shape(smiles):
    assert not a_substituent_chain_applies(Chem.MolFromSmiles(smiles))


def test_the_check_fails_a_substitutive_name_and_passes_an_a_name():
    mol = Chem.MolFromSmiles("COCCOCCOCCOCCOc1ccc(C(=O)O)cc1")
    assert a_substituent_chain_name_check(
        mol, "4-(2-{2-[2-(2-methoxyethoxy)ethoxy]ethoxy}ethoxy)benzoic acid") is not None
    assert a_substituent_chain_name_check(
        mol, "4-[(2,5,8,11-tetraoxatridecan-13-yl)oxy]benzoic acid") is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", FOUR[:3])
def test_the_substitutive_name_is_below_the_pin_and_still_given(smiles):
    assert default_tier_row(smiles)["is_pin"] is False
    be = name_best_effort(smiles)
    assert be["tier"] == "systematic_verified" and be["is_pin"] is False, be
    assert_full_rt(be["name"], smiles, what="best-effort: ")


@pytest.mark.opsin_gate
def test_three_heterounits_keep_the_pin_label():
    smiles = "COCCOCCOCCOc1ccc(C(=O)O)cc1"
    row = default_tier_row(smiles)
    assert row["name"] == "4-{2-[2-(2-methoxyethoxy)ethoxy]ethoxy}benzoic acid" and row["tier"] == "pin_verified", row


def test_a_chain_ending_in_a_hydroxy_group_needs_its_a_name():
    """ (the Blue Book) four or more heterounits take the 'a' name; a terminal -OH is the suffix on the
    chain's carbon end, not a unit: '2-[2-(2-{2-[2-(cyclohexyloxy)ethoxy]ethoxy}ethoxy)ethoxy]ethan-1-ol' is no PIN
    (the PIN is '14-(cyclohexyloxy)-3,6,9,12-tetraoxatetradecan-1-ol')."""
    from rdkit import Chem
    from orthonym.validation.spelling.checks_skeletal_chain import a_substituent_chain_applies
    assert a_substituent_chain_applies(Chem.MolFromSmiles("OCCOCCOCCOCCOCCOC1CCCCC1"))
    assert not a_substituent_chain_applies(Chem.MolFromSmiles("OCCOCCOC1CCCCC1"))
