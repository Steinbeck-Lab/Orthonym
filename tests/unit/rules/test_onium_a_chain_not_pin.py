"""A cation whose chain holds four heterounits with an onium centre among them has an 'a' name as its PIN.

### **** SKELETAL REPLACEMENT ('a') NOMENCLATURE FOR CATIONS (the Blue Book), method (1): neutral 'a'
prefixes and the suffix 'ium': '3,12-dimethyl-6,9-dioxa-3,12-dithiatetradecane-3,12-diium (PIN)' (:41785). Orthonym
builds no 'ium' 'a' name; the substitutive name ('1,2-bis{2-[ethyl(methyl)sulfaniumyl]ethoxy}ethane') was labelled
pin_verified and is now below the PIN (rules.skeletal_chain.note_onium_a_chain). An aminium nitrogen is the principal
characteristic group, not a skeletal atom (:41775), and a sulfonium without four heterounits keeps its PIN.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles", [
    "CC[S+](C)CCOCCOCC[S+](C)CC",
    "C[S+](C)CCOCCOCCSCC",
    "CC[S+](C)CCOCCOCC[S+](C)CC.[I-].[I-]",
    "CC[P+](C)(C)CCOCCOCCOCC",
    "[13CH3][S+](C)CCOCCOCC[S+](C)CC",   # the labelled cation, through the skeleton carry
])
def test_onium_a_chain_is_below_the_pin(smiles):
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, "best-effort")
    assert row["tier"] == "systematic_verified", row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("CC[S+](C)CCOCCOCC", "[2-(2-ethoxyethoxy)ethyl](ethyl)(methyl)sulfanium"),
    ("C[S+](C)C", "trimethylsulfanium"),
    ("CCSCCOCCOCCSCC", "6,9-dioxa-3,12-dithiatetradecane"),
])
def test_without_an_onium_a_chain_the_pin_stays(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


def test_a_chain_amine_is_no_heterounit():
    # the amine nitrogen of the neutral compound is its principal group (:41775): S, O, O are three heterounits
    from rdkit import Chem
    from orthonym.rules.skeletal_chain import note_onium_a_chain
    assert not note_onium_a_chain(Chem.MolFromSmiles("C[S+](C)CCNCCOCCOC"))
    assert note_onium_a_chain(Chem.MolFromSmiles("C[S+](C)CCSCCOCCOC"))
