"""The names of the exact-match natural-product list are labelled systematic_verified, not pin_verified.

 INTRODUCTION (the Blue Book), the sentence at:50943: "Preferred IUPAC names (PINs) are not identified
for the compounds in this Chapter." The list names are the semisystematic parents of ("Semisystematic
names of recommended parent structures are listed in Table 10.1",:51375), so they are no PINs. The user delegated
the decision to the Blue Book (2026-10-17) and the lead decided: the label is systematic_verified (verified
'identity': the exact structure match, since OPSIN reads none of these names), which supersedes the decision of
2026-09-30 (c), "paper-run labels", for this class only. The name and its emission at every tier are unchanged:
the default tier emits an exact-match list name as one of its exceptions whatever its label.

The checks are independent of the label code: the expected name is the entry of the list table for the canonical
SMILES of the input (``data.natural_products.NATURAL_PRODUCT_DERIVATIVES``), OPSIN is asked for the name and
reads none of them, and a name of a steroid that is no list entry keeps pin_verified (control).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.data.natural_products import NAME_EXACT_NP_PARENTS, NATURAL_PRODUCT_DERIVATIVES
from tests.support.rt_assert import _independent_parse

pytestmark = pytest.mark.opsin_gate

LIST_ROWS = [
    "CC[C@H]1C[C@@H]2C[C@@H]3[C@H]1N(C2)CCC4=C3NC5=CC=CC=C45",                       # ibogamine
    "CC(C)[C@H]1CC[C@]2([C@H]1CC[C@@]3([C@@H]2CC[C@H]4[C@]3(CC[C@@H]5[C@@]4(CCCC5(C)C)C)C)C)C",   # hopane
    "CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1",                                           # germacrane
    "c1ccc2c(c1)CC[C@@]13CCCC[C@@]21CCN3",                                              # hasubanan
    "C=C1C[C@]23C[C@H]4[C@@H]5[C@@]6(C)CCC[C@]57C(C2C[C@H]1C[C@H]37)N4C6",             # hetisan
]
CHOLESTEROL = "C[C@H](CCCC(C)C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"


def _expected(smiles):
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    return NATURAL_PRODUCT_DERIVATIVES[canonical]


@pytest.mark.parametrize("smiles", LIST_ROWS)
def test_the_name_is_the_list_entry_of_the_exact_structure(smiles):
    name = _expected(smiles)
    assert name in NAME_EXACT_NP_PARENTS
    assert _independent_parse(name) is None        # OPSIN reads no list name: the label cannot be an OPSIN verdict


@pytest.mark.parametrize("smiles", LIST_ROWS)
def test_the_default_tier_emits_the_list_name_labelled_below_the_pin(smiles):
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == _expected(smiles), row
    assert row["gate_outcome"] == "carveout:np_stereoparent", row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert row["verified"] == "identity" and row["source"] == "trivial_retained", row


def test_best_effort_keeps_the_list_name_as_the_last_resort_labelled_below_the_pin():
    smiles = "CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1"
    row = Orthonym(**_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert row["name"] == "germacrane" and row["gate_outcome"] == "carveout:np_stereoparent", row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False and row["verified"] == "identity", row


def test_a_steroid_name_that_is_no_list_entry_keeps_pin_verified():
    row = Orthonym().name_tiered(CHOLESTEROL)
    assert row["name"] == "cholesterol" and row["tier"] == "pin_verified" and row["is_pin"] is True, row
