"""A label on one of two groups cited by one multiplied suffix is not the PIN spelling.

 (heading 'Isotopically modified compounds', the Blue Book): "the isotopically modified
characteristic group with the greater number of modifications is chosen as the principal characteristic group to be
cited as a suffix; the other characteristic is then cited as a prefix" ('1-carboxycyclohexane-1-(13C,2H)carboxylic
acid (PIN)'). '(15N)benzene-1,4-dicarboxamide' reads back to the drawn structure but cites the labelled group in the
multiplied suffix: it ships below the PIN (rules.isotope_suffix_split.mark_label_on_one_of_multiplied_groups).
The same holds for chain suffixes: '(2R)-2-(2H)hydroxy-3-hydroxy(1-2H)propanal (PIN)' (the Blue Book) cites the
two hydroxy groups apart once one carries a label.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles", [
    "[15NH2]C(=O)c1ccc(C(N)=O)cc1", "[15NH2]C(=O)C1(C(N)=O)CCCCC1",
    "NC(=O)CCC(=O)[15NH2]",                     # (15N)butanediamide
    "NS(=O)(=O)c1ccc(S(=O)(=O)[15NH2])cc1",     # (15N)benzene-1,4-disulfonamide
    "OC(=O)CCC(=O)[18OH]",                      # (O-18O)butanedioic acid
    "NCCCC[15NH2]",                             # (15N)butane-1,4-diamine
    "N#CCCC#[15N]",                             # (15N)butanedinitrile
    "O=CCCC=[18O]",                             # (18O)butanedial
    "CC(=[18O])CC(C)=O",                        # (18O)pentane-2,4-dione
])
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
    # a label on a skeleton atom, not on a suffix group
    ("CC(=O)CC(=O)[13CH3]", "(1-13C)pentane-2,4-dione"),
    ("O=C1CCC(=O)C[13CH2]1", "(2-13C)cyclohexane-1,4-dione"),
    ("NC(=O)[13CH2]CC(N)=O", "(2-13C)butanediamide"),
    ("N#C[13CH2]CC#N", "(2-13C)butanedinitrile"),
    ("OC[13CH2]CO", "(2-13C)propane-1,3-diol"),
    ("[15NH2]CC[15NH2]", "(15N2)ethane-1,2-diamine"),
])
def test_alike_groups_keep_the_pin(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
