"""Ester groups as prefixes of a ring acid: the 'alkoxy...oxo' chain form and the arene writer's shared prefix.

 (the Blue Book) names an ester beside a senior group by 'alkoxy...oxo' or 'alkoxycarbonyl'; the book's
PINs use the chain form ('3-ethoxy-N,N,N-trimethyl-3-oxopropan-1-aminium bromide (PIN)', not '[2-(ethoxycarbonyl)ethyl]...',
:31709; '2-methoxy-2-oxoethyl',:31886) and name a thioanhydride link the same way ('acetic 1-[3-(acetylsulfanyl)-3-
oxopropyl]naphthalene-2-carboxylic anhydride (PIN)':32461, 'acetic 3-[(acetylsulfanyl)carbonyl]naphthalene-2-carboxylic
anhydride (PIN)':32459). These rows had no PIN-tier name (``assembly.alkoxy_oxo_branch``, ``rules.pah_shared_prefix``).
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)c1ccccc1CCC(=O)OC", "2-(3-methoxy-3-oxopropyl)benzoic acid"),
    ("OC(=O)c1ccccc1CC(=O)OCC", "2-(2-ethoxy-2-oxoethyl)benzoic acid"),
    ("OC(=O)c1ccncc1CC(C)C(=O)OC", "3-(3-methoxy-2-methyl-3-oxopropyl)pyridine-4-carboxylic acid"),
    ("OC(=O)c1cc2ccccc2cc1C(=O)OC", "3-(methoxycarbonyl)naphthalene-2-carboxylic acid"),
    ("OC(=O)c1cc2ccccc2cc1C(C)=O", "3-acetylnaphthalene-2-carboxylic acid"),
    ("OC(=O)c1ccc2ccccc2c1CCC(=O)OC", "1-(3-methoxy-3-oxopropyl)naphthalene-2-carboxylic acid"),
    ("CC(=O)OC(=O)c1ccc2ccccc2c1CCC(=O)SC(C)=O",
     "acetic 1-[3-(acetylsulfanyl)-3-oxopropyl]naphthalene-2-carboxylic anhydride"),
    ("CC(=O)OC(=O)c1cc2ccccc2cc1C(=O)SC(C)=O",
     "acetic 3-[(acetylsulfanyl)carbonyl]naphthalene-2-carboxylic anhydride"),
])
def test_pin(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_a_group_holding_the_principal_characteristic_group_is_no_prefix():
    # the acetic acid of the side chain is the suffix of another parent:18169): never '(carboxymethyl)' on the ring
    assert tier_row("Oc1cc2ccccc2cc1CC(=O)O", "pin")["name"] == "(3-hydroxynaphthalen-2-yl)acetic acid"
    assert "carboxymethyl" not in tier_row("OC(=O)c1cc2ccccc2cc1CC(=O)O", "pin")["name"]


@pytest.mark.parametrize("smiles", [
    "OC(=O)c1cc2ccccc2cc1C(=O)NN",        # the book's prefix is 'hydrazinecarbonyl':33990), not 'aminocarbamoyl'
    "OC(=O)c1cc2ccccc2cc1C(=O)CC(=O)OC",  # '3-methoxy-3-oxopropanoyl', not '3-methoxy-1,3-dioxopropyl':30430)
    "OC(=O)c1cc2ccccc2cc1C(=N)OC",        # 'C-methoxycarbonimidoyl', not '[imino(methoxy)methyl]'
])
def test_the_shared_prefix_claims_no_pin_where_it_spells_no_book_prefix(smiles):
    assert tier_row(smiles, "pin")["tier"] != "pin_verified"


@pytest.mark.parametrize("smiles,expected", [
    # a compound prefix is enclosed on an indicated-hydrogen parent too; '3-(methoxycarbonyl)phenyl',:31886)
    ("OC(=O)c1cc2CC=Cc2cc1C(=O)OC", "5-(methoxycarbonyl)-1H-indene-6-carboxylic acid"),
    ("OC(=O)c1cc2CCCc2cc1C(=O)OC", "6-(methoxycarbonyl)-2,3-dihydro-1H-indene-5-carboxylic acid"),
])
def test_indene_prefix_marks(smiles, expected):
    assert tier_row(smiles, "pin")["name"] == expected
