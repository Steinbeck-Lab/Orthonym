"""A compound amido prefix or acylate word on a monosaccharide name is enclosed.

 "Parentheses" (the Blue Book), (:7232): "Parentheses are used around compound... and complex
... prefixes"; a compound anion word of a sugar ester is enclosed too ('D-glucopyranose 6-(dihydrogen phosphate)',
:53209). (:3477): a compound prefix is alphabetized by its complete name ('deoxy' before
'(2-hydroxyacetamido)'). The sugar producers (data/sugar_names.py) cited both bare; a single compound acylate word
now declines the functional-class sugar ester, so the systematic ester name (the PIN) is built.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("O=C(CO)N[C@H]1[C@@H](O)[C@H](O)[C@@H](COP(=O)(O)O)O[C@H]1O",
     "2-deoxy-2-(2-hydroxyacetamido)-6-O-phosphono-β-D-mannopyranose"),
    ("O=C(Cc1ccc(O)cc1)O[C@@H]1[C@@H](OC(=O)Cc2ccc(O)cc2)[C@H](O)O[C@H](CO)[C@H]1OC(=O)Cc1ccc(O)cc1",
     "β-D-glucopyranose 2,3,4-tris[(4-hydroxyphenyl)acetate]"),
    # simple words stay bare
    ("CC(=O)N[C@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1O", "2-acetamido-2-deoxy-β-D-mannopyranose"),
    ("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1OC(=O)c1ccccc1", "β-D-glucopyranose 4-benzoate"),
])
def test_sugar_compound_prefix_marks(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # one compound acylate word: the systematic ester name is the PIN (the functional-class sugar name declines)
    ("CC(O)c1cccc2nc3c(C(=O)O[C@H]4[C@H](O)[C@@H](O)[C@H](C)O[C@@H]4O)cccc3nc12",
     "(2S,3S,4R,5R,6S)-2,4,5-trihydroxy-6-methyloxan-3-yl 6-(1-hydroxyethyl)phenazine-1-carboxylate"),
    ("O=C(OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O)c1cc(O)c(O)c(O)c1",
     "[(2R,3S,4S,5R,6R)-3,4,5,6-tetrahydroxyoxan-2-yl]methyl 3,4,5-trihydroxybenzoate"),
])
def test_one_compound_acylate_word_keeps_the_systematic_pin(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
