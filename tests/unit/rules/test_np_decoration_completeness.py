"""A natural-product scaffold name never drops a decoration (pre-existing-failures
plan, Task 4: TRIAGE row 93 and canary call 182).

`rules/natural_products.name_natural_product_with_substituents` names a scaffold
plus the decorations its finders recognise. Two paths dropped the rest:

- its no-silent-drop completeness invariant ran only when a CONJUGATE fired, so a
  decoration no finder recognises vanished: the 4-carboxy group of
  '(3S,4S,5R,8S,10S,11R,13R,14S,17R,20R)-3-hydroxy-4-methyl-7-oxoergosta-9,24-
  dien-11-yl acetate' (row 93; OPSIN-unparseable, and not the molecule);
- its "no decorations -> bare scaffold name" test forgot `esters`, so a scaffold
  whose only decoration is an ester came out as the bare scaffold: 'tropane' for
  tropisetron (C8H15N for C17H20N2O2, canary call 182). The carbon-count acylate
  fallback also named tropisetron's indole-3-carboxylate 'nonanoate'.

 "SUBSTITUTIVE NOMENCLATURE" (the Blue Book): every substituent is
cited as a prefix or suffix. The producer now declines (None -> the systematic
pipeline) instead.
"""
import pytest
from rdkit import Chem

from orthonym.rules.natural_products import name_natural_product

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles", [
    # row 93
    "C=C(CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3C(=O)C[C@H]4[C@](C)(C(=O)O)[C@@H](O)"
    "CC[C@]4(C)C3=C[C@@H](OC(C)=O)[C@]12C)C(C)C",
    # canary call 182 (free base; the salt names the same component)
    "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2",
])
def test_np_scaffold_declines_an_unnameable_decoration(smiles):
    assert name_natural_product(Chem.MolFromSmiles(smiles)) is None


@pytest.mark.parametrize("smiles,expected", [
    # controls: every decoration recognised -> the NP names are unchanged
    ("CC(=O)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](O)CC[C@@H]32)C1",
     "17β-hydroxy-5β-androstan-3β-yl acetate"),
    ("C[C@]12CC[C@H]3[C@@H](CC=C4C[C@@H](O)CC[C@]34C)[C@@H]1CC[C@@H]2O",
     "androst-5-ene-3β,17β-diol"),
])
def test_np_scaffold_with_recognised_decorations_unchanged(smiles, expected):
    assert name_natural_product(Chem.MolFromSmiles(smiles)) == expected
