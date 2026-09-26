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
])
def test_np_scaffold_declines_an_unnameable_decoration(smiles):
    assert name_natural_product(Chem.MolFromSmiles(smiles)) is None


# fix a performance pass (wp6-tests), change-asserted-value: canary call 182 (tropisetron, free
# base) moved out of the 'declines' list. 28e1d520b (esters.py: name_ester names a fused
# ring acid through name_polyfunctional_ester_via_acid) lets the NP producer name its
# indole-3-carboxylate, so it now returns the complete name instead of None (archive
# bisect, t12-research rest-of-suite item 29). What the old assertion protected is
# (the Blue Book, every substituent is cited): the producer may decline, or return
# the complete, verified name -- never the bare 'tropane' or a 'nonanoate'.
def test_np_scaffold_names_tropisetron_completely_or_declines():
    from orthonym.namer import _pseudoasymmetric_name_verified
    smiles = "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2"
    np = name_natural_product(Chem.MolFromSmiles(smiles))
    assert np is None or (
        np == "(1R,3r,5S)-tropan-3-yl 1H-indole-3-carboxylate"
        and _pseudoasymmetric_name_verified(np, smiles)), np


@pytest.mark.parametrize("smiles,expected", [
    # controls: every decoration recognised -> the NP names are unchanged
    ("CC(=O)O[C@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](O)CC[C@@H]32)C1",
     "17β-hydroxy-5β-androstan-3β-yl acetate"),
    ("C[C@]12CC[C@H]3[C@@H](CC=C4C[C@@H](O)CC[C@]34C)[C@@H]1CC[C@@H]2O",
     "androst-5-ene-3β,17β-diol"),
])
def test_np_scaffold_with_recognised_decorations_unchanged(smiles, expected):
    assert name_natural_product(Chem.MolFromSmiles(smiles)) == expected
