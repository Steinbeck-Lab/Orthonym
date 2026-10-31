"""The parent's stereodescriptors stand at the front of the complete name, before an N-prefix.

 NAMING OF STEREOISOMERS (the Blue Book,:44643): "They are placed at the front of the complete name when
related to the parent structure". The imine builder put the block after the prefixes
('N-methyl-1-(pyridin-4-yl)(1Z,2E)-but-2-en-1-imine') and the name was labelled a PIN; so did the ceramide
builder ('N-[...](9Z)-octadec-9-enamide') and the N-acyl join of the decomposition engine.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C/C=C/C(c1ccncc1)=N/C", "(1Z,2E)-N-methyl-1-(pyridin-4-yl)but-2-en-1-imine"),
    ("C/C=C/C=N/C", "(1E,2E)-N-methylbut-2-en-1-imine"),
    ("C/C=C/C=NC1CCCCC1", "(2E)-N-cyclohexylbut-2-en-1-imine"),
    # no stereo: unchanged
    ("CC(=NC)c1ccccc1", "N-methyl-1-phenylethan-1-imine"),
])
def test_the_parent_block_opens_the_name(smiles, expected):
    assert Orthonym().name(smiles) == expected


@pytest.mark.opsin_gate
def test_a_ceramide_cites_the_amide_block_first():
    # CHEBI:77996
    assert (Orthonym().name("CCCCCCCC/C=C\\CCCCCCCC(=O)N[C@@H](CO)[C@H](O)/C=C/CCCCCCCCCCCCC")
            == "(9Z)-N-[(2S,3R,4E)-1,3-dihydroxyoctadec-4-en-2-yl]octadec-9-enamide")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ('COCC(=O)N(c1c(C)cccc1C)[C@@H](C)C(=O)O', "(2S)-N-(methoxyacetyl)-2-(2,6-dimethylanilino)propanoic acid"),
    ('CC(=O)N[C@@H](CO)[C@@H](O[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O)[C@@H](O)[C@H](O)CO',
     "(2R,3S,4R,5S)-N-acetyl-5-amino-4-(\u03b2-D-galactopyranosyloxy)hexane-1,2,3,6-tetrol"),
])
def test_an_n_acyl_join_keeps_the_amine_block_first(smiles, expected):
    # CHEBI:183423, CHEBI:152323 (decomposition engine, below the PIN)
    assert Orthonym(style="pin", **_emit_tier_flags("best-effort")).name(smiles) == expected
