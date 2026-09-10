""" cluster C1: seleno/telluro-ether amino acids must bail to the
polyfunctional pipeline (not the carbon-count amino-acid namer).

Root cause (blueprint C1): the amino-acid systematic namer counts backbone
carbons; for a -Se-/-Te- ether it walked through the chalcogen and DROPPED it
(C[Se]CC(N)C(=O)O -> '2-aminobutanoic acid', a different molecule). BB:
the chalcogen ether is a (methylselanyl)/(methyltellanyl) substituent prefix. Fix
adds the Se/Te ether SMARTS to the amino-acid bail-out list (mirroring the
thioether entry) so the polyfunctional pipeline names it correctly.
"""
import pytest

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer_pin():
    return Orthonym(style="pin")


@pytest.mark.parametrize("smiles,expected", [
    ("C[Se]CC(N)C(=O)O", "2-amino-3-(methylselanyl)propanoic acid"),  # coverage win
    ("C[Se]CCC(=O)O", "3-(methylselanyl)propanoic acid"),             # regression guard
    ("CSCCC(=O)O", "3-(methylsulfanyl)propanoic acid"),               # thioether unchanged
])
def test_chalcogen_ether_amino_acid(namer_pin, smiles, expected):
    assert namer_pin.name(smiles) == expected
