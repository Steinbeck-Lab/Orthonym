"""v26 BP-5: the amide acyl parent chain must not absorb ring atoms.

Root cause (*.md): the private acyl
walker in rules/amides.py (_find_longest_carbon_chain / get_amide_chain_length)
had no ring awareness, so it descended from the carbonyl into an attached ring and
counted ring carbons as chain carbons (O=C(Cc1ccncc1)NCc1ccccn1 -> '...pentanamide').
BB P-44.3 / P-44.1.2.2(1): the acyclic principal chain never absorbs ring atoms
(heptylbenzene, not a C13 chain). Fix excludes ring atoms from the walk.
"""
import pytest

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer_pin():
    return Orthonym(style="pin")


@pytest.mark.parametrize("smiles,expected", [
    ("O=C(Cc1ccncc1)NCc1ccccn1",
     "2-(pyridin-4-yl)-N-[(pyridin-2-yl)methyl]acetamide"),
    ("NC(=O)Cc1ccccc1", "2-phenylacetamide"),
    ("O=C(N)CCc1ccccc1", "3-phenylpropanamide"),
    ("O=C(N)Cc1ccc(Cl)cc1", "2-(4-chlorophenyl)acetamide"),
])
def test_amide_does_not_absorb_ring(namer_pin, smiles, expected):
    assert namer_pin.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)NCc1ccccc1", "N-benzylacetamide"),      # ring on N-side (already excluded)
    ("CCCC(N)=O", "butanamide"),                      # no rings
    ("NC(=O)c1ccccc1", "benzamide"),                  # ring-attached -> -carboxamide path
    ("NC(=O)C1CCCCC1", "cyclohexanecarboxamide"),     # ring-attached path
])
def test_amide_no_regression(namer_pin, smiles, expected):
    assert namer_pin.name(smiles) == expected
