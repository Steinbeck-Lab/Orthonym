"""A suffix on the ring nitrogen of an azine takes added hydrogen.

IUPAC 2013 (the Blue Book) and: 'quinoline-1(2H)-carboxylic acid (PIN)' (:24701,:29894),
'pyridin-1(2H)-yl (preferred prefix)' (:24703).
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.ring_nitrogen_suffix import name_azine_nitrogen_suffix

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("O=C(O)N1CC=Cc2ccccc21", "quinoline-1(2H)-carboxylic acid"),
    ("OC(=O)N1C=CC=CC1", "pyridine-1(2H)-carboxylic acid"),
    ("OC(=O)N1CC=CC=C1", "pyridine-1(2H)-carboxylic acid"),
    ("OC(=O)N1CCCC=C1", "3,4-dihydropyridine-1(2H)-carboxylic acid"),
    ("O=C(O)N1CCCc2ccccc21", "3,4-dihydroquinoline-1(2H)-carboxylic acid"),
    ("NC(=O)N1CCCc2ccccc21", "3,4-dihydroquinoline-1(2H)-carboxamide"),
    ("O=C(O)N1CCc2ccccc2C1", "3,4-dihydroisoquinoline-2(1H)-carboxylic acid"),
    ("O=CN1C=CC=CC1", "pyridine-1(2H)-carbaldehyde"),
])
def test_producer(smiles, name):
    assert name_azine_nitrogen_suffix(Chem.MolFromSmiles(smiles)) == name


@pytest.mark.parametrize("smiles", ["OC(=O)N1CCCCC1", "OC(=O)N1C=CC=CC1C", "OC(=O)N1C=CC=CC1.O"])
def test_out_of_scope(smiles):
    assert name_azine_nitrogen_suffix(Chem.MolFromSmiles(smiles)) is None


@pytest.mark.roundtrip
def test_book_name():
    assert name_compound("O=C(O)N1CC=Cc2ccccc21") == "quinoline-1(2H)-carboxylic acid"
