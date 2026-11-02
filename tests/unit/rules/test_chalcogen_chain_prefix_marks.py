"""An alkyl on a chalcogen chain prefix is a compound prefix and takes enclosing marks.

IUPAC 2013 (compound prefixes are enclosed); the book's own names:
'(methyldiselanyl)(methylsulfanyl)methane (PIN)' (the Blue Book) and
'1-(methyldiselanyl)-2-(methyldisulfanyl)ethane (PIN)' (:27880).
"""

import pytest

from orthonym import name_compound
from orthonym.assembly.naming_utils import is_complex_substituent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("prefix", ["methyldisulfanyl", "methyldiselanyl", "methylditellanyl", "ethyltrisulfanyl"])
def test_alkyl_chalcogen_chain_prefix_is_compound(prefix):
    assert is_complex_substituent(prefix)


@pytest.mark.parametrize("bare", ["disulfanyl", "diselanyl", "ditellanyl", "trisulfanyl"])
def test_bare_chalcogen_chain_prefix_is_simple(bare):
    assert not is_complex_substituent(bare)


@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,name", [
    ("CSC[Se][Se]C", "(methyldiselanyl)(methylsulfanyl)methane"),
    ("C[Se][Se]CCSSC", "1-(methyldiselanyl)-2-(methyldisulfanyl)ethane"),
    ("C[Se][Se]CC(=O)O", "(methyldiselanyl)acetic acid"),
])
def test_book_names(smiles, name):
    assert name_compound(smiles) == name
