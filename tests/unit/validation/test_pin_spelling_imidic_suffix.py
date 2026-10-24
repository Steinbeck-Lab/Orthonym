"""Spelling check: the suffix of an unsaturated imidic acid sits at C-1 (``validation/spelling/models.py``).

 (the Blue Book) forms the imidic acid by changing 'oic acid' to 'imidic acid', so
'but-3-enimidic acid' is written as 'but-3-enoic acid' is, (c): the principal
characteristic group takes the lowest locant, here C-1 by the structure of the group). The numbering
model read the ending of 'but-3-en' + 'imidic' as having no suffix, so it numbered the chain from the
double bond's end and called the correct name's suffix locant '[3] > [2]': every unsaturated imidic
acid with its double bond beyond C-2 was declined at the PIN tier, whatever its spelling. The imidic
esters ('imidate', 'imidothioate') were in the ending table already; the acids, whose suffix the
hydroximic acids use too,:30096), were not.
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import check_pin_spelling
from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate


def _rules(smiles, name):
    return [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)]


@pytest.mark.parametrize("smiles,name", [
    ("N=C(O)CC=C", "but-3-enimidic acid"),
    ("N=C(S)CC=C", "but-3-enimidothioic acid"),
    ("N=C([SeH])CCC=C", "pent-4-enimidoselenoic acid"),
    ("N=C([TeH])CC=C", "but-3-enimidotelluroic acid"),
    ("ON=C(O)CCC=C", "N-hydroxypent-4-enimidic acid"),
    ("ON=C(O)CC=C", "N-hydroxybut-3-enimidic acid"),
])
def test_the_suffix_of_an_unsaturated_imidic_acid_is_at_c1(smiles, name):
    assert _rules(smiles, name) == []


@pytest.mark.parametrize("smiles,pin", [
    ("N=C(O)CC=C", "but-3-enimidic acid"),
    ("N=C(S)CC=C", "but-3-enimidothioic acid"),
    ("N=C(O)CCC=C", "pent-4-enimidic acid"),
    ("N=C(O)C=CC", "but-2-enimidic acid"),
])
def test_an_unsaturated_imidic_acid_is_the_pin_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
