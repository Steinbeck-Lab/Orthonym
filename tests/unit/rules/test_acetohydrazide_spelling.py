"""The two-carbon hydrazide is 'acetohydrazide', never 'ethanehydrazide'.

 (the Blue Book): "Names of hydrazides are formed by changing the 'ic acid' or '-oic acid'
ending of the retained names of carboxylic acids into 'ohydrazide'"; (:33938): "Only the following
five names are preferred IUPAC names and can be substituted in the same way as corresponding amides...
acetohydrazide (PIN) benzohydrazide (PIN) oxalohydrazide (PIN)" (:33944; formohydrazide and cyanohydrazide
beside them). The producers wrote the systematic stem for a substituted two-carbon hydrazide ('N'-methylethane
hydrazide', '2-(methylsulfanyl)ethanehydrazide') and shipped it as pin_verified, while the amide of the same
chain has had the retained name since (:32691). The retained C2 spelling lives in ONE table,
``composition_primitives.RETAINED_C2_ACYL_BY_SUFFIX``, read by the general assembler, the polyfunctional
producer and (through the assembler) the N/N'-substituted hydrazide producer in composer.py.
"""
import pytest

from orthonym.assembly.composition_primitives import RETAINED_C2_ACYL_BY_SUFFIX
from tests.support.read_back import same_key

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("CNNC(=O)C", "N'-methylacetohydrazide"),                         # the review's row
    ("CN(N)C(C)=O", "N-methylacetohydrazide"),
    ("NNC(=O)CSC", "2-(methylsulfanyl)acetohydrazide"),
    ("NNC(=O)CCl", "2-chloroacetohydrazide"),
])
def test_substituted_acetohydrazides(namer, smiles, expected):
    row = namer.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert same_key(smiles, expected)


@pytest.mark.parametrize("smiles,expected", [
    ("NNC(=O)C", "acetohydrazide"),                                    #:33944
    ("NNC(=O)CC", "propanehydrazide"),
    ("NNC(=O)CCC", "butanehydrazide"),
    ("NC(=O)CSC", "2-(methylsulfanyl)acetamide"),                      # the amide of the same chain: unchanged
    ("NNC(=S)C", None),                                                # thiohydrazide: not in the table
])
def test_neighbours(namer, smiles, expected):
    name = namer.name(smiles)
    if expected is None:
        assert "acetohydrazide" not in name
    else:
        assert name == expected


def test_the_table_is_the_one_home():
    assert RETAINED_C2_ACYL_BY_SUFFIX == {"amide": "acetamide", "hydrazide": "acetohydrazide"}
