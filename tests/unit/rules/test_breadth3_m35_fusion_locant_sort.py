"""Breadth job 3, class M35 (crash): a fused-ring substituent group whose locants mix
fusion letters and numerals ('4a' and 7) was sorted with a bare sorted, which raised
TypeError in heterocycles.name_substituted_heterocycle and abstained the whole name.
 (the Blue Book): "locants consisting of a number and a lower-case letter
... as 4a... are placed immediately after the corresponding numeric locant".

Every name is read back by a FRESH OPSIN call that does not go through the engine
(tests.support.rt_assert._independent_parse) and compared by full InChIKey.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.naming_utils import locant_sort_key
from tests.support.rt_assert import _independent_parse

pytestmark = [pytest.mark.opsin_gate]


def _key(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToInchiKey(mol) if mol is not None else ""


def test_fusion_locant_sorts_after_its_numeral():
    assert sorted([5, "4a", 4, 8, "8a", 1], key=locant_sort_key) == [1, 4, "4a", 5, 8, "8a"]
    # the italic-letter class still leads examples, unchanged)
    assert sorted([2, "N", 1], key=locant_sort_key) == ["N", 1, 2]


@pytest.mark.parametrize("smiles,expected", [
    # the minimal reproduction of the M35 rows
    ("CC1CCC2(C)C(=CC(=O)CC2)C1",
     "4a,7-dimethyl-4,4a,5,6,7,8-hexahydronaphthalen-2(3H)-one"),
    # the a dev split / milestone1500 M35 row
    ("C=C[C@](C)(O)CC[C@@]1(C)[C@H](C)CC[C@@]2(C)C(C)=CC(=O)C[C@H]12",
     "(4aR,7R,8S,8aR)-8-[(3R)-3-hydroxy-3-methylpent-4-en-1-yl]-4,4a,7,8-tetramethyl-"
     "4a,5,6,7,8,8a-hexahydronaphthalen-2(1H)-one"),
])
def test_fused_ketone_with_fusion_locant_prefixes_names(smiles, expected):
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified", row
    assert _key(_independent_parse(expected)) == _key(smiles)
