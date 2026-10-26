"""The ring oxygen of a lactone is not "another ester" of the alcohol component (lane TI2).

 "When anions are different, two methods are used" (the Blue Book): an ester whose alcohol
component carries another ester of a different anion, cited as a prefix, is named by method (2) below the PIN,
``(acetyloxy)methyl formate``. The lactone of a coumarin or a pyranone is a ring of the alcohol component, named as
a pseudoketone ('2-oxo-2H-1-benzopyran-7-yl'), not an acyloxy prefix, so the name of its ester is not labelled
below the PIN (``rules.esters.record_polyol_mixed_anion_non_pin``).
"""
import pytest
from rdkit import Chem

from orthonym.metrics import provenance as pv
from orthonym.rules.esters import record_polyol_mixed_anion_non_pin

pytestmark = pytest.mark.unit


def _label(smiles, name):
    """The label-only records after the producer reports the ester of ``smiles`` (the first O-C(=O) found)."""
    pv.clear_provenance()
    mol = Chem.MolFromSmiles(smiles)
    ester_o, carbonyl_c = None, None
    for match in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)([#6])[OX2;!R][#6]")):
        carbonyl_c, ester_o, alkyl_c = match[0], match[3], match[4]
        break
    record_polyol_mixed_anion_non_pin(mol, alkyl_c, ester_o, name)
    return pv.get_provenance()["non_pin_labels"]


def test_an_ester_of_a_coumarin_is_not_labelled_below_the_pin():
    assert _label("CC(=O)Oc1ccc2ccc(=O)oc2c1", "2-oxo-2H-1-benzopyran-7-yl acetate") == ()


def test_an_ester_of_an_alcohol_with_an_acyloxy_group_still_is():
    # '(acetyloxy)methyl formate' is the example of the book, method (2):31840
    assert _label("O=COCOC(C)=O", "(acetyloxy)methyl formate") == ("(acetyloxy)methyl formate",)
