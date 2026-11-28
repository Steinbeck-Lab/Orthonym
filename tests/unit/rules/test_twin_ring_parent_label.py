"""Of two identical benzene rings, the parent has more prefixes, then the lower locant set, the Blue Book,
'1-methyl-4-(phenoxymethyl)benzene (PIN) [not [(4-methylphenyl)methoxy]benzene]',:21618;,:21698). A name on the
other ring is correct but no PIN (validation/spelling/checks_twin_ring_parent.py).
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_twin_ring_parent import twin_ring_parent_check
from orthonym.validation.spelling.checks_multiplicative_required import multiplicative_required_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("Cc1ccc(COc2ccccc2)cc1", "[(4-methylphenyl)methoxy]benzene"),                       # the book's 'not' name
    ("Clc1ccc(OCc2ccccc2)cc1", "[(4-chlorophenoxy)methyl]benzene"),
    ("BrC1=CC=C(SCc2ccccc2)C=C1", "{[(4-bromophenyl)sulfanyl]methyl}benzene"),
    ("Cc1ccc(Oc2ccc(cc2[N+](=O)[O-])[N+](=O)[O-])cc1", "1-(2,4-dinitrophenoxy)-4-methylbenzene"),
    ("Cc1ccc(Oc2ccccc2[N+](=O)[O-])cc1", "1-methyl-4-(2-nitrophenoxy)benzene"),          # 1,4 is higher than 1,2
])
def test_a_name_on_the_junior_ring_is_flagged(smiles, name):
    assert twin_ring_parent_check(Chem.MolFromSmiles(smiles), name) is not None, name


@pytest.mark.parametrize("smiles,name", [
    ("Cc1ccc(COc2ccccc2)cc1", "1-methyl-4-(phenoxymethyl)benzene"),                      #:21618
    ("Clc1ccc(OCc2ccccc2)cc1", "1-(benzyloxy)-4-chlorobenzene"),
    ("Cc1ccc(Oc2ccccc2[N+](=O)[O-])cc1", "1-(4-methylphenoxy)-2-nitrobenzene"),
    ("Clc1ccc(Oc2ccc(Br)cc2)cc1", "1-bromo-4-(4-chlorophenoxy)benzene"),                 # equal sets: later criteria
    ("c1ccc(-c2ccccc2)cc1", "1,1'-biphenyl"),
    ("Oc1ccc(Oc2ccccc2)cc1", "4-phenoxyphenol"),                                         # a suffix
    ("c1ccc(Cc2ccc(Cc3ccccc3)cc2)cc1", "1,4-dibenzylbenzene"),                           # three rings
])
def test_other_names_pass(smiles, name):
    assert twin_ring_parent_check(Chem.MolFromSmiles(smiles), name) is None, name


def test_an_oh_with_an_explicit_hydrogen_does_not_break_the_twin_check():
    mol = Chem.MolFromSmiles("[H]OCCO[H]", sanitize=True)
    mol = Chem.AddHs(mol)
    assert multiplicative_required_check(mol, "ethane-1,2-diol") is None
