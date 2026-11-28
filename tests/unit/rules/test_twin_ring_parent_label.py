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


from orthonym.validation.spelling.checks_twin_ring_parent import twin_suffix_ring_parent_check  # noqa: E402


@pytest.mark.parametrize("smiles,name", [
    ("OC(=O)c1ccccc1Oc1cccc(c1)C(=O)O", "3-(2-carboxyphenoxy)benzoic acid"),
    ("Nc1c(Cl)cccc1Cc1ccc(N)c(Cl)c1", "2-[(4-amino-3-chlorophenyl)methyl]-6-chloroaniline"),
    ("Oc1ccc(C2NCCNC2c2c(Cl)cc(O)cc2Cl)c(F)c1", "4-[3-(2,6-dichloro-4-hydroxyphenyl)piperazin-2-yl]-3-fluorophenol"),
    ("N#Cc1ccc(Cl)cc1Cc1cccc(C#N)c1", "3-[(5-chloro-2-cyanophenyl)methyl]benzonitrile"),       # the book's 'not' name
])
def test_a_suffix_twin_on_the_junior_ring_is_flagged(smiles, name):
    assert twin_suffix_ring_parent_check(Chem.MolFromSmiles(smiles), name) is not None, name


@pytest.mark.parametrize("smiles,name", [
    ("N#Cc1ccc(Cl)cc1Cc1cccc(C#N)c1", "4-chloro-2-[(3-cyanophenyl)methyl]benzonitrile"),       #:21614
    ("OC(=O)c1ccccc1Oc1cccc(c1)C(=O)O", "2-(3-carboxyphenoxy)benzoic acid"),
    ("Oc1ccc(Cc2ccc(O)cc2)cc1", "4,4'-methylenediphenol"),                                      # multiplicative
    ("Oc1ccc(Oc2ccc(O)cc2Cl)cc1", "3-chloro-4-(4-hydroxyphenoxy)phenol"),                       # more prefixes win
    ("Oc1ccc(cc1)C1=C(Cc2cc(O)ccc12)c1cccnc1", "4-[6-hydroxy-2-(pyridin-3-yl)-1H-inden-3-yl]phenol"),   # a fused ring
])
def test_suffix_twin_names_on_the_senior_ring_pass(smiles, name):
    assert twin_suffix_ring_parent_check(Chem.MolFromSmiles(smiles), name) is None, name
