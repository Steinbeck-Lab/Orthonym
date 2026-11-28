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


from orthonym.validation.spelling.checks_twin_ring_parent import aniline_twin_parent_check  # noqa: E402


@pytest.mark.parametrize("smiles,name", [
    ("CN(C)C1=CC=C(C=C1)CC2=CC=C(C=C2)N", "4-{[4-(dimethylamino)phenyl]methyl}aniline"),            # count 1 < 3
    ("CN(C)C1=CC=CC=C1SC2=C(C=C(C=C2)I)N", "2-{[2-(dimethylamino)phenyl]sulfanyl}-5-iodoaniline"),  # count 2 < 3
    ("C1=CC=C(C(=C1)NC2=CC=C(C=C2)Cl)[N+](=O)[O-]", "4-chloro-N-(2-nitrophenyl)aniline"),          # N,4 > N,2
    ("C1=CC(=C(C=C1F)F)NC2=CC(=CC(=C2)F)F", "N-(2,4-difluorophenyl)-3,5-difluoroaniline"),         # N,3,5 > N,2,4
])
def test_an_aniline_twin_on_the_junior_ring_is_flagged(smiles, name):
    assert aniline_twin_parent_check(Chem.MolFromSmiles(smiles), name) is not None, name


@pytest.mark.parametrize("smiles,name", [
    ("COc1ccc(Nc2ccccc2)cc1", "4-methoxy-N-phenylaniline"),                                       #:21610
    ("C1=CC=C(C(=C1)NC2=CC=C(C=C2)Cl)[N+](=O)[O-]", "N-(4-chlorophenyl)-2-nitroaniline"),
    ("COC1=CC=C(C=C1)NC2=CC=C(C=C2)Cl", "4-chloro-N-(4-methoxyphenyl)aniline"),                   # equal sets: not judged
    ("CC(=O)Nc1ccc(Nc2ccccc2)cc1", "N-[4-(phenylamino)phenyl]acetamide"),                         # an amide name
    ("Nc1ccc(Cc2ccc(N)cc2)cc1", "4,4'-methylenedianiline"),                                       # multiplicative
])
def test_aniline_twin_names_on_the_senior_ring_pass(smiles, name):
    assert aniline_twin_parent_check(Chem.MolFromSmiles(smiles), name) is None, name
