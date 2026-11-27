"""Two identical parent structures on one symmetric link atom take a multiplicative name "Preferred IUPAC
multiplicative names", the Blue Book; '1,1'-sulfinyldibenzene (PIN)',:28110; '1,1'-methylenedibenzene (PIN)',
:19360; '2,2'-sulfanediyldiacetic acid (PIN)',:23192). A substitutive name is then no PIN
(validation/spelling/checks_multiplicative_required.py).
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_multiplicative_required import multiplicative_required_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("C1=CSC(=C1)S(=O)(=O)C2=CC=CS2", "2-(thiophene-2-sulfonyl)thiophene"),
    ("C1=CC(=CN=C1)S(=O)C2=CN=CC=C2", "3-(pyridine-3-sulfinyl)pyridine"),
    ("C(C(COCC(CO)O)O)O", "3-(2,3-dihydroxypropoxy)propane-1,2-diol"),
    ("C1=CC(=CC=C1CC2=CC=C(C=C2)N=C=O)N=C=O", "1-isocyanato-4-[(4-isocyanatophenyl)methyl]benzene"),
    ("C(CSCCC(=O)NN)C(=O)NN", "3-[(3-hydrazinyl-3-oxopropyl)sulfanyl]propanehydrazide"),
    ("Clc1ccc(Oc2ccc(Cl)cc2)cc1", "1-chloro-4-(4-chlorophenoxy)benzene"),
])
def test_a_substitutive_name_of_twins_is_flagged(smiles, name):
    failure = multiplicative_required_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-51.3.1", name


@pytest.mark.parametrize("smiles,name", [
    ("C1=CSC(=C1)S(=O)(=O)C2=CC=CS2", "2,2'-sulfonyldithiophene"),
    ("CCOCC", "ethoxyethane"),                                                     # alkane units
    ("c1ccc(Nc2ccccc2)cc1", "N-phenylaniline"),                                    # an N centre (:26233)
    ("Clc1ccccc1Oc1ccc(Cl)cc1", "1-chloro-2-(4-chlorophenoxy)benzene"),            # unequal halves (:27796)
    ("O=C(c1ccccc1)c1ccccc1", "diphenylmethanone"),                                # a C=O centre
    ("CSSSC", "dimethyltrisulfane"),                                               # a polysulfane parent
    ("OCCOCCOCCO", "2,2'-[ethane-1,2-diylbis(oxy)]di(ethan-1-ol)"),               # already multiplicative
    ("OCCOCCOCCO", "3,6-dioxaoctane-1,8-diol"),                                     # chain hetero atoms: not judged
    ("C[C@H](O)COC[C@@H](C)O", "1-[(2S)-2-hydroxypropoxy]propan-2-ol"),            # unequal stereo halves
    ("c1ccc(-c2ccccc2)cc1", "1,1'-biphenyl"),
    ("C[Si](C)(C1CCCCC1)O[Si](C)(C)C1CCCCC1", "1,3-dicyclohexyl-1,1,3,3-tetramethyldisiloxane"),   # a parent hydride
    ("OCc1ccccc1[Te]c1ccccc1CO", "[tellanediyldi(2,1-phenylene)]dimethanol"),       # already multiplicative
    ("O=C(c1ccccc1)c1ccc(Oc2ccc(C(=O)c3ccccc3)cc2)cc1",
     "4-oxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphane-2,6-dione"),                    # a phane name
    ("OCOCO", "oxydimethanol"),                                                     # unprimed multiplicative names
    ("NCOCN", "oxydimethanamine"),
    ("OCS(=O)(=O)CO", "sulfonyldimethanol"),
    ("OCCNCCNCCOCCNCCNCCO", "9-oxa-3,6,12,15-tetraazaheptadecane-1,17-diol"),      # an 'a' name
    ("CC(=O)OOC(C)=O", "acetyl ethaneperoxoate"),
    ("CSSC", "dimethyldisulfane"),
])
def test_other_names_pass(smiles, name):
    assert multiplicative_required_check(Chem.MolFromSmiles(smiles), name) is None, name
