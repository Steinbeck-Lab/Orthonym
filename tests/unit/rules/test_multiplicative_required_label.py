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
    # a ring centre or a multi-atom link,:5186; '3,3'-(1,4-phenylene)di(propan-1-ol) (PIN)',:25171)
    ("O=C(Nc1ccc(NC(=O)c2ccccc2)cc1)c1ccccc1", "N-(4-benzamidophenyl)benzamide"),
    ("CC(=O)CC(=O)c1ccc(cc1)C(=O)CC(C)=O", "1-[4-(3-oxobutanoyl)phenyl]butane-1,3-dione"),
    ("Cc1nc(CO)c(C)nc1CO", "[5-(hydroxymethyl)-3,6-dimethylpyrazin-2-yl]methanol"),
    ("O=C1CSC(c2ccccc2)N1CCN1C(=O)CSC1c1ccccc1",
     "3-[2-(4-oxo-2-phenyl-1,3-thiazolidin-3-yl)ethyl]-2-phenyl-1,3-thiazolidin-4-one"),
    ("OC(=O)COc1ccc(cc1)S(=O)(=O)c1ccc(OCC(O)=O)cc1", "(4-{[4-(carboxymethoxy)phenyl]sulfonyl}phenoxy)acetic acid"),
    ("OCCCc1ccc(CCCO)cc1", "3-[4-(3-hydroxypropyl)phenyl]propan-1-ol"),
    ("O=C=Nc1ccc(Oc2ccc(Oc3ccc(N=C=O)cc3)cc2)cc1", "1-isocyanato-4-[4-(4-isocyanatophenoxy)phenoxy]benzene"),
    ("Cc1ccc(NC(=O)c2ccccc2)cc1NC(=O)c1ccccc1", "N-(5-benzamido-2-methylphenyl)benzamide"),   # a substituted ring link
    # two chain O are fewer than the four heterounits of an 'a' name,:23348); the PIN is multiplicative,
    # as '2,2'-[oxybis(ethane-2,1-diyloxy)]diacetic acid (PIN, a multiplicative name)' (:23210)
    ("OCCOCCOCCO", "3,6-dioxaoctane-1,8-diol"),
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
    ("CC(=O)CCCC(C)=O", "heptane-2,6-dione"),                                       # one chain holds both groups
    ("O=C(CCCC(=O)c1ccccc1)c1ccccc1", "1,5-diphenylpentane-1,5-dione"),
    ("CC1OC(=O)OC1C", "4,5-dimethyl-1,3-dioxolan-2-one"),
    ("NCCNCCN", "N1-(2-aminoethyl)ethane-1,2-diamine"),                            # an amine centre
    ("OCCCc1ccc(CCCO)cc1", "3,3'-(1,4-phenylene)di(propan-1-ol)"),                  #:25171
    ("OC(=O)c1cc(Cl)c(C(=O)O)cc1Cl", "2,5-dichlorobenzene-1,4-dicarboxylic acid"),  # the groups are the ring's suffixes
    ("N#Cc1cc(Cl)cc(C#N)n1", "4-chloropyridine-2,6-dicarbonitrile"),
    ("O=Cc1ccc2c(c1)c1cc(C=O)ccc1n2C", "9-methyl-9H-carbazole-3,6-dicarbaldehyde"),
    ("O=C=Nc1c(Cl)c(Cl)c(Cl)c(N=C=O)c1Cl", "1,2,3,5-tetrachloro-4,6-diisocyanatobenzene"),
    ("COC1=C(c2ccc(O)cc2)C(Cc2ccc(O)cc2)OC1=O",
     "4-(4-hydroxyphenyl)-5-[(4-hydroxyphenyl)methyl]-3-methoxyfuran-2(5H)-one"),   # the centre ring has its own suffix
    ("COc1ccc(C2=C(C(C)=O)C(=O)OC(=O)C2C(C)=O)cc1", "3,5-diacetyl-4-(4-methoxyphenyl)-2H-pyran-2,6(3H)-dione"),
    # an unsymmetrical link of several groups is no multiplicative group (2);,:5238)
    ("Nc1ccc(CCOc2ccc(N)cc2)cc1", "4-[2-(4-aminophenoxy)ethyl]aniline"),
    ("Oc1c(Cl)cc(Cc2ccc(Oc3cc(Cl)c(O)c(Cl)c3)cc2)cc1Cl",
     "2,6-dichloro-4-{[4-(3,5-dichloro-4-hydroxyphenoxy)phenyl]methyl}phenol"),
    ("OCCOCCSCCO", "2-{[2-(2-hydroxyethoxy)ethyl]sulfanyl}ethan-1-ol"),
    ("NCc1cccc(CN)c1", "(1,3-phenylene)dimethanamine"),                             # already multiplicative
    ("OC(=O)OC(=O)O", "dicarbonic acid"),                                          # no parent carbon in the halves
    ("CNC(=O)c1ccc(C(=O)NC)cc1", "N1,N4-dimethylbenzene-1,4-dicarboxamide"),       # the amides are the ring's suffix
    ("N=C(O)c1ccc(C(=N)O)cc1", "benzene-1,4-dicarboximidic acid"),
])
def test_other_names_pass(smiles, name):
    assert multiplicative_required_check(Chem.MolFromSmiles(smiles), name) is None, name
