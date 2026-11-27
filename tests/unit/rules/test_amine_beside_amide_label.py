"""A molecule with an amide (or a pseudoketone: a ring-N acyl group, a lactam) is never named on an amine parent
, the Blue Book: amides class 11, pseudoketones 16, amines 19; "Anilides",:32849;
 "'Hidden' amides",:33125, '1-(piperidin-1-yl)ethan-1-one (PIN)',:33129). The spelling check
(validation/spelling/checks_amine_beside_amide.py) lowers the label of such a name; the name is kept. A sulfonyl on a
ring N and a sultam are left alone.
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_amine_beside_amide import amine_parent_beside_amide_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("O=C(Nc1cccc(Br)c1)c1ccc2OCOc2c1", "N-(2H-1,3-benzodioxole-5-carbonyl)-3-bromoaniline"),
    ("CC(=O)N1CCC(N)CC1", "1-acetylpiperidin-4-amine"),
    ("NCCc1ccc(CS(N)(=O)=O)cc1", "2-[4-(sulfamoylmethyl)phenyl]ethan-1-amine"),
    ("NC(=S)Nc1ccc(N)cc1", "4-(carbamothioylamino)aniline"),
    ("CC(=O)n1cnc(CCN)c1", "2-(1-acetyl-1H-imidazol-4-yl)ethan-1-amine"),       # ring-N acyl on an aromatic N
    ("NNC(=O)c1ccc(N)cc1", "4-(hydrazinecarbonyl)aniline"),                      # a hydrazide (class 12)
    ("CC(=S)Nc1ccc(N)cc1", "N-(ethanethioyl)benzene-1,4-diamine"),               # a thioamide
])
def test_an_amine_parent_beside_an_amide_is_flagged(smiles, name):
    failure = amine_parent_beside_amide_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-41", name


@pytest.mark.parametrize("smiles,name", [
    ("O=C(Nc1cccc(Br)c1)c1ccc2OCOc2c1", "N-(3-bromophenyl)-2H-1,3-benzodioxole-5-carboxamide"),
    ("CS(=O)(=O)N1CCC(N)CC1", "1-(methanesulfonyl)piperidin-4-amine"),     # sulfonyl on a ring N: left alone
    ("NC1=CC=C2C(=C1)CCS2(=O)=O", "1,1-dioxo-2,3-dihydro-1-benzothiophen-5-amine"),   # no S-N
    ("CON", "O-methylhydroxylamine"),
    ("NCCN", "ethane-1,2-diamine"),
    ("C[N+](C)(C)CCNC(C)=O", "2-acetamido-N,N,N-trimethylethan-1-aminium"),   # an ion
    ("Cc1ccc(N)cc1N=C=O", "3-isocyanato-4-methylaniline"),                   # an isocyanate is no amide
    ("NCCN=C=NC1CCCCC1", "2-[(cyclohexylimino)methylidene]aminoethan-1-amine"),   # a carbodiimide
    ("Nc1ccc([N+](=O)[O-])cc1", "4-nitroaniline"),                           # nitro: no ion, no amide
    ("NCCNC(C)=O.Cl", "N-(2-aminoethyl)acetamide hydrochloride"),            # a mixture: left alone
    ("NC1=CC=C2C(=C1)CS(=O)(=O)N2", "2,2-dioxo-1,3-dihydro-2,1-benzothiazol-5-amine"),   # a sultam
])
def test_other_names_pass(smiles, name):
    assert amine_parent_beside_amide_check(Chem.MolFromSmiles(smiles), name) is None, name
