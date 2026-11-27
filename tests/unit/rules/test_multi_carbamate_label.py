"""Two or more carbamate esters have a name of their own (validation/spelling/checks_carbamate_ester.py,
multi_carbamate_ester_name_check): a ring-N carbamate is the ester of a ring carboxylic acid, senior to an open
carbamate, the Blue Book); NH carbamates on one parent or esters of one polyol take a multiplicative
name (d),:5166;,:23178;,:31794-:31803;,:31819). A name that
cites one carbamate ester as a prefix is then correct but no PIN.
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_carbamate_ester import multi_carbamate_ester_name_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("CC(C)(C)OC(=O)NC1CCN(CC1)C(=O)OCc1ccccc1", "tert-butyl {1-[(benzyloxy)carbonyl]piperidin-4-yl}carbamate"),
    ("O=C(Nc1ccc(NC(=O)OCc2ccccc2)cc1)OCc1ccccc1", "benzyl (4-{[(benzyloxy)carbonyl]amino}phenyl)carbamate"),
    ("Cc1ccc(NC(=O)OC(C)C)cc1NC(=O)OC(C)C", "propan-2-yl {4-methyl-3-[(propan-2-yloxy)carbonyl]aminophenyl}carbamate"),
    ("CCOC(=O)Nc1cc(NC(=O)OCC)ncn1", "ethyl {6-[(ethoxycarbonyl)amino]pyrimidin-4-yl}carbamate"),
    ("COC(=O)NCCCCCCNC(=O)OCC", "ethyl {6-[(methoxycarbonyl)amino]hexyl}carbamate"),        # different alkoxy groups
    ("Cc1ccc(cc1)C(COC(N)=O)(COC(N)=O)c1ccc(C)cc1", "3-(carbamoyloxy)-2,2-bis(4-methylphenyl)propyl carbamate"),
    ("CNC(=O)OCCCOC(=O)Nc1ccccc1", "3-[(methylcarbamoyl)oxy]propyl phenylcarbamate"),        # one diol, two anions
])
def test_a_prefix_name_of_two_carbamate_esters_is_flagged(smiles, name):
    assert multi_carbamate_ester_name_check(Chem.MolFromSmiles(smiles), name) is not None, name


@pytest.mark.parametrize("smiles,name", [
    ("CC(C)(C)OC(=O)NC1CCN(CC1)C(=O)OCc1ccccc1", "benzyl 4-[(tert-butoxycarbonyl)amino]piperidine-1-carboxylate"),
    ("O=C(Nc1ccc(NC(=O)OCc2ccccc2)cc1)OCc1ccccc1", "dibenzyl 1,4-phenylenebis(carbamate)"),
    ("CN(C(=O)OC)c1cccc(NC(=O)OC)c1", "methyl {3-[(methoxycarbonyl)(methyl)amino]phenyl}carbamate"),   # N-methyl + NH
    ("COC(=O)Nc1cccc(OC(=O)N(C)CCCC)c1", "3-[(methoxycarbonyl)amino]phenyl butyl(methyl)carbamate"),  # mixed N/O
    ("CCOC(=O)NC1CC(C)(C)CC(C)(CNC(=O)OCC)C1",
     "ethyl ({5-[(ethoxycarbonyl)amino]-1,3,3-trimethylcyclohexyl}methyl)carbamate"),        # ring + side-chain N
    ("CC(C)(C)OC(=O)NCCSC(=O)NC", "tert-butyl {2-[(methylcarbamoyl)sulfanyl]ethyl}carbamate"),   # an S-thiocarbamate
    ("CC(C)(C)OC(=O)NCC(=O)O", "[(tert-butoxycarbonyl)amino]acetic acid"),                    # one carbamate, an acid
    ("CNC(=O)OCCOC(=O)N1CCC1", "ethane-1,2-diyl azetidine-1-carboxylate methylcarbamate"),   # method (1)
    ("CCC(COC(=O)Nc1ccc(C)c(NC(=O)OCC=C)c1)(OCC=C)OCC=C",
     "prop-2-en-1-yl {5-[({2,2-bis[(prop-2-en-1-yl)oxy]butoxy}carbonyl)amino]-2-methylphenyl}carbamate"),  # unequal
    ("O=C(Nc1ccc(-c2ccc(NC(=O)OC)cc2)cc1)OC", "methyl {4'-[(methoxycarbonyl)amino][1,1'-biphenyl]-4-yl}carbamate"),
    ("CC(C)(C)OC(=O)N1CCC(CC1)C1CCN(CC1)C(=O)OC(C)(C)C", "di-tert-butyl [4,4'-bipiperidine]-1,1'-dicarboxylate"),
    ("C[13CH2]OC(=O)Nc1ccc(NC(=O)OCC)cc1", "(1-13C)ethyl {4-[(ethoxycarbonyl)amino]phenyl}carbamate"),     # an isotope
])
def test_other_names_pass(smiles, name):
    assert multi_carbamate_ester_name_check(Chem.MolFromSmiles(smiles), name) is None, name
