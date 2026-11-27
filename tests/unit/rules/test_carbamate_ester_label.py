"""A carbamate ester that is the senior class is named as the ester, the Blue Book: "All preferred
IUPAC names for esters are named by functional class nomenclature";,:18182, "9 Esters" above acid halides,
amides and ureas, nitriles, ketones, alcohols, amines and the heteroatom classes; '2-hydroxypropyl (2-aminoethyl)carbamate
(PIN)',:30766). A name on a junior parent with an '(alkoxycarbonyl)' prefix is correct but no PIN: the spelling check
(validation/spelling/checks_carbamate_ester.py) lowers its label; the name is kept.
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import registered_rules
from orthonym.validation.spelling.checks_carbamate_ester import carbamate_ester_name_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("C1=CC=C2C(=C1)C(C3=CC=CC=C32)COC(=O)NCCS(=O)(=O)Cl",
     "2-({[(9H-fluoren-9-yl)methoxy]carbonyl}amino)ethane-1-sulfonyl chloride"),
    ("CC(C)(C)OC(=O)N1CCCCC1", "1-(tert-butoxycarbonyl)piperidine"),
    ("CC(C)(C)OC(=O)NC(=O)NC", "1-(tert-butoxycarbonyl)-3-methylurea"),
])
def test_a_non_ester_name_is_flagged(smiles, name):
    failure = carbamate_ester_name_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-65.6.3.2.1", name


@pytest.mark.parametrize("smiles,name", [
    ("CC(C)(C)OC(=O)N1CCCCC1", "tert-butyl piperidine-1-carboxylate"),
    ("COC(=O)N[Si](C)(C)C", "methyl (trimethylsilyl)carbamate"),
    ("CC(C)(C)OC(=O)N(C(=O)OC(C)(C)C)c1ccccc1", "di-tert-butyl phenyl-2-imidodicarbonate"),   #:31049
    ("OB(O)c1ccc(NC(=O)OC(C)(C)C)cc1", "{4-[(tert-butoxycarbonyl)amino]phenyl}boronic acid"),   # an acid outranks it
    ("CC(C)(C)OC(=O)NCC(=O)O", "[(tert-butoxycarbonyl)amino]acetic acid"),
    ("COC(=O)NCCC(=O)OC", "methyl 3-[(methoxycarbonyl)amino]propanoate"),                     # another ester: own rule
    ("CCOC(=N)CCNC(=O)OC", "ethyl 3-[(methoxycarbonyl)amino]propanimidate"),                   # an imidic ester
    ("O=C1OCCN1", "1,3-oxazolidin-2-one"),                                                     # a cyclic carbamate
    ("COC(=O)NCC[NH3+].[Cl-]", "2-[(methoxycarbonyl)amino]ethan-1-aminium chloride"),          # a salt
    ("CCOC(=O)N=[N+]=[N-]", "ethyl carbonazidate"),                 # 'carbonazidic acid (PIN)',:30834
    ("CCOC(=O)NCC(=O)OO", "[(ethoxycarbonyl)amino]ethaneperoxoic acid"),                        # a peroxy acid
    ("CCOC(=O)NCCS(=O)(=O)OS(=O)(=O)CCNC(=O)OCC",
     "bis{2-[(ethoxycarbonyl)amino]ethane-1-sulfonic} anhydride"),                              # a noncarbon anhydride
    ("CCOC(=O)NCCNC(=O)OCC", "diethyl ethane-1,2-diyldicarbamate"),
])
def test_ester_names_and_other_classes_pass(smiles, name):
    assert carbamate_ester_name_check(Chem.MolFromSmiles(smiles), name) is None, name


def test_the_check_is_registered():
    assert "P-65.6.3.2.1" in registered_rules()
