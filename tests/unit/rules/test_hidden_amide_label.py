"""An acyl on a ring nitrogen is a pseudoketone "'Hidden' amides", the Blue Book: "now preferably named as
a pseudoketone... and not as an acyl substituent on the nitrogen as in previous recommendations; the latter method can
be used in general nomenclature"; '1-(piperidin-1-yl)ethan-1-one (PIN)',:33129). A name that cites it as an acyl
prefix ('1-acetyl-1H-indole') is correct but no PIN; the spelling checks lower its label and keep the name.
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_parent import principal_class_check
from orthonym.validation.spelling.checks_suffix_class_prefix import suffix_class_as_prefix_check

pytestmark = pytest.mark.unit


def _fails(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    return principal_class_check(mol, name) or suffix_class_as_prefix_check(mol, name)


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)n1ccc2ccccc21", "1-acetyl-1H-indole"),
    ("CC(=O)n1c2ccccc2c2ccccc21", "9-acetyl-9H-carbazole"),
    ("CCC(=O)N1C2CCC1CNC2", "8-propanoyl-3,8-diazabicyclo[3.2.1]octane"),
    ("CC(=O)n1ccc2cc(O)ccc21", "1-acetyl-1H-indol-5-ol"),
    ("CC(=O)N1CCCCC1", "1-acetylpiperidine"),                    # the book's own non-PIN form (:33129)
    ("CC(=O)N1CCN(C(C)=O)CC1", "1,4-diacetylpiperazine"),
    ("CC(=O)N1CCC(N)CC1", "1-acetylpiperidin-4-amine"),
    ("OC1CCN(C=O)CC1", "1-formyl-4-hydroxypiperidine"),          # a ring-N carbaldehyde outranks the alcohol
])
def test_an_acyl_prefix_on_a_ring_n_is_flagged(smiles, name):
    assert _fails(smiles, name), name


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)n1ccc2ccccc21", "1-(1H-indol-1-yl)ethan-1-one"),
    ("CC(=O)N1CCCCC1", "1-(piperidin-1-yl)ethan-1-one"),                 #:33129
    ("NC(=O)N1CCCCC1", "piperidine-1-carboxamide"),
    ("CC(C)(C)OC(=O)N1CCCCC1", "tert-butyl piperidine-1-carboxylate"),
    ("O=CN1CCCCC1", "piperidine-1-carbaldehyde"),
    ("CC(=O)N1CCCC1=O", "1-acetylpyrrolidin-2-one"),             # a lactam: the ring ketone is expressed
    ("CC(=O)N1CCC(=O)CC1", "1-acetylpiperidin-4-one"),
    ("OC1CCN(CC1)C(N)=O", "4-hydroxypiperidine-1-carboxamide"),  # a ring-N urea is an amide
    ("CCOC(=O)N1CCC(O)CC1", "ethyl 4-hydroxypiperidine-1-carboxylate"),
])
def test_names_on_the_senior_class_pass(smiles, name):
    assert not _fails(smiles, name), name
