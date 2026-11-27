"""An N-acyl or N-sulfonyl imine is an N-substituted amide, the Blue Book): the 'acylimino' and
'sulfonylimino' prefixes are used only "When a group having preference for citation as a principal characteristic group is
present" ('methyl 4-[(methanesulfonyl)imino]cyclohexane-1-carboxylate (PIN)'). A name on an imine or ketone parent is then
correct but no PIN (validation/spelling/checks_parent.py, principal_class_check).
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_parent import principal_class_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("C1=CC=C(C=C1)CS(=O)(=O)/N=C/C2=CC=CC=C2", "(1E)-1-phenyl-N-(phenylmethanesulfonyl)methanimine"),
    ("CC1=CC=C(C=C1)S(=O)(=O)/N=C/2\\C=CC(=O)C=C2Cl", "(4E)-3-chloro-4-[(4-methylbenzene-1-sulfonyl)imino]cyclohexa-2,5-dien-1-one"),
    ("CC(=O)N=C(C)c1ccccc1", "N-acetyl-1-phenylethan-1-imine"),
])
def test_a_name_on_a_junior_parent_is_flagged(smiles, name):
    failure = principal_class_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-41", name


@pytest.mark.parametrize("smiles,name", [
    ("CS(=O)(=O)N=C1CCC(CC1)C(=O)OC", "methyl 4-[(methanesulfonyl)imino]cyclohexane-1-carboxylate"),   #:33054
    ("C1=CC=C(C=C1)CS(=O)(=O)/N=C/C2=CC=CC=C2", "N-[(E)-benzylidene]-1-phenylmethanesulfonamide"),
    ("CC(=O)N=C(C)c1ccccc1", "N-(1-phenylethylidene)acetamide"),
    ("CC(=O)N/N=C/c1ccccc1", "N'-benzylideneacetohydrazide"),                 # a hydrazone, not an N-acyl imine
])
def test_names_on_the_senior_class_pass(smiles, name):
    assert principal_class_check(Chem.MolFromSmiles(smiles), name) is None, name
