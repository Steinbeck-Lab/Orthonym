"""An ester's acid word describes the input (pre-existing-failures plan, Task 4
continuation, 2026-09-25; the tropisetron chain, TRIAGE row 36 / canary call
182). With the ring-parent defect (test_ring_ester_parent_monocycle.py) it built
'2-(ethan-1-yl)-1-methyl-4-(nonanoyloxy)piperidine' for tropisetron
(8-methyl-8-azabicyclo[3.2.1]octan-3-yl 1H-indole-3-carboxylate):

`rules/esters.py::get_acid_fragment_name` fell back to a carbon count for any
acid its ring naming declined, 'nonanoic' for 1H-indole-3-carboxylic acid and
'decanoic' for 4-cyclohexylbutanoic acid. The 'oic acid' stem of a carbon
count is the chain form only, the Blue Book, "... chain are
named by replacing the final 'e' of the name of the corresponding hydrocarbon
by the suffix 'oic acid'"); a ring acid takes 'carboxylic acid' on its ring
,:29876). The function now returns '' ("cannot name") and every
caller declines.
"""
import pytest
from rdkit import Chem

from orthonym.rules.esters import _bfs_fragment, get_acid_fragment_name

pytestmark = pytest.mark.unit


def _acid_word(smiles):
    # every SMILES below is written R'-O(1)-C(2)(=O)-..., carbonyl carbon at 2
    mol = Chem.MolFromSmiles(smiles)
    return get_acid_fragment_name(mol, _bfs_fragment(mol, 2, exclude_atom=1))


@pytest.mark.parametrize("smiles,old", [
    ("COC(=O)c1c[nH]c2ccccc12", "nonanoic"),     # 1H-indole-3-carboxylic acid
    ("COC(=O)CCCC1CCCCC1", "decanoic"),          # 4-cyclohexylbutanoic acid
])
def test_no_carbon_count_for_a_ring_bearing_acid(smiles, old):
    assert _acid_word(smiles) == ""


@pytest.mark.parametrize("smiles,expected", [
    ("COC(=O)CCC", "butanoic"),
    ("COC(=O)c1ccccc1", "benzoic"),
])
def test_chain_and_benzoic_acid_words_unchanged(smiles, expected):
    assert _acid_word(smiles) == expected
