"""Wave-2 completion Tier 2: amide-N prefix vocabulary (WAVE2-COMPLETION-PLAN).

Currently: thiourea -> carbamothioylamino. All OPSIN-RT probed.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.functional_groups import detect_functional_groups

pytestmark = pytest.mark.unit


def _name(smiles):
    return name_compound(Chem.CanonSmiles(smiles))


def test_thiourea_perceived():
    fgs = detect_functional_groups(Chem.MolFromSmiles("NC(=S)NCCC(=O)O"))
    assert "thiourea" in fgs
    # The terminal-N amine perception is suppressed by the thiourea mask.
    assert "primary_amine" not in fgs


@pytest.mark.parametrize("smiles,expected", [
    ("NC(=S)NCCC(=O)O", "3-(carbamothioylamino)propanoic acid"),
])
def test_carbamothioylamino_prefix(smiles, expected):
    assert _name(smiles) == expected


def test_urea_carbamoylamino_unchanged():
    # Protect: the pre-existing urea prefix is byte-identical.
    assert _name("NC(=O)NCCC(=O)O") == "3-(carbamoylamino)propanoic acid"


def test_sulfamoyl_unchanged():
    # Protect: C-attached H2N-SO2- keeps the retained sulfamoyl prefix.
    assert _name("O=S(=O)(N)CCC(=O)O") == "3-sulfamoylpropanoic acid"
