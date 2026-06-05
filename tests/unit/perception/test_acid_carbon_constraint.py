"""Phase 169.7 BBR-PERC / DEF-4 — acid SMARTS carbon-attachment constraint.

P-65.3: sulfonic/sulfinic/phosphonic acids are CARBON acids (the S/P bears a
C neighbour). The free inorganic oxoacids (sulfamic, phosphoric, sulfuric,
nitric) and nitrate ESTERS are named differently (P-42/P-67) and must NOT
false-match the carbon-acid classes. The 169.7 fix adds a recursive-env
``$(...)`` C-attachment constraint that preserves the match-tuple arity.
"""
import pytest
from rdkit import Chem

from orthonym.perception.functional_groups import detect_functional_groups as d


def _fgs(smi):
    return set(d(Chem.MolFromSmiles(smi)).keys())


@pytest.mark.unit
@pytest.mark.parametrize("smi,fg", [
    ("NS(=O)(=O)O", "sulfonic_acid"),     # sulfamic acid (S bears N, not C)
    ("OP(=O)(O)O", "phosphonic_acid"),    # phosphoric acid (P bears only O)
    ("CCO[N+](=O)[O-]", "nitro"),         # nitrate ester (N bears only O)
])
def test_inorganic_oxoacid_no_false_carbon_acid(smi, fg):
    assert fg not in _fgs(smi), f"{smi} must NOT match carbon-acid {fg} (P-65.3)"


@pytest.mark.unit
@pytest.mark.parametrize("smi,fg", [
    ("CCCS(=O)(=O)O", "sulfonic_acid"),       # propanesulfonic acid (C-attached)
    ("OC(=O)CCS(=O)(=O)O", "sulfonic_acid"),  # 3-sulfopropanoic acid (C-attached)
    ("CCS(=O)O", "sulfinic_acid"),            # ethanesulfinic acid
    ("CCCP(=O)(O)O", "phosphonic_acid"),      # propylphosphonic acid
    ("CC[N+](=O)[O-]", "nitro"),              # nitroethane (C-attached)
    ("c1ccccc1[N+](=O)[O-]", "nitro"),        # nitrobenzene (aromatic C)
])
def test_carbon_acids_still_match(smi, fg):
    assert fg in _fgs(smi), f"{smi} must STILL match C-attached {fg} (regression guard)"
