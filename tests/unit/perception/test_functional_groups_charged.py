"""Phase 169.7 BBR-PERC / DEF-1 — charged characteristic groups + shared accessor.

The neutral FG layer previously contained ZERO ionic patterns, so a charged
molecule reaching it (the D-1 mis-route exposure) silently lost its group.
169.7 adds the charged FG SMARTS (P-41 classes 4/6) and a single shared
``detect_features`` accessor. Every charged pattern requires a formal charge,
so NEUTRAL molecules never match (the false-positive guard).
"""
import pytest
from rdkit import Chem

from orthonym.perception.functional_groups import (
    detect_features,
    detect_functional_groups,
)


def _feat(smi):
    return set(detect_features(Chem.MolFromSmiles(smi)).keys())


@pytest.mark.unit
@pytest.mark.parametrize("smi,fg", [
    ("CC(=O)[O-]", "carboxylate"),
    ("CCC[NH3+]", "ammonium"),
    ("C[N+](C)(C)C", "ammonium"),
    ("CCCS(=O)(=O)[O-]", "sulfonate"),
    ("CCCP(=O)([O-])[O-]", "phosphonate"),
    ("CCC[S-]", "thiolate"),
    ("c1ccccc1[O-]", "phenolate"),
])
def test_charged_fg_perceived(smi, fg):
    assert fg in _feat(smi)


@pytest.mark.unit
@pytest.mark.parametrize("smi,fg", [
    ("CC(=O)O", "carboxylate"),   # neutral acid
    ("CCN", "ammonium"),          # neutral amine
    ("CCS", "thiolate"),          # neutral thiol
    ("c1ccccc1O", "phenolate"),   # neutral phenol
])
def test_charged_fg_negatives_on_neutral(smi, fg):
    assert fg not in _feat(smi)


@pytest.mark.unit
def test_detect_features_is_shared_accessor():
    # detect_features is the single named entry point; today equals the FG detector
    # (the orthogonal charge-SITE scan stays in ions.py).
    m = Chem.MolFromSmiles("CC(=O)[O-]")
    assert detect_features(m) == detect_functional_groups(m)


@pytest.mark.unit
def test_charged_route_unchanged():
    # The charged SMARTS must NOT disrupt the route_charged path (neutralize-first),
    # so the 169.6 charged names are preserved.
    from orthonym import Orthonym
    o = Orthonym()
    assert o.name("CC(=O)[O-]") == "acetate"
    assert o.name("CCC(=O)[O-].[K+]") == "potassium propanoate"
    assert o.name("C[N+](C)(C)CC(=O)[O-]") == "(trimethylazaniumyl)acetate"
