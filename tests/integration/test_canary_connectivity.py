"""
Connectivity-match canary: InChI connectivity layer matches, full InChI may differ
(e.g., stereo). These freeze the generated name for regression detection.

These compounds achieve InChI connectivity-layer match (correct atom connectivity)
but not full InChI match (stereo/charge may differ). They represent "almost correct"
naming that should not regress.

These names are frozen at v10.0 state. If v11.0 intentionally improves a name,
update the expected value.

Source: Phase 095 benchmark (500 ChEBI compounds, seed=123)
Tier: 2 of 3 (Tier 1 = RT-exact in test_canary_rt75.py, Tier 3 = name-stability)
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# 4 connectivity canary compounds: (SMILES, expected_name)
# Connectivity-layer InChI match but not full InChI match
# Phase 095 v10.0 canary expansion
# ---------------------------------------------------------------------------

CONNECTIVITY_CANARY = [
    (
        "CC(CC(=O)CC(C)C1C[C@H](O)[C@@]2(C)C3=C(C(=O)CC12C)C1(C)CC[C@H](O)C(C)(C)C1C[C@@H]3O)C(=O)O",
        "(3S,7S,14R,15S)-3,7,15,27-tetrahydroxy-4,4,14-trimethylcholest-8-en-11,23,27-trione",
    ),
    (
        "CC(C)CC[C@@H](O)[C@H]1C(=O)OC[C@@H]1CO",
        "(3S,4S)-3-(1-hydroxy-4-methylpentyl)-4-hydroxymethyloxolan-2-one",
    ),
    (
        "CSCCC(N)C(=O)Oc1ccc(CC(N)C(=O)O)cc1",
        "tyrosine methionineate",
    ),
    (
        "CC(C)C1=C(O)C(N)=C(/C=C/c2ccccc2)C(=O)C1=O",
        "4-amino-5-hydroxy-6-isopropyl-3-styrenylcyclohexa-3,5-diene-1,2-dione",
    ),
]

# Build test IDs from first 40 chars of SMILES (sanitized for pytest)
_CANARY_IDS = [
    smiles[:40].replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    for smiles, _ in CONNECTIVITY_CANARY
]


@pytest.mark.parametrize("smiles,expected_name", CONNECTIVITY_CANARY, ids=_CANARY_IDS)
def test_canary_connectivity(smiles, expected_name):
    """Connectivity canary test: verify name stability for connectivity-matching compounds (4 compounds).

    These compounds have correct InChI connectivity but may differ in stereo/charge.
    Any name change should be investigated before merging.
    """
    result = name_compound(smiles)
    assert result == expected_name, (
        f"CANARY REGRESSION (connectivity tier): {smiles}\n"
        f"  Expected: {expected_name}\n"
        f"  Got: {result}"
    )
