"""
Unit tests for N-substituent naming in amides.

Tests both the fix for cycloalkyl, branched, heterocyclic, and substituted
aryl N-substituents (which were incorrectly named as linear alkyls) and
regression protection for existing simple N-substituent naming.

Phase 106-01: FIX-11 from Phase 102.5 audit.
"""

import pytest

from orthonym import name_compound


# ---- New fixes: complex N-substituents (must contain expected substring) ----

COMPLEX_N_SUBSTITUENT_CASES = [
    # (SMILES, expected_substring, description)
    ("CC(=O)NC1CCCC1", "cyclopentyl", "cyclopentyl ring substituent"),
    ("CC(=O)NC1CCCCC1", "cyclohexyl", "cyclohexyl ring substituent"),
    ("CC(=O)NC(C)C", "isopropyl", "branched substituent (isopropyl retained name)"),
    ("CC(=O)NC(C)(C)C", "tert-butyl", "branched substituent (tert-butyl)"),
    ("CC(=O)Nc1ccncc1", "pyridin", "heterocyclic substituent"),
    ("CC(=O)Nc1ccc(C)cc1", "tolu", "substituted aryl substituent (tolyl/toluenyl)"),
]


@pytest.mark.parametrize(
    "smiles, expected_substring, description",
    COMPLEX_N_SUBSTITUENT_CASES,
    ids=[c[2] for c in COMPLEX_N_SUBSTITUENT_CASES],
)
def test_complex_n_substituent_contains(smiles, expected_substring, description):
    """Complex N-substituents must be named correctly via universal pipeline."""
    name = name_compound(smiles)
    assert expected_substring in name.lower(), (
        f"Expected '{expected_substring}' in name for {smiles}, got '{name}'"
    )


# ---- Regression: simple N-substituents that already work ----

SIMPLE_N_SUBSTITUENT_CASES = [
    # (SMILES, expected_name)
    ("CC(=O)NC", "N-methylacetamide"),
    ("CC(=O)NCC", "N-ethylacetamide"),
    ("CC(=O)Nc1ccccc1", "N-phenylacetamide"),
    ("CC(=O)NCc1ccccc1", "N-benzylacetamide"),
    ("CN(C)C=O", "N,N-dimethylformamide"),
]


@pytest.mark.parametrize(
    "smiles, expected_name",
    SIMPLE_N_SUBSTITUENT_CASES,
    ids=[f"regression-{n}" for n in ["methyl", "ethyl", "phenyl", "benzyl", "dimethyl"]],
)
def test_simple_n_substituent_regression(smiles, expected_name):
    """Simple N-substituents must still produce exact expected names."""
    name = name_compound(smiles)
    assert name == expected_name, (
        f"Regression: expected '{expected_name}' for {smiles}, got '{name}'"
    )


# ---- Mixed N-substituents (alphabetization) ----

def test_mixed_n_substituents_alphabetized():
    """Mixed N-substituents must be alphabetized per IUPAC P-14.5."""
    name = name_compound("CC(=O)N(CC)c1ccccc1")
    assert name == "N-ethyl-N-phenylacetamide", (
        f"Expected 'N-ethyl-N-phenylacetamide', got '{name}'"
    )
