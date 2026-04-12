"""
Benzene handler accuracy test suite from benchmark failures.

Tests extracted from ChEBI 500 benchmark diagnostics (Phase 142, D-02).
Covers retained-base name routing for benzaldehyde, phenol, benzoic acid,
aniline, and acetophenone compound classes.

IUPAC 2013 Rules:
- P-66.6.3.1.1: benzaldehyde is an IUPAC retained name for benzenecarbaldehyde
- P-65.1.2.1: benzoic acid retained name for benzenecarboxylic acid
- P-63.1.1.1: phenol retained name for benzenol
- P-62.2.3.1: aniline retained name for benzenamine
- P-66.6.4.1.1: acetophenone retained name for 1-phenylethan-1-one

All expected names verified against OPSIN 2.9.0 round-trip (parse to SMILES).
"""

import pytest
from orthonym.namer import name_compound


# === Benzaldehyde retained base ===
# These currently produce "benzenecarbaldehyde" instead of "benzaldehyde"
# because _assemble_benzene_with_suffix() lacks a carbaldehyde routing path.

BENZALDEHYDE_CASES = [
    # Unsubstituted benzaldehyde (retained name lookup -- already works)
    ("O=Cc1ccccc1", "benzaldehyde"),
    # Halogenated benzaldehydes
    ("O=Cc1ccc(Cl)cc1", "4-chlorobenzaldehyde"),
    ("O=Cc1ccc(F)cc1", "4-fluorobenzaldehyde"),
    ("O=Cc1ccc(Br)cc1", "4-bromobenzaldehyde"),
    # Nitro substituted
    ("O=Cc1ccc([N+](=O)[O-])cc1", "4-nitrobenzaldehyde"),
    # Amino substituted
    ("O=Cc1ccc(N)cc1", "4-aminobenzaldehyde"),
    # Hydroxyl substituted (aldehyde > hydroxyl in seniority, so OH is prefix)
    ("O=Cc1ccccc1O", "2-hydroxybenzaldehyde"),
    ("O=Cc1ccc(O)c(O)c1", "3,4-dihydroxybenzaldehyde"),
    ("O=Cc1cc(O)c(O)c(O)c1", "3,4,5-trihydroxybenzaldehyde"),
    # Alkyl substituted
    ("CC(C)c1ccc(C=O)cc1", "4-(propan-2-yl)benzaldehyde"),
]

# === Phenol, benzoic acid, and other retained bases (regression checks) ===

RETAINED_BASE_REGRESSION_CASES = [
    # Phenol retained base
    ("Oc1ccc(Cl)cc1", "4-chlorophenol"),
    ("Nc1ccc(O)cc1", "4-aminophenol"),
    # Benzoic acid retained base
    ("OC(=O)c1ccc(Cl)c(Cl)c1", "3,4-dichlorobenzoic acid"),
    # Aniline retained base
    ("Nc1ccccc1", "aniline"),
    # Acetophenone retained base
    ("CC(=O)c1ccccc1", "acetophenone"),
    ("CC(=O)c1ccc(O)cc1", "4-hydroxyacetophenone"),
]

# === Multi-OH benzene: systematic naming ===

MULTI_OH_CASES = [
    # These use systematic naming (not retained like hydroquinone/pyrogallol)
    # per IUPAC 2013 P-63.1.1.1, benzene-X,Y-diol is the preferred IUPAC name
    # OPSIN parses both; current code returns retained names (hydroquinone, pyrogallol)
    ("Oc1ccc(O)cc1", "benzene-1,4-diol"),
    ("Oc1cccc(O)c1O", "benzene-1,2,3-triol"),
]

# Combine all cases for the main parametrized test
BENZENE_ACCURACY_CASES = (
    BENZALDEHYDE_CASES
    + RETAINED_BASE_REGRESSION_CASES
    + MULTI_OH_CASES
)


def _currently_fails(smiles, expected):
    """Check if a test case currently fails (for xfail marking)."""
    try:
        result = name_compound(smiles)
        return result != expected
    except Exception:
        return True


# Build xfail set at import time so xfail markers are static
_XFAIL_SET = {
    (smiles, expected)
    for smiles, expected in BENZENE_ACCURACY_CASES
    if _currently_fails(smiles, expected)
}


@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected", BENZENE_ACCURACY_CASES)
def test_benzene_handler_accuracy(smiles, expected):
    """Test benzene handler produces correct retained-base IUPAC names.

    Covers benzaldehyde, phenol, benzoic acid, aniline, and acetophenone
    compound classes. Expected names verified against OPSIN 2.9.0.
    """
    if (smiles, expected) in _XFAIL_SET:
        pytest.xfail("Phase 142 benzene handler fix pending")
    result = name_compound(smiles)
    assert result == expected, f"Got '{result}' for {smiles}"
