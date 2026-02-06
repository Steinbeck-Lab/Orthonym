"""
Tests for benzene ring-attached suffix functional group naming.

IUPAC 2013 Rules:
- P-65.1.2: Principal group on ring uses suffix form
- P-66.1.1.1: Amides of benzoic acid -> benzamide (retained)
- P-66.4.1: Sulfonamides use -sulfonamide suffix
- Dicarboxylic acids on benzene: benzene-1,2-dicarboxylic acid

RING-FG-01: Amides (carboxamide suffix)
RING-FG-02: Sulfonamides (sulfonamide suffix)
RING-FG-03: Dicarboxylic acids and dialdehydes (suffix form)
"""

import pytest
from orthonym import name_compound


# === RING-FG-01: Ring-attached amides ===

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected", [
    # Unsubstituted benzamide - retained name
    ("NC(=O)c1ccccc1", "benzamide"),
    # Substituted benzamide - systematic suffix naming
    ("NC(=O)c1ccc(C)cc1", "4-methylbenzamide"),
    # N-substituted amide
    ("CNC(=O)c1ccccc1", "N-methylbenzamide"),
    # Di-amide suffix
    ("NC(=O)c1ccc(cc1)C(N)=O", "benzene-1,4-dicarboxamide"),
])
def test_benzene_amide_suffix(smiles, expected):
    """Test amide suffix naming on benzene ring."""
    result = name_compound(smiles)
    assert result == expected, f"For {smiles}: got {result!r}, expected {expected!r}"


# === RING-FG-02: Ring-attached sulfonamides ===

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected", [
    # Basic sulfonamide
    ("NS(=O)(=O)c1ccccc1", "benzenesulfonamide"),
    # Substituted sulfonamide
    ("NS(=O)(=O)c1ccc(C)cc1", "4-methylbenzenesulfonamide"),
])
def test_benzene_sulfonamide_suffix(smiles, expected):
    """Test sulfonamide suffix naming on benzene ring."""
    result = name_compound(smiles)
    assert result == expected, f"For {smiles}: got {result!r}, expected {expected!r}"


# === RING-FG-03: Dicarboxylic acids and dialdehydes as suffix ===

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected", [
    # Phthalic acid
    ("OC(=O)c1ccccc1C(=O)O", "benzene-1,2-dicarboxylic acid"),
    # Terephthalic acid
    ("OC(=O)c1ccc(cc1)C(=O)O", "benzene-1,4-dicarboxylic acid"),
    # Hydroxybenzoic acid (suffix acid + prefix hydroxy)
    ("OC(=O)c1ccccc1O", "2-hydroxybenzoic acid"),
    # Dialdehyde
    ("O=Cc1ccc(cc1)C=O", "benzene-1,4-dicarbaldehyde"),
])
def test_benzene_dicarboxylic_and_dialdehyde_suffix(smiles, expected):
    """Test dicarboxylic acid and dialdehyde suffix naming on benzene ring."""
    result = name_compound(smiles)
    assert result == expected, f"For {smiles}: got {result!r}, expected {expected!r}"
