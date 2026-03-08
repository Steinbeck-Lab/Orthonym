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
RING-FG-04: Comprehensive FG audit (no silent FG dropping)
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


# === RING-FG-04: Comprehensive FG audit (no silent FG dropping) ===
# Every FG type on benzene must produce a non-"benzene" output.

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected", [
    # ---- Suffix FGs (principal groups) ----
    # Carboxylic acid (retained name: benzoic acid)
    ("OC(=O)c1ccccc1", "benzoic acid"),
    # Aldehyde (retained name: benzaldehyde)
    ("O=Cc1ccccc1", "benzaldehyde"),
    # Primary amide (retained name: benzamide)
    ("NC(=O)c1ccccc1", "benzamide"),
    # Nitrile (retained name: benzonitrile)
    ("N#Cc1ccccc1", "benzonitrile"),
    # Sulfonamide
    ("NS(=O)(=O)c1ccccc1", "benzenesulfonamide"),
    # Sulfonic acid
    ("OS(=O)(=O)c1ccccc1", "benzenesulfonic acid"),
    # Acid chloride (retained name: benzoyl chloride)
    ("O=C(Cl)c1ccccc1", "benzoyl chloride"),
    # N-methylbenzamide (secondary amide)
    ("CNC(=O)c1ccccc1", "N-methylbenzamide"),
    # N,N-dimethylbenzamide (tertiary amide)
    ("CN(C)C(=O)c1ccccc1", "N,N-dimethylbenzamide"),
    # Thiocarboxylic S-acid
    ("SC(=O)c1ccccc1", "benzenecarbothioic S-acid"),

    # ---- Prefix FGs (substituent form) ----
    # Phenol (retained name)
    ("Oc1ccccc1", "phenol"),
    # Aniline (retained name)
    ("Nc1ccccc1", "aniline"),
    # Nitrobenzene (retained name)
    ("[O-][N+](=O)c1ccccc1", "nitrobenzene"),
    # Anisole (retained name, P-34.1.1.4 PIN; systematic: methoxybenzene)
    ("COc1ccccc1", "anisole"),
    # Ethoxybenzene
    ("CCOc1ccccc1", "ethoxybenzene"),
    # Halogens
    ("Fc1ccccc1", "fluorobenzene"),
    ("Clc1ccccc1", "chlorobenzene"),
    ("Brc1ccccc1", "bromobenzene"),
    ("Ic1ccccc1", "iodobenzene"),
])
def test_benzene_fg_exact_names(smiles, expected):
    """Test exact IUPAC names for all FG types on benzene."""
    result = name_compound(smiles)
    assert result == expected, f"For {smiles}: got {result!r}, expected {expected!r}"


@pytest.mark.unit
@pytest.mark.parametrize("smiles, desc", [
    # Thiol (-SH) -> should produce sulfanylbenzene or similar, NOT "benzene"
    ("Sc1ccccc1", "thiol"),
    ("c1ccc(cc1)[SH]", "thiol explicit SH"),
    # Hydroperoxy (-OOH)
    ("OOc1ccccc1", "hydroperoxy"),
    # Trifluoromethyl (-CF3)
    ("c1ccc(cc1)C(F)(F)F", "trifluoromethyl"),
    # Disulfanyl (-S-SH)
    ("SSc1ccccc1", "disulfanyl"),
    # Ketone (acetophenone - retained name)
    ("CC(=O)c1ccccc1", "ketone"),
    # Ester (methyl benzoate - routed through ester handler)
    ("COC(=O)c1ccccc1", "ester"),
    # Vinyl (ethenyl) group
    ("C=Cc1ccccc1", "vinyl/ethenyl"),
    # Methylbenzene (toluene retained)
    ("Cc1ccccc1", "methyl (toluene)"),
])
def test_benzene_fg_not_dropped(smiles, desc):
    """Audit: every FG type on benzene produces a non-'benzene' output (RING-FG-04)."""
    result = name_compound(smiles)
    assert result is not None, f"FG dropped (None) for {smiles} ({desc})"
    assert result != "benzene", f"FG dropped for {smiles} ({desc}): got 'benzene'"


# === Edge cases: multiple FG types on benzene ===

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected", [
    # Suffix acid + prefix hydroxy: acid wins
    ("OC(=O)c1ccccc1O", "2-hydroxybenzoic acid"),
    # Suffix acid + methyl substituent
    ("OC(=O)c1ccc(C)cc1", "4-methylbenzoic acid"),
    # Acid + dimethyl substituents
    ("OC(=O)c1cc(C)cc(C)c1", "3,5-dimethylbenzoic acid"),
    # Thiol + methyl (both prefix)
    ("Sc1ccc(cc1)C", "1-methyl-4-sulfanylbenzene"),
    # Fluoro + trifluoromethyl
    ("Fc1ccc(cc1)C(F)(F)F", "1-fluoro-4-(trifluoromethyl)benzene"),
])
def test_benzene_mixed_fg(smiles, expected):
    """Test benzene with multiple different functional groups."""
    result = name_compound(smiles)
    assert result == expected, f"For {smiles}: got {result!r}, expected {expected!r}"
