"""Unit tests for compound class pre-routing infrastructure.

Tests classify_compound_class() which routes molecules to class-specific
handlers before the 33-handler cascade. The function returns a class label
(steroid, alkaloid, terpene, carbohydrate) or None for general routing.

Test classes:
- TestClassifySteroid: Steroid molecules classified as 'steroid'
- TestClassifyAlkaloid: Alkaloid molecules classified as 'alkaloid'
- TestClassifyTerpene: Terpene molecules classified as 'terpene'
- TestClassifyCarbohydrate: Carbohydrate molecules classified as 'carbohydrate'
- TestClassifyGeneral: Non-NP molecules return None
- TestSugarInMainCascade: Sugar detection wired into name_compound()
- TestExistingNPPreserved: Existing NP detection order preserved
"""

import pytest
from rdkit import Chem

from orthonym.namer import classify_compound_class
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mol(smiles: str):
    """Return RDKit Mol from SMILES, raising on invalid input."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol


def _canonical(smiles: str) -> str:
    """Return canonical SMILES."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)


# ---------------------------------------------------------------------------
# SMILES constants
# ---------------------------------------------------------------------------

# Cholesterol (steroid)
CHOLESTEROL_SMILES = (
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C"
    "[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
)

# Testosterone (steroid)
TESTOSTERONE_SMILES = (
    "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]43C)"
    "[C@@H]1CC[C@@H]2O"
)

# Progesterone (steroid)
PROGESTERONE_SMILES = "CC(=O)[C@H]1CC[C@@H]2[C@@H]1CC[C@H]1[C@@H]2CCC2=CC(=O)CC[C@@]21C"

# Morphine (alkaloid)
MORPHINE_SMILES = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"

# Codeine (alkaloid)
CODEINE_SMILES = "COc1ccc2c3c1O[C@H]1[C@@H](O)C=C[C@H]4[C@@H](C2)N(C)CC[C@@]341"

# Alpha-pinene (terpene -- detected via NP scaffold)
ALPHA_PINENE_SMILES = "CC1=CCC2CC1C2(C)C"

# Glucose (non-stereo -- carbohydrate via sugar lookup)
GLUCOSE_NONSTEREO_SMILES = "OCC1OC(O)C(O)C(O)C1O"

# Ethanol (general)
ETHANOL_SMILES = "CCO"

# Benzene (general)
BENZENE_SMILES = "c1ccccc1"

# Cyclohexane (general)
CYCLOHEXANE_SMILES = "C1CCCCC1"

# A pyranose with 3 OH groups (carbohydrate via SMARTS, not in sugar lookup)
# 2-deoxy sugar: oxane ring + 3 OH on ring carbons
DEOXY_SUGAR_SMILES = "OC1CCOC(O)C1O"


# ===========================================================================
# Test Class 1: Steroid classification
# ===========================================================================

@pytest.mark.unit
class TestClassifySteroid:
    """Steroid molecules should be classified as 'steroid'."""

    def test_classify_cholesterol(self):
        smi = CHOLESTEROL_SMILES
        mol = _mol(smi)
        assert classify_compound_class(mol, _canonical(smi)) == "steroid"

    def test_classify_testosterone(self):
        smi = TESTOSTERONE_SMILES
        mol = _mol(smi)
        assert classify_compound_class(mol, _canonical(smi)) == "steroid"

    def test_classify_progesterone(self):
        smi = PROGESTERONE_SMILES
        mol = _mol(smi)
        assert classify_compound_class(mol, _canonical(smi)) == "steroid"


# ===========================================================================
# Test Class 2: Alkaloid classification
# ===========================================================================

@pytest.mark.unit
class TestClassifyAlkaloid:
    """Alkaloid molecules should be classified as 'alkaloid'."""

    def test_classify_morphine(self):
        smi = MORPHINE_SMILES
        mol = _mol(smi)
        assert classify_compound_class(mol, _canonical(smi)) == "alkaloid"

    def test_classify_codeine(self):
        smi = CODEINE_SMILES
        mol = _mol(smi)
        assert classify_compound_class(mol, _canonical(smi)) == "alkaloid"


# ===========================================================================
# Test Class 3: Terpene classification
# ===========================================================================

@pytest.mark.unit
class TestClassifyTerpene:
    """Terpene molecules should be classified as 'terpene' via NP detection."""

    def test_classify_alpha_pinene(self):
        smi = ALPHA_PINENE_SMILES
        mol = _mol(smi)
        result = classify_compound_class(mol, _canonical(smi))
        assert result == "terpene"


# ===========================================================================
# Test Class 4: Carbohydrate classification
# ===========================================================================

@pytest.mark.unit
class TestClassifyCarbohydrate:
    """Carbohydrate molecules should be classified as 'carbohydrate'."""

    def test_classify_glucose_nonstereo_via_lookup(self):
        """Non-stereo glucose should be found via sugar lookup fallback."""
        smi = GLUCOSE_NONSTEREO_SMILES
        mol = _mol(smi)
        assert classify_compound_class(mol, _canonical(smi)) == "carbohydrate"

    def test_classify_pyranose_via_smarts(self):
        """Pyranose with 2+ OH groups detected via SMARTS pattern."""
        smi = DEOXY_SUGAR_SMILES
        mol = _mol(smi)
        result = classify_compound_class(mol, _canonical(smi))
        assert result == "carbohydrate"


# ===========================================================================
# Test Class 5: General compounds return None
# ===========================================================================

@pytest.mark.unit
class TestClassifyGeneral:
    """Non-NP molecules should return None for general routing."""

    def test_classify_ethanol_returns_none(self):
        smi = ETHANOL_SMILES
        mol = _mol(smi)
        assert classify_compound_class(mol, _canonical(smi)) is None

    def test_classify_benzene_returns_none(self):
        smi = BENZENE_SMILES
        mol = _mol(smi)
        assert classify_compound_class(mol, _canonical(smi)) is None

    def test_classify_cyclohexane_returns_none(self):
        smi = CYCLOHEXANE_SMILES
        mol = _mol(smi)
        assert classify_compound_class(mol, _canonical(smi)) is None


# ===========================================================================
# Test Class 6: Sugar detection in main cascade
# ===========================================================================

@pytest.mark.unit
class TestSugarInMainCascade:
    """Sugar detection wired into name_compound() returns sugar name."""

    def test_glucose_nonstereo_returns_sugar_name(self):
        """Non-stereo glucose should return a sugar name, not systematic oxane."""
        result = name_compound(GLUCOSE_NONSTEREO_SMILES)
        assert result is not None
        result_lower = result.lower()
        # Should contain 'pyranose' or 'glucose' -- NOT be a systematic name
        assert "pyranose" in result_lower or "glucose" in result_lower, (
            f"Expected sugar name for glucose, got: '{result}'"
        )


# ===========================================================================
# Test Class 7: Existing NP detection preserved
# ===========================================================================

@pytest.mark.unit
class TestExistingNPPreserved:
    """Existing NP detection order must be preserved -- no regressions."""

    def test_cholesterol_still_returns_cholesterol(self):
        result = name_compound(CHOLESTEROL_SMILES)
        assert result == "cholesterol"

    def test_morphine_still_returns_morphine(self):
        result = name_compound(MORPHINE_SMILES)
        assert result == "morphine"

    def test_codeine_still_returns_codeine(self):
        result = name_compound(CODEINE_SMILES)
        assert result == "codeine"

    def test_ethanol_still_returns_ethanol(self):
        result = name_compound(ETHANOL_SMILES)
        assert result == "ethanol"

    def test_benzene_still_returns_benzene(self):
        result = name_compound(BENZENE_SMILES)
        assert result == "benzene"
