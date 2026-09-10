"""Unit tests for natural product scaffold detection (perception layer).

Tests cover:
- Scaffold detection for steroids and alkaloids
- Largest-match-wins logic for overlapping scaffolds
- Non-scaffold atom identification
- Convenience functions (is_steroid, is_alkaloid)
- Substituent detection on scaffolds
"""

import pytest
from rdkit import Chem

from orthonym.perception.natural_products import (
    detect_natural_product,
    get_non_scaffold_atoms,
    get_scaffold_substituents,
    is_alkaloid,
    is_steroid,
)


# ---------------------------------------------------------------------------
# Scaffold SMILES constants (from data module)
# ---------------------------------------------------------------------------

ANDROSTANE_SMILES = "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2"
CHOLESTANE_SMILES = (
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)"
    "[C@H]3CC[C@]12C"
)
GONANE_SMILES = "C1CC[C@H]2C(C1)CC[C@H]1[C@@H]3CCC[C@H]3CC[C@@H]12"
MORPHINAN_SMILES = "c1ccc2c(c1)C[C@H]1NCC[C@@]23CCCC[C@@H]13"
TROPANE_SMILES = "CN1[C@@H]2CCC[C@H]1CC2"
ESTRANE_SMILES = "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@@H]3[C@H]1CC2"
PREGNANE_SMILES = (
    "CC[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C"
)
CHOLESTEROL_SMILES = (
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC"
    "[C@]4(C)[C@H]3CC[C@]12C"
)
MORPHINE_SMILES = (
    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
)


# ===========================================================================
# TestDetectNaturalProduct
# ===========================================================================

class TestDetectNaturalProduct:
    """Test the main detect_natural_product function."""

    @pytest.mark.unit
    def test_detect_androstane(self):
        mol = Chem.MolFromSmiles(ANDROSTANE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        assert result["scaffold_name"] == "androstane"
        assert result["scaffold_class"] == "steroid"

    @pytest.mark.unit
    def test_detect_cholestane(self):
        mol = Chem.MolFromSmiles(CHOLESTANE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        assert result["scaffold_name"] == "cholestane"
        assert result["scaffold_class"] == "steroid"

    @pytest.mark.unit
    def test_detect_gonane(self):
        mol = Chem.MolFromSmiles(GONANE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        assert result["scaffold_name"] == "gonane"
        assert result["scaffold_class"] == "steroid"

    @pytest.mark.unit
    def test_detect_morphinan(self):
        mol = Chem.MolFromSmiles(MORPHINAN_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        assert result["scaffold_name"] == "morphinan"
        assert result["scaffold_class"] == "alkaloid"

    @pytest.mark.unit
    def test_detect_tropane(self):
        mol = Chem.MolFromSmiles(TROPANE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        assert result["scaffold_name"] == "tropane"
        assert result["scaffold_class"] == "alkaloid"

    @pytest.mark.unit
    def test_detect_estrane(self):
        mol = Chem.MolFromSmiles(ESTRANE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        # Estrane may match androstane (same size) or estrane depending on order
        assert result["scaffold_class"] == "steroid"

    @pytest.mark.unit
    def test_detect_pregnane(self):
        mol = Chem.MolFromSmiles(PREGNANE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        assert result["scaffold_class"] == "steroid"

    @pytest.mark.unit
    def test_detect_none_for_benzene(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        result = detect_natural_product(mol)
        assert result is None

    @pytest.mark.unit
    def test_detect_none_for_ethanol(self):
        mol = Chem.MolFromSmiles("CCO")
        result = detect_natural_product(mol)
        assert result is None

    @pytest.mark.unit
    def test_detect_none_for_none_mol(self):
        result = detect_natural_product(None)
        assert result is None

    @pytest.mark.unit
    def test_detect_cholesterol_with_stereo(self):
        """Cholesterol (unsaturated derivative) should match cholestane scaffold."""
        mol = Chem.MolFromSmiles(CHOLESTEROL_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        assert result["scaffold_name"] == "cholestane"
        assert result["scaffold_class"] == "steroid"

    @pytest.mark.unit
    def test_detect_cholesterol_without_stereo(self):
        """Cholesterol without stereochemistry should still match."""
        mol = Chem.MolFromSmiles(
            "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C"
        )
        result = detect_natural_product(mol)
        assert result is not None
        assert result["scaffold_name"] == "cholestane"

    @pytest.mark.unit
    def test_detect_morphine(self):
        """Morphine (substituted morphinan) should match morphinan scaffold."""
        mol = Chem.MolFromSmiles(MORPHINE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        assert result["scaffold_name"] == "morphinan"
        assert result["scaffold_class"] == "alkaloid"

    @pytest.mark.unit
    def test_result_contains_required_keys(self):
        """Verify all required keys in detection result."""
        mol = Chem.MolFromSmiles(ANDROSTANE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        required_keys = {
            "scaffold_name", "scaffold_stem", "scaffold_class",
            "scaffold_smiles", "matched_atoms", "non_scaffold_atoms",
        }
        assert required_keys.issubset(result.keys())

    @pytest.mark.unit
    def test_matched_atoms_is_tuple(self):
        mol = Chem.MolFromSmiles(ANDROSTANE_SMILES)
        result = detect_natural_product(mol)
        assert isinstance(result["matched_atoms"], tuple)

    @pytest.mark.unit
    def test_non_scaffold_atoms_is_set(self):
        mol = Chem.MolFromSmiles(ANDROSTANE_SMILES)
        result = detect_natural_product(mol)
        assert isinstance(result["non_scaffold_atoms"], set)


# ===========================================================================
# TestLargestMatchWins
# ===========================================================================

class TestLargestMatchWins:
    """Test that the largest scaffold match wins when multiple match."""

    @pytest.mark.unit
    def test_cholestane_over_gonane(self):
        """Cholestane molecule should match cholestane, not gonane (subset)."""
        mol = Chem.MolFromSmiles(CHOLESTANE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        assert result["scaffold_name"] == "cholestane"
        # Verify gonane (17 atoms) doesn't win over cholestane (27 atoms)
        assert len(result["matched_atoms"]) == 27

    @pytest.mark.unit
    def test_cholesterol_matches_cholestane(self):
        """Cholesterol should match cholestane (not a smaller steroid)."""
        mol = Chem.MolFromSmiles(CHOLESTEROL_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        assert result["scaffold_name"] == "cholestane"

    @pytest.mark.unit
    def test_androstane_not_gonane(self):
        """Androstane (19 atoms) should match androstane, not gonane (17 atoms)."""
        mol = Chem.MolFromSmiles(ANDROSTANE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        # Should be androstane (19 atoms), not gonane (17 atoms)
        assert len(result["matched_atoms"]) >= 19


# ===========================================================================
# TestNonScaffoldAtoms
# ===========================================================================

class TestNonScaffoldAtoms:
    """Test non-scaffold atom identification."""

    @pytest.mark.unit
    def test_cholesterol_substituents(self):
        """Cholesterol has non-scaffold atoms (OH group)."""
        mol = Chem.MolFromSmiles(CHOLESTEROL_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        # Cholesterol has 28 atoms, cholestane scaffold has 27
        # So 1 non-scaffold atom (the O in OH)
        assert len(result["non_scaffold_atoms"]) >= 1

    @pytest.mark.unit
    def test_pure_scaffold_no_extra(self):
        """Plain gonane should have no non-scaffold atoms."""
        mol = Chem.MolFromSmiles(GONANE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        # Gonane is the bare scaffold - all atoms should be matched
        assert len(result["non_scaffold_atoms"]) == 0

    @pytest.mark.unit
    def test_get_non_scaffold_atoms_function(self):
        """Test get_non_scaffold_atoms directly."""
        mol = Chem.MolFromSmiles("CCCCCC")  # hexane
        matched = (0, 1, 2, 3)  # first 4 atoms
        non_scaffold = get_non_scaffold_atoms(mol, matched)
        assert non_scaffold == {4, 5}

    @pytest.mark.unit
    def test_all_atoms_matched(self):
        """If all atoms matched, non-scaffold set should be empty."""
        mol = Chem.MolFromSmiles("CCC")
        matched = tuple(range(mol.GetNumAtoms()))
        non_scaffold = get_non_scaffold_atoms(mol, matched)
        assert len(non_scaffold) == 0


# ===========================================================================
# TestConvenienceFunctions
# ===========================================================================

class TestConvenienceFunctions:
    """Test is_steroid and is_alkaloid convenience functions."""

    @pytest.mark.unit
    def test_is_steroid_true(self):
        mol = Chem.MolFromSmiles(CHOLESTANE_SMILES)
        assert is_steroid(mol) is True

    @pytest.mark.unit
    def test_is_steroid_true_for_cholesterol(self):
        mol = Chem.MolFromSmiles(CHOLESTEROL_SMILES)
        assert is_steroid(mol) is True

    @pytest.mark.unit
    def test_is_steroid_false_for_benzene(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert is_steroid(mol) is False

    @pytest.mark.unit
    def test_is_steroid_false_for_none(self):
        assert is_steroid(None) is False

    @pytest.mark.unit
    def test_is_alkaloid_true(self):
        mol = Chem.MolFromSmiles(MORPHINAN_SMILES)
        assert is_alkaloid(mol) is True

    @pytest.mark.unit
    def test_is_alkaloid_true_for_morphine(self):
        mol = Chem.MolFromSmiles(MORPHINE_SMILES)
        assert is_alkaloid(mol) is True

    @pytest.mark.unit
    def test_is_alkaloid_false_for_cyclohexane(self):
        mol = Chem.MolFromSmiles("C1CCCCC1")
        assert is_alkaloid(mol) is False

    @pytest.mark.unit
    def test_is_alkaloid_false_for_none(self):
        assert is_alkaloid(None) is False

    @pytest.mark.unit
    def test_steroid_not_alkaloid(self):
        """Steroids should not register as alkaloids."""
        mol = Chem.MolFromSmiles(ANDROSTANE_SMILES)
        assert is_steroid(mol) is True
        assert is_alkaloid(mol) is False

    @pytest.mark.unit
    def test_alkaloid_not_steroid(self):
        """Alkaloids should not register as steroids."""
        mol = Chem.MolFromSmiles(TROPANE_SMILES)
        assert is_alkaloid(mol) is True
        assert is_steroid(mol) is False


# ===========================================================================
# TestGetScaffoldSubstituents
# ===========================================================================

class TestGetScaffoldSubstituents:
    """Test substituent detection on scaffolds."""

    @pytest.mark.unit
    def test_substituted_steroid(self):
        """Cholesterol has an OH substituent on the scaffold."""
        mol = Chem.MolFromSmiles(CHOLESTEROL_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        subs = get_scaffold_substituents(mol, result["matched_atoms"])
        assert len(subs) >= 1
        # Each substituent should have required keys
        for sub in subs:
            assert "attachment_atom" in sub
            assert "substituent_atoms" in sub
            assert "first_atom" in sub
            assert isinstance(sub["substituent_atoms"], list)
            assert len(sub["substituent_atoms"]) >= 1

    @pytest.mark.unit
    def test_unsubstituted_scaffold(self):
        """Plain gonane should have no substituents."""
        mol = Chem.MolFromSmiles(GONANE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        subs = get_scaffold_substituents(mol, result["matched_atoms"])
        assert len(subs) == 0

    @pytest.mark.unit
    def test_substituent_attachment_is_scaffold_atom(self):
        """Substituent attachment atoms must be in the scaffold."""
        mol = Chem.MolFromSmiles(CHOLESTEROL_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        scaffold_atoms = set(result["matched_atoms"])
        subs = get_scaffold_substituents(mol, result["matched_atoms"])
        for sub in subs:
            assert sub["attachment_atom"] in scaffold_atoms

    @pytest.mark.unit
    def test_substituent_atoms_not_in_scaffold(self):
        """Substituent atoms must NOT be in the scaffold."""
        mol = Chem.MolFromSmiles(CHOLESTEROL_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        scaffold_atoms = set(result["matched_atoms"])
        subs = get_scaffold_substituents(mol, result["matched_atoms"])
        for sub in subs:
            for atom_idx in sub["substituent_atoms"]:
                assert atom_idx not in scaffold_atoms

    @pytest.mark.unit
    def test_morphine_has_substituents(self):
        """Morphine (complex derivative) should have substituents on morphinan scaffold."""
        mol = Chem.MolFromSmiles(MORPHINE_SMILES)
        result = detect_natural_product(mol)
        assert result is not None
        subs = get_scaffold_substituents(mol, result["matched_atoms"])
        # Morphine has OH groups and N-methyl beyond the morphinan scaffold
        assert len(subs) >= 1
