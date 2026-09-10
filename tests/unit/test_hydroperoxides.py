"""
Tests for hydroperoxide naming.

Hydroperoxides (R-OOH) use the suffix "-peroxol" as principal group
and the prefix "hydroperoxy" when subordinate.

IUPAC 2013 Blue Book:
-: Hydroperoxides named substitutively with -peroxol suffix
-: Hydroperoxy prefix when subordinate

Examples:
- COO -> methan-1-peroxol (methyl hydroperoxide)
- CCOO -> ethan-1-peroxol (ethyl hydroperoxide)
"""

import pytest
from orthonym import name_compound
from orthonym.rules.seniority import get_suffix, get_prefix
from orthonym.perception.functional_groups import detect_functional_groups
from rdkit import Chem


# ============================================================================
# Seniority Entry Tests
# ============================================================================

class TestHydroperoxideSeniority:
    """Test hydroperoxide seniority entries in SUFFIX_FORMS and PREFIX_FORMS."""

    @pytest.mark.unit
    def test_suffix_chain(self):
        """Hydroperoxide chain suffix is 'peroxol'."""
        assert get_suffix("hydroperoxide") == "peroxol"

    @pytest.mark.unit
    def test_suffix_ring(self):
        """Hydroperoxide ring suffix is 'peroxol'."""
        assert get_suffix("hydroperoxide", is_ring=True) == "peroxol"

    @pytest.mark.unit
    def test_prefix(self):
        """Hydroperoxide prefix is 'hydroperoxy'."""
        assert get_prefix("hydroperoxide") == "hydroperoxy"


# ============================================================================
# SMARTS Detection Tests
# ============================================================================

class TestHydroperoxideDetection:
    """Test SMARTS-based hydroperoxide detection."""

    @pytest.mark.unit
    def test_methyl_hydroperoxide_detected(self):
        """COO (methyl hydroperoxide) is detected as hydroperoxide."""
        mol = Chem.MolFromSmiles("COO")
        groups = detect_functional_groups(mol)
        assert "hydroperoxide" in groups
        assert len(groups["hydroperoxide"]) == 1

    @pytest.mark.unit
    def test_ethyl_hydroperoxide_detected(self):
        """CCOO (ethyl hydroperoxide) is detected as hydroperoxide."""
        mol = Chem.MolFromSmiles("CCOO")
        groups = detect_functional_groups(mol)
        assert "hydroperoxide" in groups
        assert len(groups["hydroperoxide"]) == 1

    @pytest.mark.unit
    def test_propyl_hydroperoxide_detected(self):
        """CCCOO (propyl hydroperoxide) is detected as hydroperoxide."""
        mol = Chem.MolFromSmiles("CCCOO")
        groups = detect_functional_groups(mol)
        assert "hydroperoxide" in groups
        assert len(groups["hydroperoxide"]) == 1


# ============================================================================
# End-to-End Naming Tests
# ============================================================================

class TestHydroperoxideNaming:
    """Test end-to-end hydroperoxide naming."""

    @pytest.mark.unit
    def test_methyl_hydroperoxide(self):
        """COO -> name contains 'peroxol'."""
        result = name_compound("COO")
        assert "peroxol" in result

    @pytest.mark.unit
    def test_ethyl_hydroperoxide(self):
        """CCOO -> name contains 'peroxol'."""
        result = name_compound("CCOO")
        assert "peroxol" in result

    @pytest.mark.unit
    def test_propyl_hydroperoxide(self):
        """CCCOO -> name contains 'peroxol'."""
        result = name_compound("CCCOO")
        assert "peroxol" in result

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected_substr", [
        ("COO", "peroxol"),
        ("CCOO", "peroxol"),
        ("CCCOO", "peroxol"),
    ])
    def test_hydroperoxide_suffix_present(self, smiles, expected_substr):
        """All simple hydroperoxides include 'peroxol' suffix."""
        result = name_compound(smiles)
        assert expected_substr in result, f"Expected '{expected_substr}' in '{result}'"
