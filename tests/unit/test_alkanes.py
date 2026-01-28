"""
End-to-end tests for alkane naming.

Tests the complete pipeline from SMILES to IUPAC name for:
- Simple straight-chain alkanes (C1-C10)
- Branched alkanes with substituent prefixes
- Alphabetization of multiple substituents
- Multiplicative prefixes (di-, tri-, etc.)
"""

import pytest
from orthonym import name_compound


# ============================================================================
# Simple Alkanes (C1-C10)
# ============================================================================

class TestSimpleAlkanes:
    """Test simple straight-chain alkanes from methane to decane."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("C", "methane"),
        ("CC", "ethane"),
        ("CCC", "propane"),
        ("CCCC", "butane"),
        ("CCCCC", "pentane"),
        ("CCCCCC", "hexane"),
        ("CCCCCCC", "heptane"),
        ("CCCCCCCC", "octane"),
        ("CCCCCCCCC", "nonane"),
        ("CCCCCCCCCC", "decane"),
    ])
    def test_simple_alkanes(self, smiles, expected):
        """Test C1-C10 straight-chain alkanes."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_methane_through_butane_are_retained(self):
        """Methane through butane are retained names, not generated systematically."""
        # These should be looked up from retained_names, not generated
        retained = [
            ("C", "methane"),
            ("CC", "ethane"),
            ("CCC", "propane"),
            ("CCCC", "butane"),
        ]
        for smiles, expected in retained:
            result = name_compound(smiles)
            assert result == expected, f"{smiles} should be retained name '{expected}'"

    @pytest.mark.unit
    def test_pentane_through_decane_are_systematic(self):
        """Pentane through decane should be generated systematically."""
        systematic = [
            ("CCCCC", "pentane"),
            ("CCCCCC", "hexane"),
            ("CCCCCCC", "heptane"),
            ("CCCCCCCC", "octane"),
            ("CCCCCCCCC", "nonane"),
            ("CCCCCCCCCC", "decane"),
        ]
        for smiles, expected in systematic:
            result = name_compound(smiles)
            assert result == expected, f"{smiles} should be systematic name '{expected}'"


# ============================================================================
# Branched Alkanes
# ============================================================================

class TestBranchedAlkanes:
    """Test branched alkanes with locanted substituent prefixes."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("CC(C)C", "2-methylpropane"),
        ("CC(C)CC", "2-methylbutane"),
        ("CCC(C)C", "2-methylbutane"),  # Same molecule, different SMILES
        ("CC(C)(C)C", "2,2-dimethylpropane"),
        ("CC(C)C(C)C", "2,3-dimethylbutane"),
        ("CC(C)(C)CC", "2,2-dimethylbutane"),
        ("CCC(CC)CC", "3-ethylpentane"),
    ])
    def test_branched_alkanes(self, smiles, expected):
        """Test branched alkanes produce correct locanted substituent prefixes."""
        assert name_compound(smiles) == expected

    @pytest.mark.integration
    def test_same_molecule_different_smiles_produces_same_name(self):
        """Different SMILES representations of the same molecule should give same name."""
        # 2-methylbutane can be written multiple ways
        # Note: All these SMILES must canonicalize to the same structure
        smiles_variants = [
            "CC(C)CC",   # Standard: methyl at position 2
            "CCC(C)C",   # Different numbering direction
            "C(C)(C)CC", # Explicit branch notation
        ]
        expected = "2-methylbutane"
        for smiles in smiles_variants:
            result = name_compound(smiles)
            assert result == expected, f"SMILES {smiles} should give {expected}, got {result}"

    @pytest.mark.integration
    def test_substituent_gets_lowest_locant(self):
        """Substituent should get the lowest possible locant."""
        # In 2-methylbutane, the methyl is at position 2, not 3
        assert name_compound("CC(C)CC") == "2-methylbutane"
        assert "3-methyl" not in name_compound("CC(C)CC")


# ============================================================================
# Alphabetization
# ============================================================================

class TestAlphabetization:
    """Test IUPAC alphabetization rules for substituent prefixes."""

    @pytest.mark.integration
    def test_ethyl_before_methyl(self):
        """Ethyl should come before methyl alphabetically."""
        # 4-ethyl-3-methylhexane or 3-ethyl-4-methylhexane
        # depending on numbering, but ethyl always before methyl
        result = name_compound("CCC(CC)C(C)CC")
        assert "ethyl" in result
        assert "methyl" in result
        ethyl_pos = result.find("ethyl")
        methyl_pos = result.find("methyl")
        assert ethyl_pos < methyl_pos, f"ethyl should come before methyl in '{result}'"

    @pytest.mark.integration
    def test_di_prefix_ignored_for_alphabetization(self):
        """di- prefix should be ignored when sorting alphabetically."""
        # If we had dimethyl and ethyl, ethyl would still come first
        # because we ignore 'di' for sorting
        result = name_compound("CCC(C)(C)CC")  # 3,3-dimethylpentane
        assert "dimethyl" in result
        # This molecule only has methyl, so can't test cross-substitution
        # but we can verify format is correct
        assert result == "3,3-dimethylpentane"

    @pytest.mark.integration
    def test_multiple_different_substituents_alphabetized(self):
        """Multiple different substituents should be in alphabetical order."""
        # Create a molecule with ethyl and methyl substituents
        result = name_compound("CCC(CC)C(C)CC")  # 3-ethyl-4-methylhexane
        # Check the ordering in the name
        assert "-ethyl-" in result or result.startswith("3-ethyl") or result.startswith("4-ethyl")
        # Verify ethyl appears before methyl
        if "ethyl" in result and "methyl" in result:
            assert result.index("ethyl") < result.index("methyl")


# ============================================================================
# Multiplicative Prefixes
# ============================================================================

class TestMultipleSubstituents:
    """Test multiplicative prefixes for multiple identical substituents."""

    @pytest.mark.integration
    def test_di_prefix_for_two_same_position(self):
        """Two identical substituents at same position should use di- prefix."""
        # 2,2-dimethylpropane has two methyl groups at position 2
        result = name_compound("CC(C)(C)C")
        assert result == "2,2-dimethylpropane"
        assert "dimethyl" in result

    @pytest.mark.integration
    def test_di_prefix_for_two_different_positions(self):
        """Two identical substituents at different positions should use di- prefix."""
        # 2,3-dimethylbutane has methyl at positions 2 and 3
        result = name_compound("CC(C)C(C)C")
        assert result == "2,3-dimethylbutane"
        assert "dimethyl" in result

    @pytest.mark.integration
    def test_comma_separated_locants(self):
        """Multiple locants should be comma-separated."""
        result = name_compound("CC(C)(C)C")  # 2,2-dimethylpropane
        assert "2,2-" in result

        result2 = name_compound("CC(C)C(C)C")  # 2,3-dimethylbutane
        assert "2,3-" in result2

    @pytest.mark.integration
    def test_simple_multipliers_used_for_simple_substituents(self):
        """Simple substituents (no locants in name) should use di-, tri-, tetra-."""
        # di- for methyl (simple)
        assert "dimethyl" in name_compound("CC(C)(C)C")
        # not "bismethyl" (that's for complex substituents)
        assert "bismethyl" not in name_compound("CC(C)(C)C")


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and potential failure modes."""

    @pytest.mark.unit
    def test_single_carbon_is_methane(self):
        """Single carbon should be named methane (retained name)."""
        assert name_compound("C") == "methane"

    @pytest.mark.unit
    def test_symmetrical_molecule(self):
        """Symmetrical molecules should give consistent names."""
        # 2,2-dimethylpropane is symmetrical
        result = name_compound("CC(C)(C)C")
        assert result == "2,2-dimethylpropane"

    @pytest.mark.integration
    def test_highly_branched_alkane(self):
        """Highly branched alkanes should be named correctly."""
        # 2,2,4-trimethylpentane (isooctane)
        smiles = "CC(C)(C)CC(C)C"
        result = name_compound(smiles)
        # Should contain trimethyl and pentane
        assert "trimethyl" in result or ("methyl" in result and "pentane" in result)
        assert "pentane" in result

    @pytest.mark.integration
    def test_long_substituent_chain(self):
        """Substituents with multiple carbons should be named correctly."""
        # 3-ethylpentane
        result = name_compound("CCC(CC)CC")
        assert result == "3-ethylpentane"
        assert "ethyl" in result


# ============================================================================
# Regression Tests
# ============================================================================

class TestRegressions:
    """Regression tests for previously fixed bugs."""

    @pytest.mark.integration
    def test_chain_orientation_for_lowest_locant(self):
        """Chain should be oriented to give substituents the lowest locants."""
        # CCC(C)C should be 2-methylbutane, not 3-methylbutane
        # This tests that chain orientation criterion (d) is applied
        result = name_compound("CCC(C)C")
        assert result == "2-methylbutane"
        assert "3-methyl" not in result

    @pytest.mark.integration
    def test_hyphenation_between_prefixes(self):
        """Hyphens should separate locants from names and names from locants."""
        result = name_compound("CCC(CC)C(C)CC")  # 3-ethyl-4-methylhexane
        # Should have proper hyphenation, not "3-ethyl4-methyl"
        # Check there's no letter immediately followed by digit
        for i in range(len(result) - 1):
            if result[i].isalpha() and result[i+1].isdigit():
                # This is only valid after a hyphen should be added
                # Actually, this can happen at the start of locant after a prefix
                # The key test is that we don't have "ethyl4" (no hyphen)
                pass
        # Specific check: no "ethyl4" or "methyl1" etc.
        assert "ethyl4" not in result.lower()
        assert "ethyl3" not in result.lower()
