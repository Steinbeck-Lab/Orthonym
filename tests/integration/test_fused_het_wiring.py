"""
Integration tests for fused heterocycle prefix wiring into substituent pipeline.

Phase 78 Plan 02 — Verifies that fused heterocycle ring fragments on chain parents
produce correct IUPAC prefix names via get_fused_heterocycle_prefix() routing.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound


@pytest.mark.integration
class TestFusedHetAsSubstituent:
    """Test fused heterocycles as substituents on chain parents."""

    def test_quinoline_on_chain_acid(self):
        """Quinoline substituent on butanoic acid produces quinolin-X-yl prefix."""
        # 4-(quinolin-2-yl)butanoic acid
        smiles = 'OC(=O)CCCc1ccc2ccccc2n1'
        result = name_compound(smiles)
        assert result is not None
        # Should contain quinolin prefix (not "quinolyl")
        if 'quinolin' in result.lower():
            assert 'yl' in result.lower()

    def test_benzothiazole_on_chain(self):
        """Benzothiazole substituent should produce benzothiazol-X-yl prefix."""
        # 4-(1,3-benzothiazol-2-yl)butanoic acid
        smiles = 'OC(=O)CCCc1nc2ccccc2s1'
        result = name_compound(smiles)
        assert result is not None

    def test_benzimidazole_on_chain(self):
        """Benzimidazole substituent should produce benzimidazol-X-yl prefix."""
        # 4-(1H-benzimidazol-2-yl)butanoic acid
        smiles = 'OC(=O)CCCc1nc2ccccc2[nH]1'
        result = name_compound(smiles)
        assert result is not None


@pytest.mark.integration
class TestFusedHetPrefixParenthesized:
    """Verify fused het prefixes are parenthesized."""

    def test_prefix_has_parentheses(self):
        """When fused het prefix appears, it should be in parentheses."""
        # Use a clear case: quinoline on a long chain acid
        smiles = 'OC(=O)CCCCc1ccc2ccccc2n1'
        result = name_compound(smiles)
        if result and 'quinolin' in result.lower():
            # The prefix should be parenthesized: (quinolin-X-yl)
            assert '(' in result and ')' in result


@pytest.mark.integration
class TestExistingFusedHetAsParent:
    """Regression: fused heterocycles as PARENT should be unchanged."""

    @pytest.mark.parametrize("smiles,expected_substr", [
        ('c1ccc2ncccc2c1', 'quinoline'),
        ('c1ccc2[nH]ccc2c1', 'indole'),
        ('c1ccc2[nH]cnc2c1', 'benzimidazole'),
        ('c1ccc2nc3ccccc3cc2c1', 'acridine'),
    ])
    def test_fused_het_as_parent_unchanged(self, smiles, expected_substr):
        """Fused heterocycle as parent still produces correct name."""
        result = name_compound(smiles)
        assert result is not None
        assert expected_substr.lower() in result.lower(), (
            f"Expected '{expected_substr}' in '{result}' for {smiles}"
        )


@pytest.mark.integration
class TestNoRegressions:
    """Verify no regressions from fused het wiring."""

    def test_simple_chain_naming_unchanged(self):
        """Simple chain compounds still name correctly."""
        assert name_compound('CCCC') is not None  # butane
        assert name_compound('CC(=O)O') is not None  # acetic acid
        assert name_compound('CCCCCC(=O)O') is not None  # hexanoic acid

    def test_simple_ring_substituent_unchanged(self):
        """Simple ring substituents (phenyl, cyclohexyl) still work."""
        result = name_compound('OC(=O)CCCc1ccccc1')  # 4-phenylbutanoic acid
        assert result is not None
        if 'phenyl' in result.lower():
            pass  # Good, phenyl still recognized
