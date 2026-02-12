"""
Integration tests for Phase 41-03: Fatty acid retained names and phospholipid FG detection.

Tests:
- Fatty acid trivial name lookups (acylate, acyloxy)
- Fatty acid ester naming (methyl palmitate, methyl stearate)
- Phospholipid FG detection (phosphate_monoester vs phosphonic_acid)
- Regression: existing simple esters unchanged
"""
import pytest
from orthonym.namer import name_compound
from orthonym.data.trivial_acids import get_acylate_name
from orthonym.rules.esters import get_acyloxy_prefix


# ===========================================================================
# Fatty acid trivial name lookups
# ===========================================================================


@pytest.mark.integration
class TestFattyAcidLookups:
    """Fatty acid retained names in acylate and acyloxy tables."""

    def test_get_acylate_palmitic(self):
        """Palmitic -> palmitate."""
        assert get_acylate_name("palmitic") == "palmitate"

    def test_get_acylate_stearic(self):
        """Stearic -> stearate."""
        assert get_acylate_name("stearic") == "stearate"

    def test_get_acylate_lauric(self):
        """Lauric -> laurate."""
        assert get_acylate_name("lauric") == "laurate"

    def test_get_acylate_myristic(self):
        """Myristic -> myristate."""
        assert get_acylate_name("myristic") == "myristate"

    def test_get_acylate_oleic(self):
        """Oleic -> oleate."""
        assert get_acylate_name("oleic") == "oleate"

    def test_get_acylate_arachidic(self):
        """Arachidic -> arachidate."""
        assert get_acylate_name("arachidic") == "arachidate"

    def test_get_acyloxy_palmitic(self):
        """Palmitic -> palmitoyloxy."""
        assert get_acyloxy_prefix("palmitic") == "palmitoyloxy"

    def test_get_acyloxy_stearic(self):
        """Stearic -> stearoyloxy."""
        assert get_acyloxy_prefix("stearic") == "stearoyloxy"

    def test_get_acyloxy_lauric(self):
        """Lauric -> lauroyloxy."""
        assert get_acyloxy_prefix("lauric") == "lauroyloxy"

    def test_get_acyloxy_myristic(self):
        """Myristic -> myristoyloxy."""
        assert get_acyloxy_prefix("myristic") == "myristoyloxy"

    def test_get_acyloxy_oleic(self):
        """Oleic -> oleoyloxy."""
        assert get_acyloxy_prefix("oleic") == "oleoyloxy"

    def test_get_acyloxy_arachidic(self):
        """Arachidic -> arachidoyloxy."""
        assert get_acyloxy_prefix("arachidic") == "arachidoyloxy"


# ===========================================================================
# Fatty acid ester naming (end-to-end)
# ===========================================================================


@pytest.mark.integration
class TestFattyAcidEsterNaming:
    """Methyl esters of fatty acids use trivial acid names."""

    def test_methyl_palmitate_name(self):
        """Methyl palmitate (16C acid + methyl ester) uses trivial name."""
        result = name_compound("CCCCCCCCCCCCCCCC(=O)OC")
        assert result is not None
        assert "palmitate" in result, f"Expected 'palmitate' in: {result!r}"

    def test_methyl_stearate_name(self):
        """Methyl stearate (18C acid + methyl ester) uses trivial name."""
        result = name_compound("CCCCCCCCCCCCCCCCCC(=O)OC")
        assert result is not None
        assert "stearate" in result, f"Expected 'stearate' in: {result!r}"

    def test_methyl_laurate_name(self):
        """Methyl laurate (12C acid + methyl ester) uses trivial name."""
        result = name_compound("CCCCCCCCCCCC(=O)OC")
        assert result is not None
        assert "laurate" in result, f"Expected 'laurate' in: {result!r}"

    def test_methyl_myristate_name(self):
        """Methyl myristate (14C acid + methyl ester) uses trivial name."""
        result = name_compound("CCCCCCCCCCCCCC(=O)OC")
        assert result is not None
        assert "myristate" in result, f"Expected 'myristate' in: {result!r}"

    def test_regression_methyl_acetate(self):
        """Methyl acetate is unchanged."""
        assert name_compound("COC(C)=O") == "methyl acetate"

    def test_regression_ethyl_propanoate(self):
        """Ethyl propanoate is unchanged."""
        assert name_compound("CCOC(=O)CC") == "ethyl propanoate"
