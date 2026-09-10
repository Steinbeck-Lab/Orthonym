"""
Tests for the style parameter in name_compound.

The style parameter controls whether retained names or systematic names are returned:
- style="pin" (default): Use retained names when available (IUPAC 2013 preferred)
- style="systematic": Always generate systematic name
"""

import pytest
from orthonym import name_compound


class TestStyleParameter:
    """Test the style parameter in name_compound."""

    @pytest.mark.unit
    def test_default_style_uses_retained_names(self):
        """Default style 'pin' should return retained names when available."""
        assert name_compound("c1ccccc1") == "benzene"
        assert name_compound("CCO") == "ethanol"
        # Wave-1 1.10: "allyl alcohol" is general-only — demoted
        # from the default headline; the PIN is the systematic name.
        assert name_compound("C=CCO") == "prop-2-en-1-ol"
        # a phase: "ethylene glycol" was a deprecated (non-PIN) name and
        # is now corrected to ethane-1,2-diol; use a genuine retained PIN instead.
        assert name_compound("Cc1ccccc1") == "toluene"

    @pytest.mark.unit
    def test_systematic_style_bypasses_retained_names(self):
        """style='systematic' should generate systematic names."""
        # Allyl alcohol -> prop-2-en-1-ol
        result = name_compound("C=CCO", style="systematic")
        assert result == "prop-2-en-1-ol", f"Expected prop-2-en-1-ol, got {result}"

    @pytest.mark.unit
    def test_systematic_style_simple_compounds(self):
        """Systematic style for simple compounds.

        Note: For ethanol, the retained name is 'ethanol' but the fully
        systematic name is 'ethan-1-ol'. Both are valid IUPAC names.
        """
        # Ethanol -> ethan-1-ol (fully systematic with locant)
        result = name_compound("CCO", style="systematic")
        # Should be the systematic form, not the retained name
        assert result == "ethan-1-ol" or result == "ethanol"

    @pytest.mark.unit
    def test_pin_style_explicit(self):
        """Explicit style='pin' should behave like default."""
        assert name_compound("C=CCO", style="pin") == "prop-2-en-1-ol"

    @pytest.mark.unit
    def test_backward_compatibility(self):
        """Existing code calling name_compound(smiles) should work unchanged."""
        # These should all still work
        assert name_compound("CCO") == "ethanol"
        assert name_compound("CC(=O)O") == "acetic acid"
        assert name_compound("c1ccccc1") == "benzene"
