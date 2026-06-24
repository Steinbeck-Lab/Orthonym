"""Tests for anhydride handler substituent discovery (Phase 125).

Verifies that branched acyl fragments in anhydrides produce correct
acid names including substituent prefixes.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestAnhydrideSubstituents:
    """Ensure anhydride handler names branched acyl chains correctly."""

    def test_isobutyric_anhydride(self):
        # CC(C)C(=O)OC(=O)C(C)C = 2-methylpropanoic anhydride
        result = name_compound("CC(C)C(=O)OC(=O)C(C)C")
        assert "methylpropanoic" in result, f"Expected branch in '{result}'"
        assert "anhydride" in result

    def test_simple_ethanoic_anhydride_unchanged(self):
        # Retained acid name is the PIN (P-65.1.1.1): acetic, not ethanoic.
        result = name_compound("CC(=O)OC(=O)C")
        assert "acetic anhydride" == result

    def test_simple_butanoic_anhydride_unchanged(self):
        # CCCC(=O)OC(=O)CCC = symmetric butanoic anhydride (4C each side)
        result = name_compound("CCCC(=O)OC(=O)CCC")
        assert "butanoic anhydride" == result

    def test_cyclic_butanedioic_anhydride_unchanged(self):
        result = name_compound("O=C1CCC(=O)O1")
        assert "butanedioic anhydride" == result
