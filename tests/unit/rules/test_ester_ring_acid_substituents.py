"""Tests for /: Ring-acid substituent discovery in ester handler.

Verifies that substituted aromatic esters (e.g., ethyl 4-aminobenzoate)
include ring substituents in the generated name, and that unsubstituted
ring esters and chain esters remain unchanged.
"""
import pytest
from orthonym import name_compound


@pytest.mark.unit
class TestSubstitutedAromaticEsters:
    """/: Ring-acid substituents must appear in ester names."""

    def test_ethyl_4_aminobenzoate(self):
        """Ethyl 4-aminobenzoate: amino substituent on ring acid."""
        result = name_compound("CCOC(=O)c1ccc(N)cc1")
        assert result is not None
        assert "4-amino" in result, f"Expected '4-amino' in name, got: {result}"
        assert "benzoate" in result, f"Expected 'benzoate' in name, got: {result}"

    def test_ethyl_4_hydroxybenzoate(self):
        """Ethyl 4-hydroxybenzoate: hydroxy substituent on ring acid."""
        result = name_compound("CCOC(=O)c1ccc(O)cc1")
        assert result is not None
        assert "4-hydroxy" in result, f"Expected '4-hydroxy' in name, got: {result}"
        assert "benzoate" in result, f"Expected 'benzoate' in name, got: {result}"

    def test_ethyl_4_chlorobenzoate(self):
        """Ethyl 4-chlorobenzoate: chloro substituent on ring acid."""
        result = name_compound("CCOC(=O)c1ccc(Cl)cc1")
        assert result is not None
        assert "4-chloro" in result, f"Expected '4-chloro' in name, got: {result}"
        assert "benzoate" in result, f"Expected 'benzoate' in name, got: {result}"

    def test_ethyl_4_methylbenzoate(self):
        """Ethyl 4-methylbenzoate: methyl substituent on ring acid."""
        result = name_compound("CCOC(=O)c1ccc(C)cc1")
        assert result is not None
        assert "4-methyl" in result, f"Expected '4-methyl' in name, got: {result}"
        assert "benzoate" in result, f"Expected 'benzoate' in name, got: {result}"

    def test_ethyl_4_fluorobenzoate(self):
        """Ethyl 4-fluorobenzoate: fluoro substituent on ring acid."""
        result = name_compound("CCOC(=O)c1ccc(F)cc1")
        assert result is not None
        assert "4-fluoro" in result, f"Expected '4-fluoro' in name, got: {result}"
        assert "benzoate" in result, f"Expected 'benzoate' in name, got: {result}"

    def test_ethyl_4_nitrobenzoate(self):
        """Ethyl 4-nitrobenzoate: nitro substituent on ring acid."""
        result = name_compound("CCOC(=O)c1ccc([N+](=O)[O-])cc1")
        assert result is not None
        assert "4-nitro" in result, f"Expected '4-nitro' in name, got: {result}"
        assert "benzoate" in result, f"Expected 'benzoate' in name, got: {result}"


@pytest.mark.unit
class TestEsterRegressionGuards:
    """Ensure unsubstituted ring esters and chain esters are unchanged."""

    def test_ethyl_benzoate_unsubstituted(self):
        """Unsubstituted ethyl benzoate: no spurious substituent prefixes."""
        result = name_compound("CCOC(=O)c1ccccc1")
        assert result is not None
        assert "benzoate" in result, f"Expected 'benzoate' in name, got: {result}"
        # Should NOT have any substituent prefixes
        assert "amino" not in result
        assert "chloro" not in result
        assert "methyl" not in result

    def test_ethyl_acetate_chain_ester(self):
        """Chain ester (ethyl acetate): unchanged by ring-acid fix."""
        result = name_compound("CCOC(C)=O")
        assert result is not None
        assert "ethyl" in result, f"Expected 'ethyl' in name, got: {result}"
        # Should be either "ethyl acetate" or "ethyl ethanoate"
        assert "acetate" in result or "ethanoate" in result
