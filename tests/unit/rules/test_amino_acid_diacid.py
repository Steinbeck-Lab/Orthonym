"""Tests for AMAC-01: Dicarboxylic amino acid bailout to polyfunctional pipeline.

Verifies that amino acids with 2+ carboxylic acid groups (aspartic acid,
glutamic acid) bail out of the specialized amino acid handler and produce
correct "dioic acid" names via the polyfunctional pipeline.
"""
import pytest
from orthonym import name_compound


@pytest.mark.unit
class TestDicarboxylicAminoAcidBailout:
    """AMAC-01: Dicarboxylic amino acids should produce 'dioic' names."""

    def test_aspartic_acid_nonstereo(self):
        """Non-stereo aspartic acid should still use retained name."""
        result = name_compound("NC(CC(=O)O)C(=O)O")
        assert result == "aspartic acid"

    def test_glutamic_acid_nonstereo(self):
        """Non-stereo glutamic acid should still use retained name."""
        result = name_compound("NC(CCC(=O)O)C(=O)O")
        assert result == "glutamic acid"

    def test_aspartic_acid_stereo(self):
        """Stereo aspartic acid should produce name containing 'dioic'."""
        result = name_compound("N[C@@H](CC(=O)O)C(=O)O")
        assert result is not None
        assert "dioic" in result, f"Expected 'dioic' in name, got: {result}"
        # Should produce "(2S)-2-aminobutanedioic acid"
        assert "aminobutanedioic acid" in result

    def test_glutamic_acid_stereo(self):
        """Stereo glutamic acid should produce name containing 'dioic'."""
        result = name_compound("N[C@@H](CCC(=O)O)C(=O)O")
        assert result is not None
        assert "dioic" in result, f"Expected 'dioic' in name, got: {result}"
        # Should produce "(2S)-2-aminopentanedioic acid"
        assert "aminopentanedioic acid" in result

    def test_mono_cooh_amino_acid_unchanged(self):
        """2-aminobutanoic acid (1 COOH) should NOT bail out -- regression guard."""
        result = name_compound("NC(CC)C(=O)O")
        assert result == "2-aminobutanoic acid"

    def test_alanine_unchanged(self):
        """Alanine (1 COOH) should still produce 'alanine' -- regression guard."""
        result = name_compound("NC(C)C(=O)O")
        assert result == "alanine"
