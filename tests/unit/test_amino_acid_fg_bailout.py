"""
Tests for amino acid FG detection bailout (PEP-02).

When _name_amino_acid_systematic() encounters additional functional groups
(hydroxy, thiol, halogen, etc.) beyond amino + acid, it should bail out
(return None) so the general polyfunctional pipeline handles them correctly.
"""

import pytest
from orthonym import name_compound


class TestAminoAcidFGBailout:
    """Polyfunctional amino acids should bail out to general pipeline."""

    def test_amino_hydroxy_acid_includes_hydroxy(self):
        """2-amino-3-hydroxyoctadecanoic acid must include 'hydroxy' prefix."""
        # Sphingosine-related: long-chain amino acid with hydroxy group
        smiles = "OC(=O)C(N)C(O)CCCCCCCCCCCCCCC"
        result = name_compound(smiles)
        assert result is not None, "Should produce a name"
        assert "hydroxy" in result.lower(), (
            f"Expected 'hydroxy' in name but got: {result}"
        )

    def test_amino_thiol_acid_includes_thiol(self):
        """Non-standard amino acid with thiol: thiol group must not be lost."""
        # 2-amino-4-sulfanylbutanoic acid (homocysteine analog)
        smiles = "OC(=O)C(N)CCS"
        result = name_compound(smiles)
        assert result is not None, "Should produce a name"
        # Check for sulfanyl (IUPAC preferred) or mercapto or thio
        has_thiol = any(
            term in result.lower()
            for term in ["sulfanyl", "thio", "mercapto"]
        )
        assert has_thiol, (
            f"Expected thiol-related prefix in name but got: {result}"
        )

    def test_simple_amino_acid_unchanged(self):
        """Simple 2-aminopentanoic acid (norvaline-like) still named systematically."""
        smiles = "OC(=O)C(N)CCC"
        result = name_compound(smiles)
        assert result is not None, "Should produce a name"
        assert result == "2-aminopentanoic acid", (
            f"Expected '2-aminopentanoic acid' but got: {result}"
        )

    def test_glycine_trivial_name_preserved(self):
        """Glycine trivial name lookup unaffected by bailout logic."""
        smiles = "NCC(=O)O"
        result = name_compound(smiles)
        assert result is not None, "Should produce a name"
        assert result == "glycine", f"Expected 'glycine' but got: {result}"

    def test_alanine_trivial_name_preserved(self):
        """Alanine trivial name lookup unaffected by bailout logic."""
        smiles = "CC(N)C(=O)O"
        result = name_compound(smiles)
        assert result is not None, "Should produce a name"
        assert result == "alanine", f"Expected 'alanine' but got: {result}"

    def test_amino_acid_with_halogen_includes_halogen(self):
        """2-amino-3-chloropentanoic acid must include 'chloro' prefix."""
        smiles = "OC(=O)C(N)C(Cl)CC"
        result = name_compound(smiles)
        assert result is not None, "Should produce a name"
        assert "chloro" in result.lower(), (
            f"Expected 'chloro' in name but got: {result}"
        )

    def test_amino_hydroxy_amide(self):
        """2-amino-3-hydroxyoctadecanamide must include 'hydroxy' prefix."""
        smiles = "NC(=O)C(N)C(O)CCCCCCCCCCCCCCC"
        result = name_compound(smiles)
        assert result is not None, "Should produce a name"
        assert "hydroxy" in result.lower(), (
            f"Expected 'hydroxy' in name but got: {result}"
        )
