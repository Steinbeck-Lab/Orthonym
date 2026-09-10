"""Unit tests for chain-vs-ring suffix selection.

Verifies IUPAC suffix context rules:
- Chain-parent molecules (principal chain chosen over ring) get chain suffixes:
  -oic acid, -al, -amide, -nitrile (not -carboxylic acid, -carbaldehyde, etc.)
- Ring-parent molecules (ring is parent structure) get ring suffixes:
  -carboxylic acid, -carbaldehyde, -carboxamide, -carbonitrile

The root cause of incorrect suffix selection was using features.is_cyclic
(True for ANY molecule containing a ring) instead of the chain_is_parent flag
and principal_chain existence to determine suffix form.
"""

import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestChainParentSuffixes:
    """Chain-parent molecules with rings should use chain suffix forms."""

    def test_phenylacetic_acid_chain_suffix(self):
        """Phenylacetic acid: chain is parent (2C chain > benzene for acid).
        Should produce 'oic acid' not 'carboxylic acid'.
        SMILES: OC(=O)Cc1ccccc1
        """
        result = name_compound("OC(=O)Cc1ccccc1")
        assert "oic acid" in result or "acetic acid" in result, (
            f"Expected chain suffix 'oic acid' for chain-parent molecule, got: {result}"
        )
        # Ring suffix should NOT be present (unless it's a retained name)
        if "acetic acid" not in result:
            assert "carboxylic acid" not in result, (
                f"Chain-parent molecule incorrectly got ring suffix 'carboxylic acid': {result}"
            )

    def test_cyclohexylpropanoic_acid_chain_suffix(self):
        """3-cyclohexylpropanoic acid: 3C chain is parent.
        Should produce 'oic acid' not 'carboxylic acid'.
        SMILES: OC(=O)CCC1CCCCC1
        """
        result = name_compound("OC(=O)CCC1CCCCC1")
        assert "oic acid" in result, (
            f"Expected chain suffix 'oic acid' for chain-parent molecule, got: {result}"
        )
        assert "carboxylic acid" not in result, (
            f"Chain-parent molecule incorrectly got ring suffix 'carboxylic acid': {result}"
        )

    def test_phenylpropanal_chain_suffix(self):
        """3-phenylpropanal: 3C chain is parent.
        Should produce 'al' suffix not 'carbaldehyde'.
        SMILES: O=CCCc1ccccc1
        """
        result = name_compound("O=CCCc1ccccc1")
        assert "al" in result, (
            f"Expected chain suffix 'al' for chain-parent aldehyde, got: {result}"
        )
        assert "carbaldehyde" not in result, (
            f"Chain-parent molecule incorrectly got ring suffix 'carbaldehyde': {result}"
        )


@pytest.mark.unit
class TestRingParentSuffixes:
    """Ring-parent molecules should use ring suffix forms."""

    def test_benzoic_acid_ring_suffix(self):
        """Benzoic acid: ring is parent (benzene-COOH).
        Should produce 'carboxylic acid' or retained name 'benzoic acid'.
        SMILES: OC(=O)c1ccccc1
        """
        result = name_compound("OC(=O)c1ccccc1")
        # Accept either retained name or systematic ring suffix
        assert "carboxylic acid" in result or "benzoic acid" in result, (
            f"Expected ring suffix 'carboxylic acid' or retained 'benzoic acid', got: {result}"
        )

    def test_cyclohexanecarbaldehyde_ring_suffix(self):
        """Cyclohexanecarbaldehyde: ring is parent (6-ring > 1C chain).
        Should produce 'carbaldehyde'.
        SMILES: O=CC1CCCCC1
        """
        result = name_compound("O=CC1CCCCC1")
        assert "carbaldehyde" in result, (
            f"Expected ring suffix 'carbaldehyde' for ring-parent aldehyde, got: {result}"
        )

    def test_cyclopentanecarbonitrile_ring_suffix(self):
        """Cyclopentanecarbonitrile: ring is parent (5-ring > 1C chain).
        Should produce 'carbonitrile'.
        SMILES: N#CC1CCCC1
        """
        result = name_compound("N#CC1CCCC1")
        assert "carbonitrile" in result, (
            f"Expected ring suffix 'carbonitrile' for ring-parent nitrile, got: {result}"
        )

    def test_methylbenzamide_ring_suffix(self):
        """4-methylbenzamide: ring is parent.
        Should produce 'carboxamide' or retained 'benzamide'.
        SMILES: CC1=CC=C(C(N)=O)C=C1
        """
        result = name_compound("CC1=CC=C(C(N)=O)C=C1")
        assert "carboxamide" in result or "benzamide" in result, (
            f"Expected ring suffix 'carboxamide' or 'benzamide', got: {result}"
        )
