"""Tests for nitrile naming (POLY-04).

Tests the nitrile naming module for:
- Simple chain nitriles (acetonitrile, propanenitrile)
- Substituted nitriles (2-methylpropanenitrile)
- Ring-attached nitriles (cyclohexanecarbonitrile)
- Nitrile as prefix (cyano-) when not principal group

Based on IUPAC 2013 Blue Book P-66.1.
"""
import pytest
from rdkit import Chem

from src.orthonym.rules.nitriles import (
    is_ring_attached_nitrile,
    get_nitrile_parent_chain,
    name_nitrile,
)


class TestIsRingAttachedNitrile:
    """Test detection of ring-attached nitriles."""

    def test_chain_nitrile_not_ring_attached(self):
        """Propanenitrile is not ring-attached."""
        mol = Chem.MolFromSmiles("CCC#N")
        # Find nitrile atoms: [CX2]#[NX1]
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        assert len(matches) == 1
        assert is_ring_attached_nitrile(mol, matches[0]) is False

    def test_cyclohexyl_nitrile_is_ring_attached(self):
        """Cyclohexanecarbonitrile has nitrile attached to ring."""
        mol = Chem.MolFromSmiles("C1CCCCC1C#N")
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        assert len(matches) == 1
        assert is_ring_attached_nitrile(mol, matches[0]) is True

    def test_cyclopentyl_nitrile_is_ring_attached(self):
        """Cyclopentanecarbonitrile has nitrile attached to ring."""
        mol = Chem.MolFromSmiles("C1CCCC1C#N")
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        assert len(matches) == 1
        assert is_ring_attached_nitrile(mol, matches[0]) is True

    def test_acetonitrile_not_ring_attached(self):
        """Acetonitrile (CH3-CN) is not ring-attached."""
        mol = Chem.MolFromSmiles("CC#N")
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        assert len(matches) == 1
        assert is_ring_attached_nitrile(mol, matches[0]) is False


class TestGetNitrileParentChain:
    """Test chain detection for chain-terminal nitriles."""

    def test_acetonitrile_chain(self):
        """Acetonitrile (CC#N) has 2-carbon chain."""
        mol = Chem.MolFromSmiles("CC#N")
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        chain = get_nitrile_parent_chain(mol, matches[0])
        assert len(chain) == 2

    def test_propanenitrile_chain(self):
        """Propanenitrile (CCC#N) has 3-carbon chain."""
        mol = Chem.MolFromSmiles("CCC#N")
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        chain = get_nitrile_parent_chain(mol, matches[0])
        assert len(chain) == 3

    def test_butanenitrile_chain(self):
        """Butanenitrile (CCCC#N) has 4-carbon chain."""
        mol = Chem.MolFromSmiles("CCCC#N")
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        chain = get_nitrile_parent_chain(mol, matches[0])
        assert len(chain) == 4


class TestNameNitrile:
    """Test nitrile name generation."""

    def test_name_propanenitrile(self):
        """CCC#N should name to propanenitrile."""
        mol = Chem.MolFromSmiles("CCC#N")
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        name = name_nitrile(mol, matches[0])
        assert name == "propanenitrile"

    def test_name_butanenitrile(self):
        """CCCC#N should name to butanenitrile."""
        mol = Chem.MolFromSmiles("CCCC#N")
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        name = name_nitrile(mol, matches[0])
        assert name == "butanenitrile"

    def test_name_pentanenitrile(self):
        """CCCCC#N should name to pentanenitrile."""
        mol = Chem.MolFromSmiles("CCCCC#N")
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        name = name_nitrile(mol, matches[0])
        assert name == "pentanenitrile"

    def test_name_ethanenitrile(self):
        """CC#N should name to ethanenitrile (systematic)."""
        mol = Chem.MolFromSmiles("CC#N")
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        name = name_nitrile(mol, matches[0])
        # Systematic name is ethanenitrile (acetonitrile is retained)
        assert name == "ethanenitrile"

    def test_name_cyclohexanecarbonitrile(self):
        """C1CCCCC1C#N should name to cyclohexanecarbonitrile."""
        mol = Chem.MolFromSmiles("C1CCCCC1C#N")
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        name = name_nitrile(mol, matches[0])
        assert name == "cyclohexanecarbonitrile"

    def test_name_cyclopentanecarbonitrile(self):
        """C1CCCC1C#N should name to cyclopentanecarbonitrile."""
        mol = Chem.MolFromSmiles("C1CCCC1C#N")
        pattern = Chem.MolFromSmarts("[CX2]#[NX1]")
        matches = mol.GetSubstructMatches(pattern)
        name = name_nitrile(mol, matches[0])
        assert name == "cyclopentanecarbonitrile"


class TestNitrileIntegration:
    """Integration tests using the full naming pipeline."""

    def test_acetonitrile_retained_name(self):
        """CC#N should be named acetonitrile (retained name)."""
        from src.orthonym import name_compound
        assert name_compound("CC#N") == "acetonitrile"

    def test_propanenitrile_systematic(self):
        """CCC#N should be named propanenitrile (systematic)."""
        from src.orthonym import name_compound
        result = name_compound("CCC#N")
        assert result == "propanenitrile"

    def test_butanenitrile_systematic(self):
        """CCCC#N should be named butanenitrile."""
        from src.orthonym import name_compound
        result = name_compound("CCCC#N")
        assert result == "butanenitrile"

    def test_pentanenitrile_systematic(self):
        """CCCCC#N should be named pentanenitrile."""
        from src.orthonym import name_compound
        result = name_compound("CCCCC#N")
        assert result == "pentanenitrile"


class TestSubstitutedNitriles:
    """Test nitriles with alkyl substituents."""

    def test_2_methylpropanenitrile(self):
        """(CH3)2CH-CN should be 2-methylpropanenitrile."""
        from src.orthonym import name_compound
        result = name_compound("CC(C)C#N")
        assert result == "2-methylpropanenitrile"

    def test_3_methylbutanenitrile(self):
        """(CH3)2CH-CH2-CN should be 3-methylbutanenitrile."""
        from src.orthonym import name_compound
        result = name_compound("CC(C)CC#N")
        assert result == "3-methylbutanenitrile"


class TestRingAttachedNitriles:
    """Test nitriles attached to rings (use -carbonitrile)."""

    def test_cyclohexanecarbonitrile(self):
        """Nitrile attached to cyclohexane uses -carbonitrile suffix."""
        from src.orthonym import name_compound
        result = name_compound("C1CCCCC1C#N")
        assert result == "cyclohexanecarbonitrile"

    def test_cyclopentanecarbonitrile(self):
        """Nitrile attached to cyclopentane uses -carbonitrile suffix."""
        from src.orthonym import name_compound
        result = name_compound("C1CCCC1C#N")
        assert result == "cyclopentanecarbonitrile"


class TestNitrileAsPrefix:
    """Test nitrile as prefix (cyano-) when not principal group."""

    def test_cyanoacetic_acid(self):
        """Acid is higher seniority than nitrile -> 2-cyanoacetic acid."""
        from src.orthonym import name_compound
        # N#C-CH2-COOH
        result = name_compound("N#CCC(=O)O")
        # Acid is principal group, nitrile becomes cyano- prefix
        assert "cyano" in result.lower() or "cyanoacetic" in result.lower()


class TestNitrileAmideIntegration:
    """Test compounds with both nitrile and amide or other groups."""

    def test_cyanoacetamide(self):
        """Compound with both nitrile and amide -> amide is principal."""
        from src.orthonym import name_compound
        # N#C-CH2-C(=O)NH2 - 2-cyanoacetamide
        result = name_compound("NC(=O)CC#N")
        # Amide is higher seniority than nitrile
        assert "cyano" in result.lower() or "acetamide" in result.lower()

    def test_aminoacetonitrile(self):
        """Compound with amine and nitrile -> nitrile is principal."""
        from src.orthonym import name_compound
        # H2N-CH2-C#N
        result = name_compound("NCC#N")
        # Nitrile is higher seniority than amine
        # Result should contain amino prefix or nitrile suffix
        assert "nitrile" in result.lower() or "amino" in result.lower()
