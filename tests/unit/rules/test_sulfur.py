"""Unit tests for sulfur compound naming."""

import pytest
from rdkit import Chem

from orthonym.rules.sulfur import (
    name_thiol,
    name_sulfide,
    name_sulfoxide,
    name_sulfone,
    name_sulfonic_acid,
    get_sulfur_prefix,
    _count_alkyl_carbons,
)


class TestThiolNaming:
    """Tests for thiol (-SH) naming."""

    def test_methanethiol(self):
        """CS -> methanethiol"""
        mol = Chem.MolFromSmiles("CS")
        result = name_thiol(mol, (1, 0), "methan", locant=1)
        assert result == "methanethiol"

    def test_ethanethiol(self):
        """CCS -> ethanethiol"""
        mol = Chem.MolFromSmiles("CCS")
        result = name_thiol(mol, (2, 1), "ethan", locant=1)
        assert result == "ethanethiol"

    def test_propane_1_thiol(self):
        """CCCS -> propane-1-thiol"""
        mol = Chem.MolFromSmiles("CCCS")
        result = name_thiol(mol, (3, 2), "propan", locant=1)
        assert result == "propane-1-thiol"

    def test_propane_2_thiol(self):
        """CC(S)C -> propane-2-thiol"""
        mol = Chem.MolFromSmiles("CC(S)C")
        result = name_thiol(mol, (2, 1), "propan", locant=2)
        assert result == "propane-2-thiol"


class TestSulfideNaming:
    """Tests for sulfide (thioether) naming."""

    def test_dimethyl_sulfide(self):
        """CSC -> dimethyl sulfide"""
        mol = Chem.MolFromSmiles("CSC")
        result = name_sulfide(mol, 1)  # S is at index 1
        assert result == "dimethyl sulfide"

    def test_diethyl_sulfide(self):
        """CCSCC -> diethyl sulfide"""
        mol = Chem.MolFromSmiles("CCSCC")
        result = name_sulfide(mol, 2)  # S is at index 2
        assert result == "diethyl sulfide"

    def test_ethyl_methyl_sulfide(self):
        """CCSC -> ethyl methyl sulfide"""
        mol = Chem.MolFromSmiles("CCSC")
        result = name_sulfide(mol, 2)  # S is at index 2
        assert result == "ethyl methyl sulfide"

    def test_methyl_propyl_sulfide(self):
        """CCCSC -> methyl propyl sulfide"""
        mol = Chem.MolFromSmiles("CCCSC")
        result = name_sulfide(mol, 3)  # S is at index 3
        assert result == "methyl propyl sulfide"


class TestSulfoxideNaming:
    """Tests for sulfoxide (R-SO-R') naming."""

    def test_dimethyl_sulfoxide(self):
        """CS(=O)C -> dimethyl sulfoxide"""
        mol = Chem.MolFromSmiles("CS(=O)C")
        # SMARTS match: S, O, C, C
        result = name_sulfoxide(mol, (1, 2, 0, 3))
        assert result == "dimethyl sulfoxide"

    def test_diethyl_sulfoxide(self):
        """CCS(=O)CC -> diethyl sulfoxide"""
        mol = Chem.MolFromSmiles("CCS(=O)CC")
        result = name_sulfoxide(mol, (2, 3, 1, 4))
        assert result == "diethyl sulfoxide"

    def test_ethyl_methyl_sulfoxide(self):
        """CCS(=O)C -> ethyl methyl sulfoxide"""
        mol = Chem.MolFromSmiles("CCS(=O)C")
        result = name_sulfoxide(mol, (2, 3, 1, 4))
        assert result == "ethyl methyl sulfoxide"


class TestSulfoneNaming:
    """Tests for sulfone (R-SO2-R') naming."""

    def test_dimethyl_sulfone(self):
        """CS(=O)(=O)C -> dimethyl sulfone"""
        mol = Chem.MolFromSmiles("CS(=O)(=O)C")
        # SMARTS match includes S, 2 O, 2 C
        result = name_sulfone(mol, (1, 2, 3, 0, 4))
        assert result == "dimethyl sulfone"

    def test_diethyl_sulfone(self):
        """CCS(=O)(=O)CC -> diethyl sulfone"""
        mol = Chem.MolFromSmiles("CCS(=O)(=O)CC")
        result = name_sulfone(mol, (2, 3, 4, 1, 5))
        assert result == "diethyl sulfone"

    def test_ethyl_methyl_sulfone(self):
        """CCS(=O)(=O)C -> ethyl methyl sulfone"""
        mol = Chem.MolFromSmiles("CCS(=O)(=O)C")
        result = name_sulfone(mol, (2, 3, 4, 1, 5))
        assert result == "ethyl methyl sulfone"


class TestSulfonicAcidNaming:
    """Tests for sulfonic acid (-SO3H) naming."""

    def test_methanesulfonic_acid(self):
        """CS(=O)(=O)O -> methanesulfonic acid"""
        mol = Chem.MolFromSmiles("CS(=O)(=O)O")
        result = name_sulfonic_acid(mol, (1, 2, 3, 4), "methane")
        assert result == "methanesulfonic acid"

    def test_ethanesulfonic_acid(self):
        """CCS(=O)(=O)O -> ethanesulfonic acid"""
        mol = Chem.MolFromSmiles("CCS(=O)(=O)O")
        result = name_sulfonic_acid(mol, (2, 3, 4, 5), "ethane")
        assert result == "ethanesulfonic acid"

    def test_benzenesulfonic_acid(self):
        """c1ccccc1S(=O)(=O)O -> benzenesulfonic acid"""
        mol = Chem.MolFromSmiles("c1ccccc1S(=O)(=O)O")
        result = name_sulfonic_acid(mol, (6, 7, 8, 9), "benzene")
        assert result == "benzenesulfonic acid"


class TestSulfurPrefix:
    """Tests for sulfur group prefix forms."""

    def test_thiol_prefix(self):
        """Thiol prefix is 'sulfanyl', not 'mercapto'."""
        assert get_sulfur_prefix("thiol") == "sulfanyl"

    def test_sulfonic_acid_prefix(self):
        """Sulfonic acid prefix is 'sulfo'."""
        assert get_sulfur_prefix("sulfonic_acid") == "sulfo"

    def test_sulfide_no_prefix(self):
        """Sulfides use functional class naming, no prefix."""
        assert get_sulfur_prefix("sulfide") is None
        assert get_sulfur_prefix("thioether") is None

    def test_sulfoxide_no_prefix(self):
        """Sulfoxides use functional class naming, no prefix."""
        assert get_sulfur_prefix("sulfoxide") is None

    def test_sulfone_no_prefix(self):
        """Sulfones use functional class naming, no prefix."""
        assert get_sulfur_prefix("sulfone") is None


class TestAlkylCounting:
    """Tests for alkyl carbon counting helper."""

    def test_methyl(self):
        """Single carbon = 1."""
        mol = Chem.MolFromSmiles("CSC")
        count = _count_alkyl_carbons(mol, 0, {1})  # Start at C, exclude S
        assert count == 1

    def test_ethyl(self):
        """Two carbons = 2."""
        mol = Chem.MolFromSmiles("CCSC")
        count = _count_alkyl_carbons(mol, 0, {2})  # Start at first C, exclude S
        assert count == 2

    def test_propyl(self):
        """Three carbons = 3."""
        mol = Chem.MolFromSmiles("CCCSC")
        count = _count_alkyl_carbons(mol, 0, {3})
        assert count == 3

    def test_butyl(self):
        """Four carbons = 4."""
        mol = Chem.MolFromSmiles("CCCCSC")
        count = _count_alkyl_carbons(mol, 0, {4})
        assert count == 4

    def test_pentyl(self):
        """Five carbons = 5."""
        mol = Chem.MolFromSmiles("CCCCCSC")
        count = _count_alkyl_carbons(mol, 0, {5})
        assert count == 5


class TestAdditionalSulfurNaming:
    """Additional tests for edge cases and coverage."""

    def test_dipropyl_sulfide(self):
        """CCCSCC -> dipropyl sulfide"""
        mol = Chem.MolFromSmiles("CCCSCC")
        result = name_sulfide(mol, 3)
        # This is ethyl propyl sulfide, not dipropyl
        assert result == "ethyl propyl sulfide"

    def test_actual_dipropyl_sulfide(self):
        """CCCSCCC -> dipropyl sulfide"""
        mol = Chem.MolFromSmiles("CCCSCCC")
        result = name_sulfide(mol, 3)
        assert result == "dipropyl sulfide"

    def test_dibutyl_sulfide(self):
        """CCCCSCCCC -> dibutyl sulfide"""
        mol = Chem.MolFromSmiles("CCCCSCCCC")
        result = name_sulfide(mol, 4)
        assert result == "dibutyl sulfide"

    def test_methyl_butyl_sulfide(self):
        """CCCCSC -> butyl methyl sulfide"""
        mol = Chem.MolFromSmiles("CCCCSC")
        result = name_sulfide(mol, 4)
        assert result == "butyl methyl sulfide"

    def test_dipropyl_sulfoxide(self):
        """CCCS(=O)CCC -> dipropyl sulfoxide"""
        mol = Chem.MolFromSmiles("CCCS(=O)CCC")
        # Find the sulfur atom
        sulfur_idx = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S':
                sulfur_idx = atom.GetIdx()
                break
        result = name_sulfoxide(mol, (sulfur_idx,))
        assert result == "dipropyl sulfoxide"

    def test_dipropyl_sulfone(self):
        """CCCS(=O)(=O)CCC -> dipropyl sulfone"""
        mol = Chem.MolFromSmiles("CCCS(=O)(=O)CCC")
        sulfur_idx = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S':
                sulfur_idx = atom.GetIdx()
                break
        result = name_sulfone(mol, (sulfur_idx,))
        assert result == "dipropyl sulfone"
