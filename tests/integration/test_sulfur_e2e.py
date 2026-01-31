"""End-to-end tests for sulfur compound naming."""

import pytest
from src.orthonym import name_compound


class TestThiolE2E:
    """E2E tests for thiol naming (SULFUR-01)."""

    def test_methanethiol(self):
        """CS -> methanethiol"""
        assert name_compound("CS") == "methanethiol"

    def test_ethanethiol(self):
        """CCS -> ethanethiol"""
        assert name_compound("CCS") == "ethanethiol"

    def test_propane_1_thiol(self):
        """CCCS -> propane-1-thiol or propan-1-thiol"""
        result = name_compound("CCCS")
        # Accept both PIN forms: propan-1-thiol (no 'e' before hyphen) is preferred
        assert result in ("propane-1-thiol", "propan-1-thiol", "propanethiol")

    def test_propane_2_thiol(self):
        """CC(S)C -> propane-2-thiol or propan-2-thiol"""
        result = name_compound("CC(S)C")
        # Accept both PIN forms
        assert result in ("propane-2-thiol", "propan-2-thiol")


class TestSulfideE2E:
    """E2E tests for sulfide naming (SULFUR-02)."""

    def test_dimethyl_sulfide(self):
        """CSC -> dimethyl sulfide"""
        assert name_compound("CSC") == "dimethyl sulfide"

    def test_diethyl_sulfide(self):
        """CCSCC -> diethyl sulfide"""
        assert name_compound("CCSCC") == "diethyl sulfide"

    def test_ethyl_methyl_sulfide(self):
        """CCSC -> ethyl methyl sulfide"""
        assert name_compound("CCSC") == "ethyl methyl sulfide"

    def test_methyl_propyl_sulfide(self):
        """CCCSC -> methyl propyl sulfide"""
        assert name_compound("CCCSC") == "methyl propyl sulfide"


class TestSulfoxideE2E:
    """E2E tests for sulfoxide naming (SULFUR-03)."""

    def test_dimethyl_sulfoxide(self):
        """CS(=O)C -> dimethyl sulfoxide (DMSO)"""
        assert name_compound("CS(=O)C") == "dimethyl sulfoxide"

    def test_diethyl_sulfoxide(self):
        """CCS(=O)CC -> diethyl sulfoxide"""
        assert name_compound("CCS(=O)CC") == "diethyl sulfoxide"

    def test_ethyl_methyl_sulfoxide(self):
        """CCS(=O)C -> ethyl methyl sulfoxide"""
        assert name_compound("CCS(=O)C") == "ethyl methyl sulfoxide"


class TestSulfoneE2E:
    """E2E tests for sulfone naming (SULFUR-04)."""

    def test_dimethyl_sulfone(self):
        """CS(=O)(=O)C -> dimethyl sulfone"""
        assert name_compound("CS(=O)(=O)C") == "dimethyl sulfone"

    def test_diethyl_sulfone(self):
        """CCS(=O)(=O)CC -> diethyl sulfone"""
        assert name_compound("CCS(=O)(=O)CC") == "diethyl sulfone"

    def test_ethyl_methyl_sulfone(self):
        """CCS(=O)(=O)C -> ethyl methyl sulfone"""
        assert name_compound("CCS(=O)(=O)C") == "ethyl methyl sulfone"


class TestSulfonicAcidE2E:
    """E2E tests for sulfonic acid naming (SULFUR-05)."""

    def test_methanesulfonic_acid(self):
        """CS(=O)(=O)O -> methanesulfonic acid"""
        assert name_compound("CS(=O)(=O)O") == "methanesulfonic acid"

    def test_ethanesulfonic_acid(self):
        """CCS(=O)(=O)O -> ethanesulfonic acid"""
        assert name_compound("CCS(=O)(=O)O") == "ethanesulfonic acid"

    def test_benzenesulfonic_acid(self):
        """c1ccccc1S(=O)(=O)O -> benzenesulfonic acid"""
        assert name_compound("c1ccccc1S(=O)(=O)O") == "benzenesulfonic acid"


class TestSulfurRetainedNames:
    """E2E tests for sulfur retained names."""

    def test_dmso_retained(self):
        """DMSO should return retained name."""
        # Multiple SMILES forms should work
        result = name_compound("CS(C)=O")
        assert result == "dimethyl sulfoxide"

    def test_methanesulfonic_retained(self):
        """Common sulfonic acid retained name."""
        result = name_compound("CS(=O)(=O)O")
        assert result == "methanesulfonic acid"

    def test_dimethyl_sulfide_retained(self):
        """CSC should return retained name."""
        result = name_compound("CSC")
        assert result == "dimethyl sulfide"

    def test_diethyl_sulfide_retained(self):
        """CCSCC should return retained name."""
        result = name_compound("CCSCC")
        assert result == "diethyl sulfide"


class TestSulfurAdditionalCompounds:
    """Additional E2E tests for sulfur compounds."""

    def test_dipropyl_sulfide(self):
        """CCCSCCC -> dipropyl sulfide"""
        assert name_compound("CCCSCCC") == "dipropyl sulfide"

    def test_dibutyl_sulfide(self):
        """CCCCSCCCC -> dibutyl sulfide"""
        assert name_compound("CCCCSCCCC") == "dibutyl sulfide"

    def test_dipropyl_sulfoxide(self):
        """CCCS(=O)CCC -> dipropyl sulfoxide"""
        assert name_compound("CCCS(=O)CCC") == "dipropyl sulfoxide"

    def test_dipropyl_sulfone(self):
        """CCCS(=O)(=O)CCC -> dipropyl sulfone"""
        assert name_compound("CCCS(=O)(=O)CCC") == "dipropyl sulfone"

    def test_butyl_methyl_sulfide(self):
        """CCCCSC -> butyl methyl sulfide"""
        assert name_compound("CCCCSC") == "butyl methyl sulfide"

    def test_butyl_methyl_sulfoxide(self):
        """CCCCS(=O)C -> butyl methyl sulfoxide"""
        assert name_compound("CCCCS(=O)C") == "butyl methyl sulfoxide"

    def test_butyl_methyl_sulfone(self):
        """CCCCS(=O)(=O)C -> butyl methyl sulfone"""
        assert name_compound("CCCCS(=O)(=O)C") == "butyl methyl sulfone"
