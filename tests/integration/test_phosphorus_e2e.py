"""End-to-end tests for phosphorus compound naming."""

import pytest
from src.orthonym import name_compound


class TestPhosphineE2E:
    """E2E tests for phosphine (phosphane) naming (PHOSPH-01, PHOSPH-02)."""

    def test_methylphosphane(self):
        """CP -> methylphosphane (PHOSPH-01)"""
        assert name_compound("CP") == "methylphosphane"

    def test_ethylphosphane(self):
        """CCP -> ethylphosphane"""
        assert name_compound("CCP") == "ethylphosphane"

    def test_trimethylphosphane(self):
        """CP(C)C -> trimethylphosphane (PHOSPH-02)"""
        assert name_compound("CP(C)C") == "trimethylphosphane"

    def test_triethylphosphane(self):
        """CCP(CC)CC -> triethylphosphane"""
        assert name_compound("CCP(CC)CC") == "triethylphosphane"

    def test_dimethylphosphane(self):
        """CPC -> dimethylphosphane"""
        assert name_compound("CPC") == "dimethylphosphane"


class TestPhosphineOxideE2E:
    """E2E tests for phosphine oxide naming (PHOSPH-03)."""

    def test_trimethylphosphane_oxide(self):
        """CP(C)(C)=O -> trimethylphosphane oxide (PHOSPH-03)"""
        assert name_compound("CP(C)(C)=O") == "trimethylphosphane oxide"

    def test_triethylphosphane_oxide(self):
        """CCP(CC)(CC)=O -> triethylphosphane oxide"""
        assert name_compound("CCP(CC)(CC)=O") == "triethylphosphane oxide"


class TestPhosphonicAcidE2E:
    """E2E tests for phosphonic acid naming (PHOSPH-04)."""

    def test_methanephosphonic_acid(self):
        """CP(=O)(O)O -> methanephosphonic acid (PHOSPH-04)"""
        assert name_compound("CP(=O)(O)O") == "methanephosphonic acid"

    def test_ethanephosphonic_acid(self):
        """CCP(=O)(O)O -> ethanephosphonic acid"""
        assert name_compound("CCP(=O)(O)O") == "ethanephosphonic acid"

    def test_phenylphosphonic_acid(self):
        """c1ccccc1P(=O)(O)O -> phenylphosphonic acid"""
        assert name_compound("c1ccccc1P(=O)(O)O") == "phenylphosphonic acid"


class TestPhosphinicAcidE2E:
    """E2E tests for phosphinic acid naming (PHOSPH-05)."""

    def test_dimethylphosphinic_acid(self):
        """CP(C)(=O)O -> dimethylphosphinic acid (PHOSPH-05)"""
        assert name_compound("CP(C)(=O)O") == "dimethylphosphinic acid"

    def test_diethylphosphinic_acid(self):
        """CCP(CC)(=O)O -> diethylphosphinic acid"""
        assert name_compound("CCP(CC)(=O)O") == "diethylphosphinic acid"

    def test_ethylmethylphosphinic_acid(self):
        """CCP(C)(=O)O -> ethylmethylphosphinic acid"""
        assert name_compound("CCP(C)(=O)O") == "ethylmethylphosphinic acid"


class TestPhosphateEsterE2E:
    """E2E tests for phosphate ester naming (PHOSPH-06)."""

    def test_methyl_phosphate(self):
        """COP(=O)(O)O -> methyl phosphate (PHOSPH-06)"""
        assert name_compound("COP(=O)(O)O") == "methyl phosphate"

    def test_dimethyl_phosphate(self):
        """COP(=O)(OC)O -> dimethyl phosphate"""
        assert name_compound("COP(=O)(OC)O") == "dimethyl phosphate"

    def test_trimethyl_phosphate(self):
        """COP(=O)(OC)OC -> trimethyl phosphate"""
        assert name_compound("COP(=O)(OC)OC") == "trimethyl phosphate"

    def test_ethyl_phosphate(self):
        """CCOP(=O)(O)O -> ethyl phosphate"""
        assert name_compound("CCOP(=O)(O)O") == "ethyl phosphate"

    def test_diethyl_phosphate(self):
        """CCOP(=O)(OCC)O -> diethyl phosphate"""
        assert name_compound("CCOP(=O)(OCC)O") == "diethyl phosphate"

    def test_triethyl_phosphate(self):
        """CCOP(=O)(OCC)OCC -> triethyl phosphate"""
        assert name_compound("CCOP(=O)(OCC)OCC") == "triethyl phosphate"


class TestPhosphorusRetainedNames:
    """E2E tests for phosphorus retained names."""

    def test_triphenylphosphane(self):
        """Triphenylphosphane retained name."""
        result = name_compound("c1ccccc1P(c2ccccc2)c3ccccc3")
        assert result == "triphenylphosphane"

    def test_triphenylphosphane_oxide(self):
        """Triphenylphosphane oxide retained name."""
        result = name_compound("O=P(c1ccccc1)(c2ccccc2)c3ccccc3")
        assert result == "triphenylphosphane oxide"


class TestPhosphorusEdgeCases:
    """Edge case tests for phosphorus naming."""

    def test_primary_phosphine(self):
        """Primary phosphine with single substituent."""
        result = name_compound("CP")
        assert result == "methylphosphane"

    def test_asymmetric_phosphate(self):
        """Asymmetric phosphate diester."""
        result = name_compound("COP(=O)(OCC)O")
        assert result == "ethyl methyl phosphate"

    def test_dipropylphosphinic_acid(self):
        """Dipropylphosphinic acid."""
        result = name_compound("CCCP(CCC)(=O)O")
        assert result == "dipropylphosphinic acid"


class TestPhosphorusRequirements:
    """Tests verifying all PHOSPH requirements are met."""

    def test_phosph_01_methylphosphane(self):
        """PHOSPH-01: name_compound('CP') returns 'methylphosphane'"""
        assert name_compound("CP") == "methylphosphane"

    def test_phosph_02_trimethylphosphane(self):
        """PHOSPH-02: name_compound('CP(C)C') returns 'trimethylphosphane'"""
        assert name_compound("CP(C)C") == "trimethylphosphane"

    def test_phosph_03_trimethylphosphane_oxide(self):
        """PHOSPH-03: name_compound('CP(C)(C)=O') returns 'trimethylphosphane oxide'"""
        assert name_compound("CP(C)(C)=O") == "trimethylphosphane oxide"

    def test_phosph_04_methanephosphonic_acid(self):
        """PHOSPH-04: name_compound('CP(=O)(O)O') returns 'methanephosphonic acid'"""
        assert name_compound("CP(=O)(O)O") == "methanephosphonic acid"

    def test_phosph_05_dimethylphosphinic_acid(self):
        """PHOSPH-05: name_compound('CP(C)(=O)O') returns 'dimethylphosphinic acid'"""
        assert name_compound("CP(C)(=O)O") == "dimethylphosphinic acid"

    def test_phosph_06_methyl_phosphate(self):
        """PHOSPH-06: name_compound('COP(=O)(O)O') returns 'methyl phosphate'"""
        assert name_compound("COP(=O)(O)O") == "methyl phosphate"
