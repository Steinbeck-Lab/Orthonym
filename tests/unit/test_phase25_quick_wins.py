"""Tests for a phase quick-win naming fixes."""
import pytest
from orthonym import name_compound


class TestQW01MultiplierPrefixFix:
    """: get_multiplier_prefix missing-arg fixes."""

    def test_biphenyl_ether_no_typeerror(self):
        """Biphenyl ether should produce valid name, not TypeError."""
        smiles = "COc1cc(-c2ccccc2)ccc1"
        result = name_compound(smiles)
        assert result is not None
        assert result != "unknown"
        assert "substituent" not in result.lower()

    def test_biphenyl_methoxy_contains_methoxy(self):
        """Methoxy on biphenyl should appear in name."""
        smiles = "COc1ccc(-c2ccccc2)cc1"
        result = name_compound(smiles)
        assert result is not None
        assert result != "unknown"
        # Should contain methoxy or similar ether prefix
        assert "methoxy" in result.lower() or "oxy" in result.lower()

    def test_carbamate_multiplier_no_crash(self):
        """Carbamate with multiple N-substituents should not crash."""
        # N,N-diethyl carbamate
        smiles = "CCN(CC)C(=O)OC"
        result = name_compound(smiles)
        assert result is not None
        assert result != "unknown"


class TestQW04PeroxynitricAcid:
    """: Peroxynitric acid retained name."""

    def test_peroxynitric_acid(self):
        """Peroxynitric acid should be named."""
        result = name_compound("O=[N+]([O-])OO")
        assert result == "peroxynitric acid"

    def test_nitric_acid(self):
        """Nitric acid should be named."""
        result = name_compound("O=[N+]([O-])O")
        assert result == "nitric acid"


class TestQW02QW03AlreadyFixed:
    """ and: Verify previously fixed items still work."""

    def test_disulfane_retained_name(self):
        """: disulfane should already work (phase 24 fix)."""
        result = name_compound("SS")
        assert result == "disulfane"

    def test_dicarboxylate_anion(self):
        """: dicarboxylate anions should produce names (phase 24 fix)."""
        result = name_compound("O=C([O-])CC=CC(=O)C(=O)[O-]")
        assert result is not None
        assert result != "unknown"
