"""Tests for _name_r_group() phenyl shortcut fix (Phase 125).

Verifies that functional-class handlers correctly name substituted
aromatic R-groups instead of returning bare 'phenyl' or wrong 'benzyl'.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestSubstitutedRGroupNaming:
    """Ensure _name_r_group() handles substituted phenyl rings."""

    def test_chlorophenyl_isocyanate(self):
        result = name_compound("Clc1ccc(N=C=O)cc1")
        assert "chloro" in result, f"Expected 'chloro' in '{result}'"
        assert "isocyanate" in result, f"Expected 'isocyanate' in '{result}'"

    def test_methylphenyl_boronic_acid(self):
        result = name_compound("Cc1ccc(B(O)O)cc1")
        assert "methyl" in result, f"Expected 'methyl' in '{result}'"
        assert "boronic acid" in result, f"Expected 'boronic acid' in '{result}'"
        assert "benzyl" not in result, f"Should not be 'benzyl' in '{result}'"

    def test_hydroxyphenyl_isocyanate(self):
        result = name_compound("Oc1ccc(N=C=O)cc1")
        assert "hydroxy" in result, f"Expected 'hydroxy' in '{result}'"

    def test_unsubstituted_phenyl_isocyanate_unchanged(self):
        result = name_compound("c1ccc(N=C=O)cc1")
        assert "phenyl isocyanate" == result, f"Expected 'phenyl isocyanate', got '{result}'"

    def test_unsubstituted_phenylboronic_acid_unchanged(self):
        result = name_compound("c1ccc(B(O)O)cc1")
        assert "phenylboronic acid" in result, f"Expected 'phenylboronic acid' in '{result}'"

    def test_substituted_carbamate(self):
        result = name_compound("O=C(OC)Nc1ccc(Cl)cc1")
        assert "chloro" in result, f"Expected 'chloro' in '{result}'"

    def test_substituted_urea(self):
        result = name_compound("O=C(NC)Nc1ccc(C)cc1")
        assert "methyl" in result, f"Expected 'methyl' in '{result}'"
