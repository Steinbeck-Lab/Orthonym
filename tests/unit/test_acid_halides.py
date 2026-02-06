"""Tests for acid halide naming (19-02).

Tests functional class naming for acid halides following IUPAC conventions:
- Retained acyl names: acetyl, formyl, benzoyl
- Systematic: propanoyl chloride, butanoyl bromide
- Diacid halides: pentanedioyl dichloride
- Ring-attached: benzoyl chloride
- Consumed-atom filtering: no double-counting of halogens
"""
import pytest
from orthonym import name_compound


class TestRetainedAcylNames:
    """Test retained acyl name usage for common acid halides."""

    def test_acetyl_chloride(self):
        """Acetyl chloride - retained acyl name for C2."""
        result = name_compound("CC(=O)Cl")
        assert result == "acetyl chloride"

    def test_acetyl_bromide(self):
        """Acetyl bromide - retained acyl + bromide."""
        result = name_compound("CC(=O)Br")
        assert result == "acetyl bromide"

    def test_acetyl_fluoride(self):
        """Acetyl fluoride - retained acyl + fluoride."""
        result = name_compound("CC(=O)F")
        assert result == "acetyl fluoride"

    def test_formyl_chloride(self):
        """Formyl chloride - retained acyl name for C1."""
        result = name_compound("C(=O)Cl")
        assert result == "formyl chloride"


class TestSystematicAcidHalides:
    """Test systematic acid halide naming."""

    def test_propanoyl_chloride(self):
        """Propanoyl chloride - C3 acid chloride."""
        result = name_compound("CCC(=O)Cl")
        assert result == "propanoyl chloride"

    def test_butanoyl_chloride(self):
        """Butanoyl chloride - C4 acid chloride."""
        result = name_compound("CCCC(=O)Cl")
        assert result == "butanoyl chloride"

    def test_pentanoyl_chloride(self):
        """Pentanoyl chloride - C5 acid chloride."""
        result = name_compound("CCCCC(=O)Cl")
        assert result == "pentanoyl chloride"

    def test_hexanoyl_chloride(self):
        """Hexanoyl chloride - C6 acid chloride."""
        result = name_compound("CCCCCC(=O)Cl")
        assert result == "hexanoyl chloride"

    def test_propanoyl_bromide(self):
        """Propanoyl bromide - C3 acid bromide."""
        result = name_compound("CCC(=O)Br")
        assert result == "propanoyl bromide"

    def test_propanoyl_fluoride(self):
        """Propanoyl fluoride - C3 acid fluoride."""
        result = name_compound("CCC(=O)F")
        assert result == "propanoyl fluoride"


class TestRingAttachedAcidHalides:
    """Test ring-attached acid halide naming."""

    def test_benzoyl_chloride(self):
        """Benzoyl chloride - benzene-attached acid chloride."""
        result = name_compound("O=C(Cl)c1ccccc1")
        assert result == "benzoyl chloride"

    def test_benzoyl_bromide(self):
        """Benzoyl bromide - benzene-attached acid bromide."""
        result = name_compound("O=C(Br)c1ccccc1")
        assert result == "benzoyl bromide"


class TestDiacidHalides:
    """Test diacid halide naming (both chain ends)."""

    def test_pentanedioyl_dichloride(self):
        """Pentanedioyl dichloride - C5 diacid dichloride."""
        result = name_compound("ClC(=O)CCCC(=O)Cl")
        assert result == "pentanedioyl dichloride"

    def test_butanedioyl_dichloride(self):
        """Butanedioyl dichloride - C4 diacid dichloride."""
        result = name_compound("ClC(=O)CCC(=O)Cl")
        assert result == "butanedioyl dichloride"

    def test_hexanedioyl_dichloride(self):
        """Hexanedioyl dichloride - C6 diacid dichloride."""
        result = name_compound("ClC(=O)CCCCC(=O)Cl")
        assert result == "hexanedioyl dichloride"


class TestSubstitutedAcidHalides:
    """Test acid halides with non-acid-halide substituents."""

    def test_3_chlorobutanoyl_chloride(self):
        """3-chlorobutanoyl chloride - substituent Cl distinct from acid halide Cl."""
        result = name_compound("CC(Cl)CC(=O)Cl")
        assert "chloro" in result and "oyl chloride" in result

    def test_no_double_counting_simple(self):
        """Acetyl chloride should NOT contain 'chloro' prefix."""
        result = name_compound("CC(=O)Cl")
        assert "chloro" not in result
        assert result == "acetyl chloride"

    def test_no_double_counting_systematic(self):
        """Propanoyl chloride should NOT contain 'chloro' prefix."""
        result = name_compound("CCC(=O)Cl")
        assert "chloro" not in result
        assert result == "propanoyl chloride"


class TestConsumedAtomFiltering:
    """Test that consumed-atom filtering works correctly."""

    def test_acid_chloride_filters_chloro(self):
        """Acid chloride Cl should be removed from chloro FG detection."""
        from rdkit import Chem
        from orthonym.perception.functional_groups import detect_functional_groups
        from orthonym.namer import _filter_consumed_fg_atoms

        mol = Chem.MolFromSmiles("CC(=O)Cl")
        fgs = detect_functional_groups(mol)
        assert "chloro" in fgs  # Before filtering, chloro is detected

        filtered = _filter_consumed_fg_atoms(fgs)
        assert "chloro" not in filtered  # After filtering, chloro is removed
        assert "acid_chloride" in filtered  # acid_chloride still present

    def test_non_acid_halide_chloro_preserved(self):
        """Chloro substituents NOT part of acid halide should be preserved."""
        from rdkit import Chem
        from orthonym.perception.functional_groups import detect_functional_groups
        from orthonym.namer import _filter_consumed_fg_atoms

        mol = Chem.MolFromSmiles("CC(Cl)CC(=O)Cl")
        fgs = detect_functional_groups(mol)
        # Before filtering: 2 chloro matches (acid halide Cl + substituent Cl)
        assert len(fgs.get("chloro", [])) == 2

        filtered = _filter_consumed_fg_atoms(fgs)
        # After filtering: 1 chloro match (only substituent Cl)
        assert len(filtered.get("chloro", [])) == 1

    def test_acid_bromide_filters_bromo(self):
        """Acid bromide Br should be removed from bromo FG detection."""
        from rdkit import Chem
        from orthonym.perception.functional_groups import detect_functional_groups
        from orthonym.namer import _filter_consumed_fg_atoms

        mol = Chem.MolFromSmiles("CC(=O)Br")
        fgs = detect_functional_groups(mol)
        filtered = _filter_consumed_fg_atoms(fgs)
        assert "bromo" not in filtered
        assert "acid_bromide" in filtered
