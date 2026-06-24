"""Tests for anhydride naming (19-02).

Tests functional class naming for anhydrides following IUPAC conventions:
- Symmetric acyclic: acetic anhydride (retained acid name, PIN per P-65.1.1.1)
- Mixed/asymmetric: acetic propanoic anhydride (alphabetical)
- Cyclic (from diacids): butanedioic anhydride
- Consumed-atom filtering: anhydride atoms removed from ester FG detection
"""
import pytest
from orthonym import name_compound


class TestSymmetricAnhydrides:
    """Test symmetric acyclic anhydride naming."""

    def test_ethanoic_anhydride(self):
        """Acetic anhydride - simplest symmetric (retained acid PIN P-65.1.1.1)."""
        result = name_compound("CC(=O)OC(=O)C")
        assert result == "acetic anhydride"

    def test_propanoic_anhydride(self):
        """Propanoic anhydride - C3 symmetric."""
        result = name_compound("CCC(=O)OC(=O)CC")
        assert result == "propanoic anhydride"

    def test_butanoic_anhydride(self):
        """Butanoic anhydride - C4 symmetric."""
        result = name_compound("CCCC(=O)OC(=O)CCC")
        assert result == "butanoic anhydride"

    def test_methanoic_anhydride(self):
        """Formic anhydride - C1 symmetric (retained acid PIN P-65.1.1.1)."""
        result = name_compound("O=COC=O")
        assert result == "formic anhydride"


class TestMixedAnhydrides:
    """Test mixed/asymmetric acyclic anhydride naming."""

    def test_ethanoic_propanoic_anhydride(self):
        """Acetic propanoic anhydride - alphabetical order (acetic retained)."""
        result = name_compound("CC(=O)OC(=O)CC")
        assert result == "acetic propanoic anhydride"

    def test_butanoic_propanoic_anhydride(self):
        """Butanoic propanoic anhydride - mixed C4+C3."""
        result = name_compound("CCC(=O)OC(=O)CCC")
        assert result == "butanoic propanoic anhydride"


class TestCyclicAnhydrides:
    """Test cyclic anhydride naming (from dicarboxylic acids)."""

    def test_butanedioic_anhydride(self):
        """Butanedioic anhydride (succinic anhydride) - 5-membered ring."""
        result = name_compound("O=C1CCC(=O)O1")
        assert result == "butanedioic anhydride"

    def test_pentanedioic_anhydride(self):
        """Pentanedioic anhydride (glutaric anhydride) - 6-membered ring."""
        result = name_compound("O=C1CCCC(=O)O1")
        assert result == "pentanedioic anhydride"

    def test_cyclic_anhydride_not_lactone(self):
        """Cyclic anhydrides should NOT be named as lactones."""
        # O=C1CCC(=O)O1 has two C=O in ring - it's an anhydride, not a lactone
        result = name_compound("O=C1CCC(=O)O1")
        assert "anhydride" in result
        assert "one" not in result  # Not oxolan-2-one


class TestConsumedAtomFilteringAnhydride:
    """Test that anhydride consumed-atom filtering removes ester matches."""

    def test_anhydride_filters_ester(self):
        """Anhydride atoms should be removed from ester FG detection."""
        from rdkit import Chem
        from orthonym.perception.functional_groups import detect_functional_groups
        from orthonym.namer import _filter_consumed_fg_atoms

        mol = Chem.MolFromSmiles("CC(=O)OC(=O)C")
        fgs = detect_functional_groups(mol)
        assert "ester" in fgs  # Before filtering
        assert "anhydride" in fgs

        filtered = _filter_consumed_fg_atoms(fgs)
        assert "ester" not in filtered  # After filtering, ester removed
        assert "anhydride" in filtered  # Anhydride still present
