"""Tests for cyclic imide retained names (IUPAC P-31.1.3.4).

Succinimide and maleimide are retained names for common cyclic imides.
They should be returned by the retained name lookup before systematic naming.
Phthalimide (isoindoline-1,3-dione) already has a correct systematic name.
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_compound
from orthonym.data.retained_names import get_retained_name


@pytest.mark.unit
class TestCyclicImideRetainedNames:
    """Retained name lookup tests for cyclic imides."""

    def test_succinimide_retained_name_lookup(self):
        """Canonical SMILES for succinimide should be in retained names."""
        canonical = Chem.CanonSmiles("O=C1CCC(=O)N1")
        assert get_retained_name(canonical) == "succinimide"

    def test_maleimide_retained_name_lookup(self):
        """Canonical SMILES for maleimide should be in retained names."""
        canonical = Chem.CanonSmiles("O=C1C=CC(=O)N1")
        assert get_retained_name(canonical) == "maleimide"

    def test_phthalimide_not_in_retained_names(self):
        """Phthalimide should NOT be added as retained name (systematic name is correct)."""
        canonical = Chem.CanonSmiles("O=C1NC(=O)c2ccccc21")
        assert get_retained_name(canonical) is None


@pytest.mark.unit
class TestCyclicImideNaming:
    """End-to-end naming tests for cyclic imides."""

    def test_succinimide(self):
        """O=C1CCC(=O)N1 -> succinimide."""
        assert name_compound("O=C1CCC(=O)N1") == "succinimide"

    def test_maleimide(self):
        """O=C1C=CC(=O)N1 -> maleimide."""
        assert name_compound("O=C1C=CC(=O)N1") == "maleimide"

    def test_phthalimide_systematic_name(self):
        """O=C1NC(=O)c2ccccc21 -> isoindoline-1,3-dione (no regression)."""
        assert name_compound("O=C1NC(=O)c2ccccc21") == "isoindoline-1,3-dione"

    def test_n_methyl_succinimide_not_retained(self):
        """N-methylsuccinimide should NOT match the retained name.
        It has different canonical SMILES due to N-methyl substituent."""
        canonical = Chem.CanonSmiles("O=C1CCC(=O)N1C")
        # Should not return "succinimide" -- different SMILES
        result = get_retained_name(canonical)
        assert result != "succinimide"

    def test_n_methyl_succinimide_naming(self):
        """N-methylsuccinimide should get systematic name, not retained name."""
        result = name_compound("O=C1CCC(=O)N1C")
        # Should NOT be just "succinimide" -- must include the N-methyl
        assert result != "succinimide"
        # The exact systematic name may vary, but it should contain methyl
        assert "methyl" in result.lower()

    def test_succinimide_alternate_input(self):
        """Alternative SMILES input for succinimide should also work."""
        # Various equivalent representations
        assert name_compound("C1CC(=O)NC1=O") == "succinimide"

    def test_maleimide_alternate_input(self):
        """Alternative SMILES input for maleimide should also work."""
        assert name_compound("C1=CC(=O)NC1=O") == "maleimide"
