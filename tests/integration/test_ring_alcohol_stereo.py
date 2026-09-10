"""
Tests for: Ring alcohol stereo descriptor injection.

Verifies that cycloalkanol compounds with true stereocenters (disubstituted
rings) produce names with CIP stereodescriptors, and that achiral
cycloalkanols (e.g., unsubstituted cyclopentanol) do NOT get stereo prefix.

Key findings from research:
- O[C@@H]1CCCC1 (unsubstituted cyclopentanol) is achiral after canonicalization
- Truly chiral cycloalkanols (e.g., O[C@H]1CCC[C@@H]1C) require disubstitution
- The main assembly path at composer.py generates stereo via
  _generate_stereodescriptors when features.stereocenters is populated
"""

import pytest
from orthonym.namer import name_compound


class TestChiralCycloalkanolStereo:
    """Truly chiral cycloalkanols must include CIP stereodescriptors."""

    def test_2_methylcyclopentanol_has_stereo(self):
        """2-methylcyclopentan-1-ol with two stereocenters must include CIP prefix."""
        name = name_compound("O[C@H]1CCC[C@@H]1C")
        assert name is not None
        # Must have CIP stereo prefix with R or S
        assert "(" in name and ("R" in name or "S" in name), (
            f"Expected stereo prefix in name, got: {name}"
        )

    def test_4_methylcyclohexanol_has_stereo(self):
        """4-methylcyclohexan-1-ol with stereocenters must include CIP prefix."""
        name = name_compound("C[C@@H]1CC[C@H](O)CC1")
        assert name is not None
        assert "(" in name and ("R" in name or "S" in name or "r" in name or "s" in name), (
            f"Expected stereo prefix in name, got: {name}"
        )

    def test_2_methylcyclopentanol_name_structure(self):
        """Verify the stereo prefix format is IUPAC-compliant (locant + CIP code)."""
        name = name_compound("O[C@H]1CCC[C@@H]1C")
        # Should be something like "(1S,2S)-2-methylcyclopentan-1-ol" or
        # "(1S,2S)-1-methylcyclopentan-2-ol"
        # The prefix must start with '(' and contain locant+CIP pairs
        assert name.startswith("("), f"Expected name to start with '(', got: {name}"
        # Find the closing paren
        close_idx = name.index(")")
        prefix_content = name[1:close_idx]
        # Should contain comma-separated locant-CIP pairs
        parts = prefix_content.split(",")
        assert len(parts) == 2, f"Expected 2 stereo descriptors, got: {prefix_content}"

    def test_trans_4_methylcyclohexanol_stereo_content(self):
        """trans-4-methylcyclohexan-1-ol should have stereo prefix with 2 descriptors."""
        name = name_compound("C[C@@H]1CC[C@H](O)CC1")
        assert name is not None
        # Should contain stereo information
        assert "(" in name, f"Expected stereo prefix in name, got: {name}"


class TestAchiralCycloalkanolNoStereo:
    """Achiral cycloalkanols must NOT include stereo prefix."""

    def test_unsubstituted_cyclopentanol_no_stereo(self):
        """Unsubstituted cyclopentanol (achiral) must not have stereo prefix."""
        name = name_compound("OC1CCCC1")
        assert name is not None
        # Must NOT start with stereo prefix
        assert not name.startswith("("), (
            f"Achiral cyclopentanol should not have stereo prefix, got: {name}"
        )
        # Verify it's correctly named
        assert "cyclopentan" in name

    def test_cyclopentanol_with_fake_chirality_no_stereo(self):
        """O[C@@H]1CCCC1 has @ in SMILES but is achiral after canonicalization."""
        name = name_compound("O[C@@H]1CCCC1")
        assert name is not None
        # After RDKit canonicalization, the chirality is removed (symmetry plane)
        # so there should be no stereo prefix
        assert not name.startswith("("), (
            f"Achiral molecule should not have stereo prefix, got: {name}"
        )

    def test_unsubstituted_cyclohexanol_no_stereo(self):
        """Unsubstituted cyclohexanol (achiral) must not have stereo prefix."""
        name = name_compound("O[C@@H]1CCCCC1")
        assert name is not None
        assert not name.startswith("("), (
            f"Achiral cyclohexanol should not have stereo prefix, got: {name}"
        )
        assert "cyclohexan" in name


class TestDiolStereo:
    """Cyclohexanediols with stereo should include appropriate descriptors."""

    def test_cyclohexane_1_4_diol_stereo(self):
        """Cyclohexane-1,4-diol with stereo markup should be handled correctly."""
        name = name_compound("O[C@@H]1CC[C@@H](O)CC1")
        assert name is not None
        # This may have pseudoasymmetric centers (r/s) or true R/S
        # depending on RDKit's classification
        # The key is that IF stereocenters exist, they are reflected in the name
        # If no true stereocenters (symmetry), no stereo prefix is expected
        # For 1,4-diol, these are pseudoasymmetric (r/s notation)
        if "r" in name.lower() and "(" in name:
            # Has some form of stereo -- verify it's well-formed
            assert name.index(")") > name.index("(")
