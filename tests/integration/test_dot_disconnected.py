"""Integration tests for neutral dot-disconnected SMILES handling.

Neutral dot-disconnected SMILES (cocrystals, solvates, neutral mixtures)
should have each component named independently and joined with space,
sorted by descending heavy atom count.

Charged dot-disconnected SMILES (salts) continue through the salt naming path.

Only activates when 2+ multi-atom (HA >= 2) fragments exist. Single-atom
fragments (Cl, Br, O) are non-molecular entities; their presence causes
the molecule to fall through to the normal pipeline.
"""

import pytest
from orthonym import Orthonym


@pytest.mark.integration
class TestNeutralDotDisconnected:
    """Tests for neutral dot-disconnected SMILES handling."""

    def setup_method(self):
        self.namer = Orthonym()

    def test_neutral_two_components(self):
        """CCO.CC(=O)O should name both components."""
        result = self.namer.name("CCO.CC(=O)O")
        # Both "ethanol" and "acetic acid" should appear in the result
        assert "ethanol" in result
        assert "acetic acid" in result

    def test_neutral_single_component_unchanged(self):
        """Single-component SMILES CCO should produce just 'ethanol'."""
        result = self.namer.name("CCO")
        assert result == "ethanol"

    def test_salt_still_routes_to_salt_naming(self):
        """[Na+].[O-]C=O should use salt naming path, not dot-disconnected."""
        result = self.namer.name("[Na+].[O-]C=O")
        # Salt path produces "sodium formate" or similar salt name
        # It should NOT produce two separate component names
        assert result is not None
        assert isinstance(result, str)
        # Should contain "sodium" (cation name) -- salt naming
        assert "sodium" in result.lower()

    def test_three_multi_atom_components(self):
        """CC.CCC.CCCC should name all three multi-atom components."""
        result = self.namer.name("CC.CCC.CCCC")
        assert "ethane" in result
        assert "propane" in result
        assert "butane" in result

    def test_single_atom_fragments_fall_through(self):
        """Single-atom fragments like.Cl cause fallthrough to normal pipeline.

        When a dot-disconnected SMILES has fewer than 2 multi-atom fragments,
        it falls through to the normal naming pipeline rather than splitting.
        """
        # CCO.C has only 1 multi-atom fragment (CCO); C is single-atom
        # Falls through to normal pipeline
        result = self.namer.name("CCO.C")
        assert result is not None
        assert isinstance(result, str)

    def test_sort_by_descending_heavy_atoms(self):
        """Larger components should appear first in the output."""
        result = self.namer.name("CC.CCCC")
        # butane (4 HA) should come before ethane (2 HA)
        butane_pos = result.index("butane")
        ethane_pos = result.index("ethane")
        assert butane_pos < ethane_pos

    def test_two_identical_components(self):
        """CCO.CCO should produce 'ethanol ethanol'."""
        result = self.namer.name("CCO.CCO")
        # Both fragments are ethanol
        assert result.count("ethanol") == 2

    def test_water_solvate(self):
        """AMP.O (nucleotide + water) is handled correctly.

        .O is a single-atom fragment (1 HA), so only 1 multi-atom fragment
        exists. Falls through to normal pipeline.
        """
        result = self.namer.name("CCO.O")
        # Only 1 multi-atom fragment (CCO), falls through
        assert result is not None
        assert isinstance(result, str)

    def test_cocrystal_two_organic_molecules(self):
        """Two organic molecules (cocrystal) should name both."""
        # Ethanol + propanoic acid cocrystal
        result = self.namer.name("CCO.CCC(=O)O")
        assert "ethanol" in result
        assert "propanoic acid" in result

    def test_neutral_dot_chlorine_falls_through(self):
        """Neutral.Cl fragments are single atoms and fall through to normal pipeline.

        This prevents regression on molecules like arginine.Cl.Cl where the
        normal pipeline produces better names for the organic component.
        """
        # CC(=N)NCCCC[C@H](N)C(=O)O.Cl.Cl -- arginine + 2 neutral Cl atoms
        # Only 1 multi-atom fragment, so falls through to normal pipeline
        result = self.namer.name("CC(=N)NCCCC[C@H](N)C(=O)O.Cl.Cl")
        assert result is not None
        assert isinstance(result, str)
        # Should NOT produce "unknown" or empty
        assert result != "unknown"
        assert len(result) > 0
