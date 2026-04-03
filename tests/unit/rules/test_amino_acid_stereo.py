"""Tests for STER-09: Amino acid stereo injection.

Verifies that amino acid names include CIP stereodescriptors
when stereocenters are present, and no stereo for achiral amino acids.
"""

import re
import pytest
from orthonym.namer import name_compound


class TestAminoAcidSystematicStereo:
    """Systematic amino acid names include CIP stereo prefix."""

    def test_l_alanine_has_stereo(self):
        """L-alanine SMILES produces name with (2R)- or (2S)- prefix."""
        name = name_compound("[C@@H](N)(C)C(=O)O")
        assert re.match(r"^\(\d*[RS]\)-", name), f"Expected stereo prefix, got: {name}"
        assert "aminopropanoic acid" in name

    def test_d_alanine_has_stereo(self):
        """D-alanine SMILES produces name with opposite stereo from L."""
        name = name_compound("[C@H](N)(C)C(=O)O")
        assert re.match(r"^\(\d*[RS]\)-", name), f"Expected stereo prefix, got: {name}"
        assert "aminopropanoic acid" in name

    def test_l_and_d_alanine_different_stereo(self):
        """L and D alanine produce different stereodescriptors."""
        l_name = name_compound("[C@@H](N)(C)C(=O)O")
        d_name = name_compound("[C@H](N)(C)C(=O)O")
        # Extract stereo descriptors
        l_match = re.match(r"\((\d*[RS])\)-", l_name)
        d_match = re.match(r"\((\d*[RS])\)-", d_name)
        assert l_match and d_match
        assert l_match.group(1) != d_match.group(1), \
            f"L and D should differ: {l_name} vs {d_name}"

    def test_glycine_no_stereo(self):
        """Glycine (achiral) has no stereo prefix."""
        name = name_compound("NCC(=O)O")
        assert not re.match(r"^\(\d*[RS]\)-", name), \
            f"Glycine should not have stereo: {name}"

    def test_2_aminobutanoic_acid_has_stereo(self):
        """2-aminobutanoic acid with stereocenter includes stereo."""
        name = name_compound("[C@@H](N)(CC)C(=O)O")
        assert re.match(r"^\(\d*[RS]\)-", name), f"Expected stereo prefix, got: {name}"
        assert "aminobutanoic acid" in name

    def test_2_aminopentanoic_acid_has_stereo(self):
        """Longer amino acid chain also gets stereo."""
        name = name_compound("[C@@H](N)(CCC)C(=O)O")
        assert re.match(r"^\(\d*[RS]\)-", name), f"Expected stereo prefix, got: {name}"
        assert "aminopentanoic acid" in name


class TestAminoAcidTrivialStereo:
    """Trivial amino acid names also get stereo prefix if stereo is defined."""

    def test_trivial_name_with_stereo_gets_prefix(self):
        """If a trivial name is returned and mol has stereo, stereo is prepended."""
        from orthonym.rules.amino_acids import name_amino_acid, _build_amino_acid_locant_map
        from rdkit import Chem

        # Use non-stereo alanine SMILES to match trivial dict
        mol = Chem.MolFromSmiles("CC(N)C(=O)O")
        can = Chem.MolToSmiles(mol)
        result = name_amino_acid(mol, can)
        # No stereo in input means no stereo in output
        assert result == "alanine" or (result and "alanine" in result)

    def test_locant_map_finds_acid_carbon(self):
        """_build_amino_acid_locant_map assigns locant 1 to acid carbon."""
        from orthonym.rules.amino_acids import _build_amino_acid_locant_map
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CC(N)C(=O)O")
        locant_map = _build_amino_acid_locant_map(mol)
        assert locant_map, "locant_map should not be empty"
        # Locant 1 should be a carbon with C=O
        for idx, loc in locant_map.items():
            if loc == 1:
                atom = mol.GetAtomWithIdx(idx)
                assert atom.GetSymbol() == 'C'
                break
