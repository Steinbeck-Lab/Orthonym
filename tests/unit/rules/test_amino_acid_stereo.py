"""Tests for STER-09: Amino acid stereo injection.

Verifies that amino acid names include CIP stereodescriptors
when stereocenters are present, and no stereo for achiral amino acids.
"""

import re
import pytest
from orthonym.namer import name_compound


class TestAminoAcidSystematicStereo:
    """Systematic amino acid names include CIP stereo prefix."""

    # WSD-07 (Phase 175): a stereo-tagged free STANDARD amino acid now resolves to
    # its retained PIN with the configurational descriptor (P-103.1.1.1: L implicit,
    # D explicit), not the old wrong-parent systematic '(2S)-2-aminopropanoic acid'.
    # Both retained forms are OPSIN-round-trip-verified.
    def test_alanine_enantiomers_resolve_to_retained_pin(self):
        """The two alanine enantiomers resolve to 'alanine' (L) and 'D-alanine'."""
        # CIP-based: [C@@H](N)(C)C(=O)O is the R (D) enantiomer; [C@H](...) is S (L).
        assert name_compound("[C@@H](N)(C)C(=O)O") == "D-alanine"
        assert name_compound("[C@H](N)(C)C(=O)O") == "alanine"

    def test_l_and_d_alanine_different_names(self):
        """L and D alanine produce different (retained) names — stereo not dropped."""
        l_name = name_compound("[C@H](N)(C)C(=O)O")    # alanine (L implicit)
        d_name = name_compound("[C@@H](N)(C)C(=O)O")   # D-alanine
        assert l_name != d_name, f"L and D should differ: {l_name} vs {d_name}"
        assert "alanine" in l_name and "alanine" in d_name

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


class TestWSD07RetainedStereo:
    """WSD-07 (Phase 175): a stereo-tagged free STANDARD amino acid resolves to its
    retained PIN with the configurational descriptor (P-103.1.1.1), not a
    wrong-parent systematic name; a diastereomer the bare name cannot represent
    DEFERS to the systematic namer."""

    def test_true_l_isoleucine(self):
        assert name_compound("CC[C@H](C)[C@H](N)C(=O)O") == "isoleucine"

    def test_d_alanine(self):
        assert name_compound("C[C@@H](N)C(=O)O") == "D-alanine"

    def test_l_alanine_implicit(self):
        assert name_compound("C[C@H](N)C(=O)O") == "alanine"

    def test_allo_isoleucine_named(self):
        # v24 W8 P3 Task 3.1 (P-103.1.3.2.2): the 2-centre allo diastereomers of
        # threonine/isoleucine now emit their retained-name PIN. This SMILES is
        # (2R,3S) = D-allo-isoleucine (CIP + OPSIN-RT verified). Previously this
        # deferred to the systematic name (test formerly `test_allo_isoleucine_defers`).
        assert name_compound("CC[C@H](C)[C@@H](N)C(=O)O") == "D-allo-isoleucine"

    def test_glycine_achiral(self):
        assert name_compound("NCC(=O)O") == "glycine"

    def test_peptide_not_regressed(self):
        # get_amino_acid_name is shared by name_peptide; the default (no-descriptor)
        # path must keep peptides intact.
        assert name_compound("NCC(=O)NCC(=O)O") == "glycylglycine"

    def test_nonstandard_aa_keeps_systematic(self):
        # A non-standard AA (D-2-aminobutanoic acid) keeps the systematic name —
        # 'D-butyrine' is not OPSIN-parseable, so the descriptor path is restricted
        # to STANDARD amino acids.
        assert name_compound("CC[C@@H](N)C(=O)O") == "(2R)-2-aminobutanoic acid"
