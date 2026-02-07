"""
Tests for peptide naming using acylamino convention (IUPAC 3AA-13).

Peptides are named by appending -yl acyl forms of N-terminal residues
to the C-terminal amino acid name, joined with hyphens.

TDD RED phase: These tests should FAIL initially until peptide naming is implemented.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound


# ── Dipeptides (achiral) ──────────────────────────────────────────────

@pytest.mark.unit
class TestDipeptidesAchiral:
    """Test dipeptide naming for achiral residues."""

    def test_glycylglycine(self):
        """Gly-Gly: simplest dipeptide, achiral, single word."""
        result = name_compound("NCC(=O)NCC(=O)O")
        assert result == "glycylglycine"

    def test_glycylglycine_is_single_word(self):
        """Glycylglycine has no hyphen (both residues are achiral glycine)."""
        result = name_compound("NCC(=O)NCC(=O)O")
        assert "-" not in result


# ── Dipeptides (with stereochemistry) ─────────────────────────────────

@pytest.mark.unit
class TestDipeptidesWithStereo:
    """Test dipeptide naming with L/D stereochemistry prefixes."""

    def test_glycyl_l_alanine(self):
        """Gly-L-Ala: achiral N-terminal + chiral C-terminal."""
        result = name_compound("NCC(=O)N[C@@H](C)C(=O)O")
        assert result == "glycyl-L-alanine"

    def test_l_alanylglycine(self):
        """L-Ala-Gly: chiral N-terminal + achiral C-terminal."""
        result = name_compound("N[C@@H](C)C(=O)NCC(=O)O")
        assert result == "L-alanylglycine"

    def test_l_alanyl_l_alanine(self):
        """L-Ala-L-Ala: both chiral, same residue."""
        result = name_compound("N[C@@H](C)C(=O)N[C@@H](C)C(=O)O")
        assert result == "L-alanyl-L-alanine"


# ── Tripeptides ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestTripeptides:
    """Test tripeptide naming (3 residues)."""

    def test_glycyl_l_alanyl_l_leucine(self):
        """Gly-L-Ala-L-Leu: three residues, mixed chirality."""
        result = name_compound("NCC(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)O")
        assert result == "glycyl-L-alanyl-L-leucine"


# ── Edge cases: should NOT trigger peptide naming ─────────────────────

@pytest.mark.unit
class TestPeptideEdgeCases:
    """Test that non-peptide amides are NOT misrouted to peptide naming."""

    def test_asparagine_not_misrouted(self):
        """Asparagine has a primary amide side chain but is a single amino acid."""
        result = name_compound("NC(CC(N)=O)C(=O)O")
        assert result == "asparagine"

    def test_n_acetylglycine_not_misrouted(self):
        """N-acetylglycine has an amide bond but NO terminal NH2 on acyl side."""
        result = name_compound("CC(=O)NCC(=O)O")
        # Should NOT produce a peptide name (acetyl is not an amino acid residue)
        assert "glycylglycine" not in result.lower()
        assert "peptide" not in result.lower()

    def test_simple_acetamide_not_misrouted(self):
        """Simple amide: no amino acid pattern at all."""
        result = name_compound("CC(=O)N")
        assert result == "acetamide"


# ── Data validation ───────────────────────────────────────────────────

@pytest.mark.unit
class TestAminoAcidAcylNames:
    """Test that the amino acid acyl name data is complete and correct."""

    def test_acyl_names_for_all_20_amino_acids(self):
        """All 20 standard amino acids must have acyl form entries."""
        from orthonym.data.amino_acids import AMINO_ACID_ACYL_NAMES
        assert len(AMINO_ACID_ACYL_NAMES) >= 20

    def test_all_acyl_names_end_in_yl(self):
        """Every acyl name must end in '-yl' per IUPAC convention."""
        from orthonym.data.amino_acids import AMINO_ACID_ACYL_NAMES
        for name, acyl in AMINO_ACID_ACYL_NAMES.items():
            assert acyl.endswith("yl"), f"{name} -> {acyl} does not end in 'yl'"

    def test_known_acyl_names(self):
        """Verify specific acyl names for key amino acids."""
        from orthonym.data.amino_acids import AMINO_ACID_ACYL_NAMES
        expected = {
            "glycine": "glycyl",
            "alanine": "alanyl",
            "valine": "valyl",
            "leucine": "leucyl",
            "proline": "prolyl",
            "phenylalanine": "phenylalanyl",
        }
        for aa, expected_acyl in expected.items():
            assert AMINO_ACID_ACYL_NAMES.get(aa) == expected_acyl, \
                f"Expected {aa} -> {expected_acyl}, got {AMINO_ACID_ACYL_NAMES.get(aa)}"


# ── Stereochemistry mapping ──────────────────────────────────────────

@pytest.mark.unit
class TestStereoMapping:
    """Test S->L and R->D stereo prefix mapping."""

    def test_s_config_maps_to_l(self):
        """S-configured alpha-carbon should get L- prefix."""
        # L-alanine has S configuration at alpha carbon
        result = name_compound("N[C@@H](C)C(=O)NCC(=O)O")
        # N-terminal L-Ala should produce "L-alanyl..."
        assert result.startswith("L-alanyl")

    def test_glycine_no_stereo_prefix(self):
        """Glycine is achiral -- no L/D prefix."""
        result = name_compound("NCC(=O)NCC(=O)O")
        assert not result.startswith("L-")
        assert not result.startswith("D-")
