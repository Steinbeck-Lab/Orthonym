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
        """Gly-L-Ala: P-103.3.4 omits the L descriptor -> glycylalanine."""
        result = name_compound("NCC(=O)N[C@@H](C)C(=O)O")
        assert result == "glycylalanine"

    def test_l_alanylglycine(self):
        """L-Ala-Gly: P-103.3.4 omits the L descriptor -> alanylglycine."""
        result = name_compound("N[C@@H](C)C(=O)NCC(=O)O")
        assert result == "alanylglycine"

    def test_l_alanyl_l_alanine(self):
        """L-Ala-L-Ala: P-103.3.4 omits the L descriptor -> alanylalanine."""
        result = name_compound("N[C@@H](C)C(=O)N[C@@H](C)C(=O)O")
        assert result == "alanylalanine"


# ── Tripeptides ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestTripeptides:
    """Test tripeptide naming (3 residues)."""

    def test_glycyl_l_alanyl_l_leucine(self):
        """Gly-L-Ala-L-Leu: three residues, all L -> P-103.3.4 omits L."""
        result = name_compound("NCC(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)O")
        assert result == "glycylalanylleucine"


# ── Constitution-robust residue identification (InChIKey skeleton match) ──
# A residue whose reconstructed guanidine/imidazole tautomer differs from the
# table's SMILES spelling (arginine, histidine) used to fail exact-string lookup,
# so the WHOLE peptide returned None. Identification now falls back to the
# InChIKey first block (constitution), which is tautomer/isotope-invariant.

@pytest.mark.unit
class TestTautomerRobustResidues:
    """Peptides whose residues are guanidine/imidazole tautomer-sensitive."""

    def test_arginylglycine(self):
        """Arg-Gly: arginine's guanidine tautomer must still identify."""
        result = name_compound("N=C(N)NCCC[C@H](N)C(=O)NCC(=O)O")
        assert result == "arginylglycine"

    def test_glutaminylarginyltyrosine(self):
        """Gln-Arg-Tyr: an internal arginine in a tripeptide (was None)."""
        result = name_compound(
            "NC(=O)CC[C@H](N)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O")
        assert result == "glutaminylarginyltyrosine"

    def test_alanylhistidylglycine(self):
        """Ala-His-Gly: histidine's imidazole tautomer must still identify."""
        result = name_compound("C[C@H](N)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)NCC(=O)O")
        assert result == "alanylhistidylglycine"


# ── N-terminal proline (secondary-amine free N-terminus) ─────────────────
# _is_valid_peptide required a PRIMARY terminal NH2; N-terminal proline's ring N
# is a secondary amine, so every Pro-N-terminal peptide was rejected. The free-
# amine test now admits H1 as well, still barring an acylated N.

@pytest.mark.unit
class TestProlineNTerminus:
    """Peptides whose N-terminal residue is the cyclic imino acid proline."""

    def test_prolylglycine(self):
        result = name_compound("OC(=O)CNC(=O)[C@@H]1CCCN1")
        assert result == "prolylglycine"

    def test_prolylalanine(self):
        result = name_compound("C[C@@H](C(=O)O)NC(=O)[C@@H]1CCCN1")
        assert result == "prolylalanine"

    def test_n_acetylglycine_still_excluded(self):
        """The relaxation must NOT admit an N-acyl amino acid (acylated N)."""
        result = name_compound("CC(=O)NCC(=O)O")
        assert "glycyl" not in result.lower()


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
        """S-configured alpha-carbon is identified as L, then omitted (P-103.3.4).

        L-alanine has S configuration; in a peptide the L descriptor is not cited,
        so a correctly-L-identified N-terminal residue yields a bare acyl stem with
        NO 'D-' prefix (a D misidentification would surface as 'D-alanyl').
        """
        result = name_compound("N[C@@H](C)C(=O)NCC(=O)O")
        assert result == "alanylglycine"
        assert not result.startswith("D-")

    def test_glycine_no_stereo_prefix(self):
        """Glycine is achiral -- no L/D prefix."""
        result = name_compound("NCC(=O)NCC(=O)O")
        assert not result.startswith("L-")
        assert not result.startswith("D-")


# ── Cysteine L/D inversion (FMT-04a) ────────────────────────────────

@pytest.mark.unit
class TestCysteineStereoInversion:
    """Test that cysteine CIP inversion is handled correctly.

    FMT-04a: Cysteine has sulfur (Z=16) in its side chain which
    outranks oxygen (Z=8) in COOH, inverting CIP priorities.
    Result: L-cysteine = R (CIP), D-cysteine = S (CIP).
    """

    def test_l_cysteine_gets_l_prefix(self):
        """L-cysteine (R config, inverted) is identified as L, then omitted.

        P-103.3.4 suppresses the L descriptor; the CIP inversion still matters
        because a broken inversion would tag R-cysteine as D and emit 'D-cysteine'.
        So the invariant is: the residue names as (L-)cysteine with NO 'D-'.
        """
        # L-cysteine: N[C@@H](CS)C(=O)O -- R configuration at alpha carbon
        result = name_compound("N[C@@H](CS)C(=O)NCC(=O)O")
        assert "cysteinyl" in result or "cysteine" in result, \
            f"Expected a cysteine residue, got: {result}"
        assert "D-" not in result, \
            f"L-cysteine must not be mis-tagged D-, got: {result}"

    def test_d_cysteine_gets_d_prefix(self):
        """D-cysteine (S configuration) should get D- prefix."""
        # D-cysteine: N[C@H](CS)C(=O)O -- S configuration at alpha carbon
        result = name_compound("N[C@H](CS)C(=O)NCC(=O)O")
        assert "D-cysteine" in result or "D-cysteyl" in result or "D-cysteinyl" in result, \
            f"Expected D-cysteine prefix, got: {result}"

    def test_l_alanine_still_correct(self):
        """L-alanine (S config, NOT inverted) is identified as L, then omitted.

        In a peptide the L is omitted (P-103.3.4), so the N-terminal residue must
        be a bare 'alanyl' with no 'D-' (a broken S->L mapping would tag it D-).
        """
        result_peptide = name_compound("N[C@@H](C)C(=O)NCC(=O)O")
        assert result_peptide == "alanylglycine", \
            f"Expected alanylglycine, got: {result_peptide}"

    def test_benchmark_peptide_with_l_cysteine_1(self):
        """Benchmark peptide 1: Lys-Thr-Cys, C-terminal L-cysteine (L omitted)."""
        smiles = "C[C@@H](O)[C@H](NC(=O)[C@@H](N)CCCCN)C(=O)N[C@@H](CS)C(=O)O"
        result = name_compound(smiles)
        assert result is not None, "Should produce a name"
        assert result.endswith("cysteine"), \
            f"Expected an L-cysteine C-terminal (L omitted, P-103.3.4), got: {result}"
        assert "D-cysteine" not in result and "D-" not in result, \
            f"Should NOT mis-tag D, got: {result}"

    def test_benchmark_peptide_with_l_cysteine_2(self):
        """Benchmark peptide 2: Gln-Lys-Cys, C-terminal L-cysteine (L omitted)."""
        smiles = "NCCCC[C@H](NC(=O)[C@@H](N)CCC(N)=O)C(=O)N[C@@H](CS)C(=O)O"
        result = name_compound(smiles)
        assert result is not None, "Should produce a name"
        assert result.endswith("cysteine"), \
            f"Expected an L-cysteine C-terminal (L omitted, P-103.3.4), got: {result}"
        assert "D-cysteine" not in result and "D-" not in result, \
            f"Should NOT mis-tag D, got: {result}"
