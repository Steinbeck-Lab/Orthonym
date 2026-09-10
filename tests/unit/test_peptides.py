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
        """Gly-Gly PIN is the SUBSTITUTIVE form (V38-PEPTIDE-PIN-VERDICT.md:
        Chapter identifies NO PINs, so the retained peptide name
        'glycylglycine' is non-PIN). Full-InChIKey round-trip verified."""
        result = name_compound("NCC(=O)NCC(=O)O")
        assert result == "(2-aminoacetamido)acetic acid"

    def test_glycylglycine_is_single_word(self):
        """: Gly-Gly now emits the substitutive PIN (see test_glycylglycine
        and V38-PEPTIDE-PIN-VERDICT.md); the old 'single word / no hyphen'
        property was a property of the retained name, no longer emitted."""
        result = name_compound("NCC(=O)NCC(=O)O")
        assert result == "(2-aminoacetamido)acetic acid"


# ── Dipeptides (with stereochemistry) ─────────────────────────────────

@pytest.mark.unit
class TestDipeptidesWithStereo:
    """Test dipeptide naming with L/D stereochemistry prefixes."""

    def test_glycyl_l_alanine(self):
        """Gly-L-Ala PIN is substitutive (V38-PEPTIDE-PIN-VERDICT.md; peptide
        names are non-PIN). Full-InChIKey round-trip verified."""
        result = name_compound("NCC(=O)N[C@@H](C)C(=O)O")
        assert result == "(2S)-2-(2-aminoacetamido)propanoic acid"

    def test_l_alanylglycine(self):
        """L-Ala-Gly PIN is substitutive (V38-PEPTIDE-PIN-VERDICT.md).
        Full-InChIKey round-trip verified."""
        result = name_compound("N[C@@H](C)C(=O)NCC(=O)O")
        assert result == "2-[(2S)-2-aminopropanamido]ethanoic acid"

    def test_l_alanyl_l_alanine(self):
        """L-Ala-L-Ala PIN is substitutive (V38-PEPTIDE-PIN-VERDICT.md).
        Full-InChIKey round-trip verified."""
        result = name_compound("N[C@@H](C)C(=O)N[C@@H](C)C(=O)O")
        assert result == "(2S)-2-[(2S)-2-aminopropanamido]propanoic acid"


# ── Tripeptides ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestTripeptides:
    """Test tripeptide naming (3 residues)."""

    def test_glycyl_l_alanyl_l_leucine(self):
        """Gly-L-Ala-L-Leu: three residues, all L -> omits L."""
        result = name_compound("NCC(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)O")
        assert result == "glycylalanylleucine"


# ── Tetrapeptide residue ORDER (backbone walk, not atom index) ────────────
# _extract_residues used to order internal residues by atom index, which is
# SMILES-spelling-dependent, so a tetrapeptide's two internal residues could
# swap (Val-Glu-Ile-Arg -> Val-Ile-Glu-Arg) — a wrong constitution that only
# stopped. Residues are now ordered by walking the amide backbone N->C.

@pytest.mark.unit
class TestTetrapeptideOrder:
    """Internal-residue order must follow the backbone, deterministically."""

    def _canon_from(self, name):
        # OPSIN-independent: name must denote the input's constitution; here we
        # assert the residue ORDER directly via the produced string.
        return name

    def test_val_glu_ile_arg_order(self):
        """L-Val-L-Glu-L-Ile-L-Arg: internals Glu,Ile must stay in backbone order."""
        smi = ("CC(C)[C@@H](N)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H]"
               "([C@@H](C)CC)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O")
        result = name_compound(smi)
        # glutamyl (Glu) MUST precede isoleucyl (Ile) — the backbone order
        assert result is not None and "glutamyl" in result and "isoleucyl" in result
        assert result.index("glutamyl") < result.index("isoleucyl"), result

    def test_tetrapeptide_order_is_spelling_stable(self):
        """The same molecule in different SMILES spellings gives ONE name."""
        smi = "C[C@H](N)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)NCC(=O)O"
        m = Chem.MolFromSmiles(smi)
        names = set()
        for _ in range(200):
            names.add(name_compound(Chem.MolToSmiles(m, canonical=False, doRandom=True)))
            if len(names) >= 15:
                break
        names.add(name_compound(Chem.MolToSmiles(m, canonical=True)))
        # 'seryl' before 'phenylalanyl' in every spelling
        assert len(names) == 1, names


# ── Isopeptide (non-alpha linkage) must fail closed ──────────────────────
# Glutathione is gamma-Glu-Cys-Gly: its first amide is on the glutamate SIDE-chain
# carboxyl, not the alpha-carboxyl. Naming it alpha ("glutamylcysteinylglycine")
# is a WRONG constitution, so the alpha-carboxyl check makes it abstain.

@pytest.mark.unit
class TestIsopeptideFailClosed:
    def test_glutathione_not_named_alpha(self):
        result = name_compound("NC(CCC(=O)NC(CS)C(=O)NCC(=O)O)C(=O)O")
        assert result != "glutamylcysteinylglycine"

    def test_gamma_glutamyl_glycine_not_named_alpha(self):
        """gamma-Glu-Gly: side-chain (gamma) carboxyl amide, not alpha."""
        result = name_compound("N[C@@H](CCC(=O)NCC(=O)O)C(=O)O")
        assert result != "glutamylglycine"

    def test_epsilon_lysine_isopeptide_not_named_alpha(self):
        """Gly-eps-Lys: amide on lysine's side-chain (epsilon) amine, not alpha."""
        result = name_compound("NCC(=O)NCCCC[C@H](N)C(=O)O")
        assert result != "glycyllysine"


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
        #: Pro-Gly PIN is substitutive (V38-PEPTIDE-PIN-VERDICT.md; the
        # sliceC target). Full-InChIKey round-trip verified.
        result = name_compound("OC(=O)CNC(=O)[C@@H]1CCCN1")
        assert result == "2-[(2S)-pyrrolidine-2-carboxamido]ethanoic acid"

    def test_prolylalanine(self):
        #: Pro-Ala PIN is substitutive (V38-PEPTIDE-PIN-VERDICT.md).
        # Full-InChIKey round-trip verified.
        result = name_compound("C[C@@H](C(=O)O)NC(=O)[C@@H]1CCCN1")
        assert result == "(2S)-2-[(2S)-pyrrolidine-2-carboxamido]propanoic acid"

    def test_n_acetylglycine_still_excluded(self):
        """The relaxation must NOT admit an N-acyl amino acid (acylated N)."""
        result = name_compound("CC(=O)NCC(=O)O")
        assert "glycyl" not in result.lower()


# ── Edge cases: should NOT trigger peptide naming ─────────────────────

@pytest.mark.unit
class TestPeptideEdgeCases:
    """Test that non-peptide amides are NOT misrouted to peptide naming."""

    def test_asparagine_not_misrouted(self):
        """Asparagine has a primary amide side chain but is a single amino acid.

         a phase (change-asserted-value, was `== "asparagine"`): the input
        SMILES has NO wedge/parity at the alpha-carbon (CHI_UNSPECIFIED) -- a
        genuinely stereo-undefined structure. The bare retained name `asparagine`
        is Table 10.4's name for the DEFINED (L) configuration (`## ****
        The stereodescriptors 'D' and 'L'`, the Blue Book: "The
        stereodescriptor 'xi'... indicates unknown configuration"), and OPSIN's
        grammar always resolves a bare amino-acid retained name to that ONE
        defined stereocentre -- so asserting it against this input is provably
        impossible to round-trip: input full InChIKey
        `DCXYFEDJOCDNAF-UHFFFAOYSA-N` (no stereo layer) vs OPSIN's parse of
        'asparagine' `DCXYFEDJOCDNAF-REOHCLBHSA-N` (defined stereo layer) --
        same skeleton, different (missing-vs-present) stereo layer, so a
        byte-identical full round-trip is impossible by construction, not by
        chance. `2,4-diamino-4-oxobutanoic acid` is the Blue Book's OWN
        systematic name for asparagine (Table 10.4, the Blue Book) and
        full-InChIKey RT-exacts to this exact input (`DCXYFEDJOCDNAF-
        UHFFFAOYSA-N` both sides, OPSIN-verified independently of this fix's
        code). Mutation-tested: `scripts/an A/B check` with the pre-fix
        (HEAD-committed-at-35d5e921) versions of `data/amino_acids.py` +
        `data/retained_names.py` + `rules/esters.py` swapped in makes this
        assertion FAIL (old code still emits bare 'asparagine'); the working
        tree's fix makes it PASS.
        """
        result = name_compound("NC(CC(N)=O)C(=O)O")
        assert result == "2,4-diamino-4-oxobutanoic acid"

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
        """S-configured alpha-carbon must be preserved as (2S) in the name.

        : L-Ala-Gly now emits the substitutive PIN (V38-PEPTIDE-PIN-VERDICT.md;
        full-InChIKey round-trip verified). The N-terminal L-alanine's S centre
        surfaces as (2S) in the acyl amido prefix (a broken S mapping would emit
        (2R) and fail the round-trip gate -> fallback), and never a 'D-' prefix.
        """
        result = name_compound("N[C@@H](C)C(=O)NCC(=O)O")
        assert result == "2-[(2S)-2-aminopropanamido]ethanoic acid"
        assert not result.startswith("D-")

    def test_glycine_no_stereo_prefix(self):
        """Glycine is achiral -- no L/D prefix."""
        result = name_compound("NCC(=O)NCC(=O)O")
        assert not result.startswith("L-")
        assert not result.startswith("D-")


# ── Cysteine L/D inversion  ────────────────────────────────

@pytest.mark.unit
class TestCysteineStereoInversion:
    """Test that cysteine CIP inversion is handled correctly.

    : Cysteine has sulfur (Z=16) in its side chain which
    outranks oxygen (Z=8) in COOH, inverting CIP priorities.
    Result: L-cysteine = R (CIP), D-cysteine = S (CIP).
    """

    def test_l_cysteine_gets_l_prefix(self):
        """L-cysteine (R config, inverted) is identified as L, then omitted.

         suppresses the L descriptor; the CIP inversion still matters
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

        In a peptide the L is omitted, so the N-terminal residue must
        be a bare 'alanyl' with no 'D-' (a broken S->L mapping would tag it D-).
        """
        #: L-Ala-Gly emits the substitutive PIN (V38-PEPTIDE-PIN-VERDICT.md);
        # the S centre is preserved as (2S) and never mis-tagged 'D-'.
        result_peptide = name_compound("N[C@@H](C)C(=O)NCC(=O)O")
        assert result_peptide == "2-[(2S)-2-aminopropanamido]ethanoic acid", \
            f"got: {result_peptide}"

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


# ── Stereo honesty: never fabricate an implicit L on stereo-UNSPECIFIED
# residues (a phase L3-2e) ───────────────────────────────────────────
# "The stereodescriptors 'D' and 'L'" (the Blue Book): a
# bare retained amino-acid name asserts a SPECIFIC configuration -- "The
# stereodescriptor 'xi' (Greek letter xi) indicates unknown configuration."
# "Indication of configuration in peptides" (:54715): omitting 'L'
# in a peptide name is a DISPLAY convention for a residue KNOWN to be L --
# its own text: "A residue of unknown configuration is indicated by the
# prefix xi (Greek letter xi)." Orthonym does not emit xi-prefixed names,
# so a residue whose alpha-carbon the input leaves stereo-UNSPECIFIED must
# not be silently folded into that omission -- the retained acylamino name
# must be declined, not emitted with a fabricated implicit L.

@pytest.mark.unit
class TestStereoHonestyUndefinedResidues:
    """A stereo-UNSPECIFIED residue must never be named with the L-implying
    retained acylamino name; a DEFINED-stereo peptide must be unaffected."""

    def test_stereo_unspecified_dipeptide_declines_retained_name(self):
        """Gly-Asn, NO stereo tags anywhere: 'glycylasparagine' would assert
        an L-configured asparagine the input does not define -- must decline."""
        result = name_compound("NCC(=O)NC(CC(N)=O)C(=O)O")
        assert result != "glycylasparagine", (
            f"Fabricated an implicit-L retained name for stereo-undefined "
            f"input: {result!r}"
        )

    def test_stereo_unspecified_tetrapeptide_declines_retained_name(self):
        """Ile-Glu-Thr-Asn, NO stereo tags anywhere: the retained
        acylamino chain must not be emitted for an undefined residue."""
        smi = "CCC(C)C(C(=O)NC(CCC(=O)O)C(=O)NC(C(C)O)C(=O)NC(CC(=O)N)C(=O)O)N"
        result = name_compound(smi)
        assert result != "isoleucylglutamylthreonylasparagine", (
            f"Fabricated an implicit-L retained name for stereo-undefined "
            f"input: {result!r}"
        )

    def test_stereo_unspecified_glycine_only_dipeptide_still_names(self):
        """Gly-Gly has NO stereocentre at all (both residues achiral) -- the
        stereo-honesty guard must not affect it; must still name normally.
        : names via the substitutive PIN (V38-PEPTIDE-PIN-VERDICT.md; RT
        verified), not the retained peptide name."""
        result = name_compound("NCC(=O)NCC(=O)O")
        assert result == "(2-aminoacetamido)acetic acid"

    def test_defined_stereo_dipeptide_unchanged(self):
        """CRITICAL guard: a normal DEFINED-stereo peptide (L-Ala-Gly) must
        be completely unaffected by the stereo-honesty check.: names via
        the substitutive PIN (V38-PEPTIDE-PIN-VERDICT.md; RT verified)."""
        result = name_compound("N[C@@H](C)C(=O)NCC(=O)O")
        assert result == "2-[(2S)-2-aminopropanamido]ethanoic acid"

    def test_defined_stereo_tripeptide_unchanged(self):
        """CRITICAL guard: Gly-L-Ala-L-Leu, all-defined, unaffected."""
        result = name_compound("NCC(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)O")
        assert result == "glycylalanylleucine"

    def test_defined_stereo_arginine_tautomer_unchanged(self):
        """CRITICAL guard: the InChIKey-skeleton-fallback residue path
        (arginine's guanidine tautomer) is unaffected when stereo IS defined."""
        result = name_compound("N=C(N)NCCC[C@H](N)C(=O)NCC(=O)O")
        assert result == "arginylglycine"

    def test_partially_specified_tripeptide_declines(self):
        """One residue defined (Gly, achiral -- N/A), one residue (Ala)
        stereo-UNSPECIFIED: the whole chain must decline, not partially
        fabricate the unspecified residue's L."""
        # Gly-Ala(unspecified)-Leu(L, defined)
        smi = "NCC(=O)NC(C)C(=O)N[C@@H](CC(C)C)C(=O)O"
        result = name_compound(smi)
        assert result != "glycylalanylleucine", (
            f"Fabricated L on the stereo-unspecified alanine residue: {result!r}"
        )
