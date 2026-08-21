"""
v33 Phase 2 Task 2.2 (breadth): backbone-substitutive peptide producer.

After Task 2.0 (dispatch fix) + Task 2.1 (Lever C, capped termini), ~214 of
the true-alpha-peptide backlog rows (
family `peptide`) still abstain -- mostly giant / non-standard-residue
peptides the residue-table/Lever-A/B/C path cannot name at all.

This module adds `rules.peptides._try_backbone_substitutive` -- an
ADDITIVE, LAST-RESORT fallback tried only when the flat acylamino
convention and Levers A/B/C have all declined:

  - Parent = the C-terminal residue's own free carboxylic acid (must be
    EXACTLY one free -COOH on that residue's own fragment -- P-41 senior
    principal group).
  - Every OTHER residue (working N-terminal-ward) is folded in as an
    N-acyl / N-(...amido) substituent, one residue at a time: a STANDARD
    residue uses its tabulated retained acyl form (PREPEND, exactly the
    ordinary flat convention); a NON-STANDARD residue is named as a free
    acid via the general/systematic namer, then converted to an acyl
    (prepend) or amido (splice) prefix via the SAME helpers Lever A/B
    already use (`_acid_to_acyl` / `acid_name_to_amido_prefix` /
    `_swap_amino_for_amido`).
  - The **side-chain-acid trap** (the contributor guide; 57/226 backlog rows carry >=2
    free -COOH): a Glu/Asp side-chain acid on any NON-parent residue is a
    genuine competing principal group this producer must never misplace --
    it ABSTAINS rather than guess. Likewise a capped (amide/ester/aldehyde)
    or dual-acid C-terminus is out of this producer's scope (correctly
    abstains; other levers or a future one may cover it).
  - Every candidate is gated by `_rt_verified` (full-InChI OPSIN
    round-trip) via `name_peptide`'s existing call sites -- a wrong or
    unparseable guess degrades to abstention, never ships (0-wrong
    ABSOLUTE).

Three witnesses spanning simple -> bigger, per plan:
  1. A hand-built 5-residue chain (one non-standard, defined-(2R)-stereo
     residue at the true N-terminus, prepended onto 3 standard residues
     and a standard parent) -- proves the mechanism CONVERTS when the
     shape is in scope, and does so via the FAST peptide path directly
     (no fallthrough to the general/composer engine, which for a chain
     this size otherwise mis-parents and fails -- verified directly
     building this producer).
  2. Real backlog idx 97 (HA=43, 4 residues, ALL standard-shaped but with
     completely UNDEFINED alpha stereochemistry throughout) -- must
     ABSTAIN: P-103.1.3.1/P-103.3.4 stereo-honesty already blocks the flat
     path, and the resulting systematic parent name (a multiplied
     "x,y-diamino..." glutamine shape) has no single addressable 'amino'
     locant to splice into -- a correct, principled abstention, not a
     defect.
  3. Real backlog idx 121 (HA=143, 19 residues) -- a genuine Glu side-chain
     free acid sits on an INTERNAL (non-parent) residue alongside a clean
     single-COOH C-terminus. Must ABSTAIN (the side-chain-acid trap),
     never misplace the amidation onto the wrong acid.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import name_compound
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse

# GATE ON (production default) -- matches test_peptide_capped_termini.py's
# rationale: this module tests exactly what `name_compound` emits.
pytestmark = pytest.mark.opsin_gate


def _full_rt(smiles: str, name: str) -> bool:
    """name -> OPSIN -> full InChIKey, compared to the input's InChIKey."""
    if not name:
        return False
    o = opsin_parse(name)
    if not o:
        return False
    mol_in = Chem.MolFromSmiles(smiles)
    mol_out = Chem.MolFromSmiles(o)
    if mol_in is None or mol_out is None:
        return False
    return inchi.MolToInchiKey(mol_in) == inchi.MolToInchiKey(mol_out)


@pytest.mark.unit
class TestBackboneSubstitutiveConverts:
    """A shape squarely in the new producer's scope must actually convert."""

    def test_nonstandard_nterm_residue_over_a_standard_chain(self):
        """(2-amino-2-cyclopropylacetyl, non-standard, defined (2R) stereo,
        NOT in the amino-acid table) - Gly - Ala - Val - Leu(parent). The
        flat convention declines at `_identify_residues` (the N-terminal
        residue is non-standard); Levers A/B/C do not apply either (the
        cap carries its own free amino group, so Lever A's plain-acyl
        requirement fails; there is no non-alpha bond for Lever B; no cap
        for Lever C to strip). `_try_backbone_substitutive` builds it by
        PREPENDING the non-standard residue's own systematic acyl form,
        then the three standard acyl residues, onto the parent -- a pure
        prepend chain (the N-terminal residue's own open 'amino' locant is
        never itself a splice TARGET here, since nothing precedes it)."""
        smi = (
            "N[C@H](C1CC1)C(=O)NCC(=O)N[C@@H](C)C(=O)"
            "N[C@@H](C(C)C)C(=O)N[C@@H](CC(C)C)C(=O)O"
        )
        result = name_compound(smi)
        assert not is_failure_name(result), result
        assert result == (
            "(2R)-2-amino-2-cyclopropylethanoylglycylalanylvalylleucine"
        ), result
        assert _full_rt(smi, result), result


@pytest.mark.unit
class TestBackboneSubstitutiveAbstainsCorrectly:
    """Real backlog witnesses that must ABSTAIN -- never misplace or ship
    a wrong molecule."""

    # HA=43, 4 residues (Ala-Trp-Trp-Gln shaped), completely undefined
    # alpha stereochemistry throughout.
    UNDEFINED_STEREO_WITNESS = (
        "CC(N)C(=O)NC(Cc1c[nH]c2ccccc12)C(=O)NC(Cc1c[nH]c2ccccc12)C(=O)"
        "NC(CCC(N)=O)C(=O)O"
    )

    # HA=143, 19 residues -- a free Glu side-chain acid on an internal
    # residue, alongside a clean single-COOH C-terminus.
    SIDE_CHAIN_ACID_TRAP_WITNESS = (
        "CCC(C)C(NC(=O)C(CCC(=O)O)NC(=O)C(C)NC(=O)C(C)NC(=O)C(C)NC(=O)"
        "C1CCCN1C(=O)CNC(=O)C1CCCN1C(=O)C(NC(=O)C1CCCN1C(=O)C(C)NC(=O)"
        "C(CC(C)C)NC(=O)C(CCC(=O)O)NC(=O)C(CCCNC(=N)N)NC(=O)"
        "C(Cc1ccc(O)cc1)NC(=O)C(CC(C)C)NC(=O)C(CC(N)=O)NC(=O)C(CC(N)=O)"
        "NC(=O)C(N)CCSC)C(C)CC)C(=O)O"
    )

    def test_undefined_stereo_witness_abstains(self):
        result = name_compound(self.UNDEFINED_STEREO_WITNESS)
        assert is_failure_name(result), (
            f"undefined-stereo backlog witness must abstain "
            f"(P-103.3.4 stereo honesty; no addressable splice point for "
            f"the systematic glutamine-shaped parent), got: {result}"
        )

    def test_side_chain_acid_trap_witness_abstains(self):
        result = name_compound(self.SIDE_CHAIN_ACID_TRAP_WITNESS)
        assert is_failure_name(result), (
            f"side-chain-acid-trap backlog witness must abstain (a free "
            f"Glu side-chain acid on a non-parent residue must never be "
            f"misplaced), got: {result}"
        )


@pytest.mark.unit
class TestRegressionUnaffected:
    """Spot-checks that ordinary peptides (and Task 2.0/2.1's levers) stay
    PIN byte-identical -- this producer is tried LAST, only on decline."""

    def test_glycylglycine_unchanged(self):
        assert name_compound("NCC(=O)NCC(=O)O") == "glycylglycine"

    def test_standard_tripeptide_unchanged(self):
        smi = "CC[C@H](C)[C@H](N)C(=O)N[C@@H](C)C(=O)N1CCC[C@@H]1C(=O)O"
        result = name_compound(smi)
        assert not is_failure_name(result), result
        assert _full_rt(smi, result), result

    def test_lever_c_amide_cterm_unchanged(self):
        """Task 2.1's own witness -- must still be handled by Lever C, not
        accidentally short-circuited by this new, later-tried producer."""
        smi = "N[C@@H](C)C(=O)NCC(N)=O"
        result = name_compound(smi)
        assert result == "alanylglycinamide", result
