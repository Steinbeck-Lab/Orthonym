"""
 a phase (breadth): peptide naming levers.

Two ADDITIVE fallbacks in rules/peptides.py, tried only when the ordinary
flat acylamino chain convention declines, both gated by an OPSIN round-trip
so a table/topology miss degrades to abstention, never a wrong or
atom-dropping name (0-wrong ABSOLUTE):

- Lever A (N-acyl cap): a fatty/simple-acyl N-terminal cap in front of an
  otherwise fully standard >=2-residue alpha chain.
- Lever B (gamma/beta-linked donor): a single non-alpha bond formed from a
  standard amino acid's OWN side-chain carboxyl (glutamic acid's gamma-,
  aspartic acid's beta-carboxyl -- glutathione's linkage type), built via the
  systematic substitutive acyl/amido construction rather than the flat
  acylamino-chain shorthand.

Derivation, a trace citations and OPSIN-hand-verification for every target name
below: internal notes
"""

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import name_compound
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse

# GATE ON (production default): tests/conftest.py's autouse fixture disables
# the final OPSIN validity gate for every test UNLESS this marker is set, so
# without it a wrong/atom-dropping candidate that production always
# suppresses (e.g. chebi 371's underlying pipeline emits a WRONG
# 'unrelated fragment' name that only the gate catches) leaks through as a
# green-but-unrealistic result. This module tests exactly what
# `name_compound` emits for a real caller, so the gate must be ON.
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


def _heavy_atoms(smiles: str) -> int:
    return Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()


# ── Lever B: gamma-glutamyl / non-standard terminal residue ──────────────


@pytest.mark.unit
class TestLeverBGammaLink:
    """a trace: chebi 494 (gamma-Glu-ACC) + hand-RT-verified gamma-glutamyl
    targets. Was ABSTAIN ('unknown organic compound') before this change."""

    def test_chebi494_gamma_glu_acc(self):
        """chebi 494: gamma-Glu amide-bonded to 1-aminocyclopropane-1-
        carboxylic acid (ACC, non-standard). a trace-derived, OPSIN-hand-
        verified (no stereo imposed, matches the input's undefined stereo):
        '1-(4-amino-4-carboxybutanamido)cyclopropane-1-carboxylic acid'.
        """
        smi = "NC(CCC(=O)NC1(C(=O)O)CC1)C(=O)O"
        result = name_compound(smi)
        assert not is_failure_name(result), result
        assert result == (
            "1-(4-amino-4-carboxybutanamido)cyclopropane-1-carboxylic acid"
        ), result
        assert _full_rt(smi, result), result
        assert _heavy_atoms(smi) == _heavy_atoms(opsin_parse(result))

    def test_real_glutathione_stereo_defined(self):
        """Real (L-Glu, L-Cys) reduced glutathione: a 3-residue gamma-link
        chain (Glu-gamma-Cys-Gly). Hand-RT-verified via direct OPSIN
        subprocess (independent of the acylamino table):
        '(4S)-4-amino-4-carboxybutanoylcysteinylglycine'. Was ABSTAIN
        before this change (the general/composer engine does not reach a
        3-residue conjugate of this shape)."""
        smi = "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O"
        result = name_compound(smi)
        assert not is_failure_name(result), result
        assert result == "(4S)-4-amino-4-carboxybutanoylcysteinylglycine", result
        assert _full_rt(smi, result), result

    def test_gamma_glu_gly_stereo_defined(self):
        """gamma-L-Glu-Gly (2-residue, standard Gly acceptor). Hand-RT-
        verified: '(4S)-4-amino-4-carboxybutanoylglycine'. NOTE: this exact
        molecule was ALREADY named correctly before this change, via a
        DIFFERENT existing mechanism ('(2S)-2-amino-5-(carboxymethylamino)-
        5-oxopentanoic acid', also round-trip-verified) -- Lever B now
        answers first via peptides.py instead. Both forms are valid IUPAC
        names for this structure; this test only asserts the NEW output
        still round-trips (0-wrong), not that it is the ONLY valid name."""
        smi = "N[C@@H](CCC(=O)NCC(=O)O)C(=O)O"
        result = name_compound(smi)
        assert not is_failure_name(result), result
        assert _full_rt(smi, result), result

    def test_gamma_glu_gly_undefined_stereo_not_named_alpha(self):
        """Regression (pre-existing test, tests/unit/test_peptides.py):
        gamma-Glu-Gly must never be named via the WRONG alpha convention."""
        result = name_compound("N[C@@H](CCC(=O)NCC(=O)O)C(=O)O")
        assert result != "glutamylglycine"


# ── Lever A: N-acyl cap on an otherwise-standard >=2-residue chain ───────


@pytest.mark.unit
class TestLeverANAcylCap:
    """a trace a: a fatty/simple-acyl N-terminal cap. Was ABSTAIN before this
    change for a chain length >= 2 residues behind the cap."""

    def test_synthetic_fatty_acyl_dipeptide(self):
        """a trace's own cited RT-verified target name, forward-parsed via OPSIN
        to get the input SMILES (name -> SMILES -> name round-trip):
        '3-hydroxy-11-methyltridecanoylglycylglycine'."""
        smi = "OC(CC(=O)NCC(=O)NCC(=O)O)CCCCCCCC(CC)C"
        result = name_compound(smi)
        assert not is_failure_name(result), result
        # Decision A part 2 (2026-09-27): was the Lever-A retained name
        # '3-hydroxy-11-methyltridecanoylglycylglycine'. The peptide PIN is the
        # SUBSTITUTIVE form (V38-PEPTIDE-PIN-VERDICT.md, as for the dodecanoyl
        # sibling below); the decorated acyl is now the method (1) amido prefix
        #, the Blue Book). Full-InChIKey round trip below.
        assert result == "[2-(3-hydroxy-11-methyltridecanamido)acetamido]acetic acid", result
        assert _full_rt(smi, result), result

    def test_simple_fatty_acyl_dipeptide(self):
        """A plainer synthetic case: dodecanoyl (lauroyl) cap on Gly-Gly.
         (V38-PEPTIDE-PIN-VERDICT.md): the peptide PIN is the SUBSTITUTIVE
        form, so this reroutes off the Lever-A retained name
        'dodecanoylglycylglycine' to '(2-dodecanamidoacetamido)acetic acid'.
        Full-InChIKey round-trip verified."""
        smi = "CCCCCCCCCCCC(=O)NCC(=O)NCC(=O)O"
        result = name_compound(smi)
        assert not is_failure_name(result), result
        assert result == "(2-dodecanamidoacetamido)acetic acid", result
        assert _full_rt(smi, result), result

    def test_single_residue_n_acetylglycine_unchanged(self):
        """Lever A must NOT fire for a bare N-acyl AMINO ACID (a single
        residue behind the cap, not a genuine >=2-residue peptide chain) --
        that shape already names correctly via the general/composer
        pipeline today ('acetamidoacetic acid'). This is the existing test
        suite's own invariant (tests/unit/test_peptides.py
        test_n_acetylglycine_still_excluded /
        test_n_acetylglycine_not_misrouted); asserted again here at the
        exact-string level as a Lever-A-specific regression guard."""
        result = name_compound("CC(=O)NCC(=O)O")
        assert result == "acetamidoacetic acid", result


# ── Fail-closed: mid-chain non-standard residue must abstain cleanly ─────


@pytest.mark.unit
class TestFailClosedMidChainNonStandard:
    """a trace (chebi 371): OPSIN itself mis-parses a non-retained acyl word
    used as a CONTINUING chain link (verified via 3 independent controls in
    the a trace), so no phrasing of this molecule via the peptide-chain
    convention can round-trip. Must abstain (a failure-name sentinel),
    never emit a wrong or atom-dropping name."""

    CHEBI_371 = (
        "CCC(C)CCCCCCCC(O)CC(=O)N[C@@H](CCC(=O)O)C(=O)N[C@H](CC(=O)N[C@H]"
        "(CC(C)C)C(=O)N[C@H](C(=O)N[C@@H](CC(=O)O)C(=O)N[C@H](CC(C)C)C(=O)"
        "N[C@@H](CC(C)C)C(=O)O)C(C)C)CC(C)C"
    )

    def test_chebi371_abstains(self):
        result = name_compound(self.CHEBI_371)
        assert is_failure_name(result), (
            f"chebi 371 must abstain (mid-chain non-standard residue is not "
            f"solvable via the peptide-chain convention -- SPY §2), got: {result}"
        )

    def test_chebi371_never_a_smaller_molecule(self):
        """Even if some future change makes this molecule emit a name, it
        must never describe a SMALLER molecule (an atom-drop) -- if a name
        is emitted at all, it must round-trip and keep every heavy atom."""
        result = name_compound(self.CHEBI_371)
        if is_failure_name(result):
            return
        o = opsin_parse(result)
        assert o is not None, result
        assert _heavy_atoms(self.CHEBI_371) == _heavy_atoms(o), (
            result, self.CHEBI_371, o,
        )
        assert _full_rt(self.CHEBI_371, result), result


# ── No-atom-drop invariant, over every emitted target above ──────────────


@pytest.mark.unit
class TestNoAtomDropInvariant:
    """For every molecule this module's new levers name, the OPSIN-parsed
    output must have the SAME heavy-atom count as the input -- a name is
    never allowed to silently describe a smaller molecule."""

    CASES = [
        "NC(CCC(=O)NC1(C(=O)O)CC1)C(=O)O",                     # chebi 494
        "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O",       # glutathione
        "N[C@@H](CCC(=O)NCC(=O)O)C(=O)O",                       # gamma-Glu-Gly
        "OC(CC(=O)NCC(=O)NCC(=O)O)CCCCCCCC(CC)C",               # fatty N-cap
        "CCCCCCCCCCCC(=O)NCC(=O)NCC(=O)O",                      # lauroyl-Gly-Gly
    ]

    def test_every_emitted_name_preserves_heavy_atom_count(self):
        for smi in self.CASES:
            result = name_compound(smi)
            assert not is_failure_name(result), (smi, result)
            o = opsin_parse(result)
            assert o is not None, (smi, result)
            in_count = _heavy_atoms(smi)
            out_count = _heavy_atoms(o)
            assert in_count == out_count, (
                f"{smi} -> {result!r} -> {o!r}: "
                f"heavy atoms {in_count} != {out_count} (ATOM DROP)"
            )


# ── Regression: existing standard-peptide + fail-closed tests unaffected ─


@pytest.mark.unit
class TestRegressionUnaffected:
    """Spot-checks that ordinary standard-chain peptides (untouched code
    path) and the pre-existing isopeptide fail-closed guards are unchanged."""

    def test_glycylglycine_unchanged(self):
        #: substitutive PIN (V38-PEPTIDE-PIN-VERDICT.md); RT verified.
        assert name_compound("NCC(=O)NCC(=O)O") == "(2-aminoacetamido)acetic acid"

    def test_alanylglycine_unchanged(self):
        #: substitutive PIN (V38-PEPTIDE-PIN-VERDICT.md); RT verified.
        assert (name_compound("N[C@@H](C)C(=O)NCC(=O)O")
                == "[(2S)-2-aminopropanamido]acetic acid")

    def test_glutathione_achiral_still_declines_or_roundtrips(self):
        """The ACHIRAL-drawn glutathione test molecule (no stereo defined
        anywhere) from tests/unit/test_peptides.py
        (test_glutathione_not_named_alpha): must never be named
        'glutamylcysteinylglycine' (the wrong alpha constitution). Lever B's
        RT-verify gate correctly declines this specific undefined-stereo
        drawing today (a bare 'cysteinyl' acceptor word imposes an L default
        OPSIN cannot know is unspecified here) -- assert whichever holds
        (decline, or a round-tripping name) but never the wrong string."""
        smi = "NC(CCC(=O)NC(CS)C(=O)NCC(=O)O)C(=O)O"
        result = name_compound(smi)
        assert result != "glutamylcysteinylglycine"
        if not is_failure_name(result):
            assert _full_rt(smi, result), result

    def test_epsilon_lysine_isopeptide_not_named_alpha(self):
        result = name_compound("NCC(=O)NCCCC[C@H](N)C(=O)O")
        assert result != "glycyllysine"
