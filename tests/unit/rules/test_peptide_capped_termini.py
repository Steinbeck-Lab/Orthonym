"""
 a phase Task 2.1 (breadth): capped-termini peptides.

`rules/peptides.py`'s ordinary flat acylamino-chain convention assumes a
FREE N-terminal amine and a FREE C-terminal carboxylic acid. Phase-0's a trace
(internal notes sec.3) measured that a
capped terminus is the DOMINANT decline site over the true-peptide backlog
(153/215 = 71.2% of all `name_peptide` declines are `_is_valid_peptide`
False; the largest single sub-shape, 85/215 = 40%, is BOTH termini capped
-- almost always an N-methylated backbone plus a C-terminal PRIMARY AMIDE).

This module adds "Lever C" (`rules.peptides._try_capped_termini`): an
ADDITIVE fallback, tried only when the ordinary convention declines, that
accepts:
  - a C-terminal PRIMARY CARBOXAMIDE (``-C(=O)NH2``) instead of the free
    acid, rendered with the standard amino-acid-amide suffix
    (``glycinamide``, ``phenylalaninamide``,...).
  - a mono-N-METHYLATED (free, non-acylated) N-terminus, rendered as an
    "N-methyl" prefix on the whole assembled name (the ``N-methyl-D-
    aspartic acid`` convention).
  - both together (composed).

Both caps are gated by ``_rt_verified`` (full-InChI OPSIN round-trip), so a
mis-rendering degrades to abstention, never a wrong name (0-wrong
ABSOLUTE). The C-terminal-amide side explicitly declines (abstains) when
the identified residue is aspartic/glutamic/asparagine/glutamine -- the
"side-chain-acid trap" (a free side-chain acid/amide would outrank a
suffix-amide swap under seniority, which this flat convention cannot
express).

The dispatch predicate (`rules.amino_acids.is_peptide`) also needed a
matching fix: a C-terminal-amide peptide has NO free -COOH anywhere in the
molecule, so the pre-existing `ALPHA_AMINO_ACID_SMARTS` (which hard-
requires ``[OX2H1]``) never matched and the dispatcher never called
`rules.peptides.name_peptide` at all -- fixed by OR-ing in a new,
peptide-dispatch-only terminal-primary-amide alternative that does NOT
touch `detect_amino_acid`/`name_amino_acid` (the STANDALONE free-amino-acid
namer), so that unrelated path is unaffected.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.errors import is_failure_name
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)
from tests.support.read_back import same_key

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "CC[C@H](C)[C@H](N)C(=O)N[C@@H](C)C(=O)N1CCC[C@@H]1C(=O)O",
    "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](C)C(=O)N[C@H](C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)C(N)=O)C(C)C)[C@@H](C)CC",
    "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](C)C(=O)N[C@H](C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(N)=O)[C@@H](C)CC",
    "CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H](NC(=O)[C@H](C)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCSC)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CO)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)CNC(=O)[C@@H](N)Cc1c[nH]cn1)[C@@H](C)O)[C@@H](C)O)C(C)C)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC(N)=O)C(=O)NCC(=O)NCC(=O)N1CCC[C@H]1C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)NCC(=O)N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)N1CCC[C@H]1C(=O)N[C@@H](CO)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(N)=O",
    "CC[C@H](C)[C@H](NC)C(=O)N[C@@H](CO)C(=O)N1CCC[C@H]1C(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](C(N)=O)C(C)C",
})


def _dt_name_compound(smiles):
    from orthonym import name_compound
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return declined_pin_row(smiles)["name"]
    return name_compound(smiles, on_failure="label", emit_tier="pin")


# GATE ON (production default) -- see test_peptide_breadth.py's identical
# rationale: this module tests exactly what `name_compound` emits for a
# real caller.
pytestmark = pytest.mark.opsin_gate


# ── C-terminal primary amide (real backlog witnesses) ────────────────────


@pytest.mark.unit
class TestCTerminalAmide:
    """Real true-peptide backlog witnesses
    (internal notes, `peptide` bucket) whose
    C-terminus is a primary carboxamide instead of the free acid. Both were
    ABSTAIN ('unknown organic compound') before this change."""

    # 14-residue backlog witness, C-terminus = phenylalaninamide.
    ILE_TO_PHE_NH2 = (
        "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)"
        "N[C@@H](CCCCN)C(=O)N[C@@H](C)C(=O)N[C@H](C(=O)N[C@@H](C)C(=O)"
        "N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCCN)"
        "C(=O)N[C@@H](CCCCN)C(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)C(N)=O)C(C)C)"
        "[C@@H](C)CC"
    )

    # 14-residue backlog witness, C-terminus = leucinamide.
    ILE_TO_LEU_NH2 = (
        "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)"
        "N[C@@H](CCCCN)C(=O)N[C@@H](C)C(=O)N[C@H](C(=O)N[C@@H](C)C(=O)"
        "N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCCN)"
        "C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(N)=O)"
        "[C@@H](C)CC"
    )

    def test_phenylalaninamide_cterm(self):
        result = _dt_name_compound(self.ILE_TO_PHE_NH2)
        assert not is_failure_name(result), result
        assert result.endswith("phenylalaninamide"), result
        assert same_key(self.ILE_TO_PHE_NH2, result), result

    def test_leucinamide_cterm(self):
        result = _dt_name_compound(self.ILE_TO_LEU_NH2)
        assert not is_failure_name(result), result
        # change-asserted-value (lane/l17 item 6): was ``result.endswith("-4-methylpentanamide")`` (lane/l15 task A4), the
        # substitutive amido name. Primary source: "SENIORITY ORDER FOR PARENT STRUCTURES", (the Blue Book)
        # "The senior parent structure has the maximum number of substituents corresponding to the principal characteristic
        # group (suffix)...": the amido name takes the 4-methylpentanamide chain (one amide group) as its parent, and the
        # asparagine residue is a chain with two (a butanediamide: ``NC(=O)CC(N-)C(=O)N-``), so that name is not the PIN
        # spelling. The handler prefers a substitutive name because it is the PIN (decision A, method (1),
        #:32998); one that breaks a rule of the PIN spelling is offered after the peptide name,:54158), exactly as
        # the Phe witness above, whose general path builds no substitutive name at all. Independent check: the structure
        # carries the two amide groups on one chain (RDKit, below) and ``same_key`` reads the name back with OPSIN to the
        # input's full InChIKey. Mutation: the test fails at the lane start, which names the amido name.
        assert result.endswith("leucinamide") and result.startswith("isoleucyl"), result
        assert same_key(self.ILE_TO_LEU_NH2, result), result
        asn_chain = Chem.MolFromSmarts("[NX3;H2][CX3](=O)[CH2][CX4][CX3](=O)[NX3;H1]")
        assert Chem.MolFromSmiles(self.ILE_TO_LEU_NH2).HasSubstructMatch(asn_chain)

    def test_simple_hand_built_alanylglycinamide(self):
        """Minimal synthetic witness (Ala-Gly-NH2) for a fast, easy-to-read
        positive control alongside the two large real witnesses above."""
        smi = "N[C@@H](C)C(=O)NCC(N)=O"
        result = _dt_name_compound(smi)
        assert not is_failure_name(result), result
        # Decision A part 2 (2026-09-27): was 'alanylglycinamide' (labelled below the
        # PIN tier). Peptides are not PINs; the PIN is the substitutive name
        # (V38-PEPTIDE-PIN-VERDICT.md; controller ruling, CHEBI:141425): the Ala acyl
        # is the method (1) amido prefix, the Blue Book).
        # change-asserted-value (lane/l17 item 7): was '2-[(2S)-2-aminopropanamido]acetamide', the amide of the CONH2 end
        # as the parent. Primary source: "SENIORITY OF ACYCLIC CHAINS (THE PRINCIPAL CHAIN)" (the Blue Book):
        # "When there is a choice for the principal chain, the following criteria are applied, in the order listed... The
        # principal chain: (a) contains the greater number of heteroatoms of any kind; (b) has the greater number of skeletal
        # atoms"; the amides are one class, so both amide chains are candidates, and the alanine chain
        # (CH3-CH(NH2)-CO, three atoms) has more skeletal atoms than the glycine chain (CH2-CO, two). cites the
        # amido prefix only "when a group having preference for citation as a principal characteristic group is present".
        # Independent check: ``same_key`` (OPSIN, outside the engine) reads the new name back to the input's full InChIKey,
        # and the same rule gives the mirrored 2-(2-chloro-N-methylacetamido)propanamide of lane/l15 A9 its longer chain.
        # Mutation: the assertion fails at the lane start, which names the old string.
        # What would make it wrong: a book rule that ranks a primary amide above a secondary one inside the one class (none
        # found; example (5),:21730, takes the N-methylpropanamide as the parent beside a primary amide).
        assert result == "(2S)-2-amino-N-(2-amino-2-oxoethyl)propanamide", result
        assert same_key(smi, result), result

    def test_giant_30_residue_lysinamide_cterm_and_perf(self):
        """The exact HA=342, ~30-residue giant-peptide witness from
        internal notes sec.1 -- flagged there
        as "the PERF/hang witness" (77.6s under the pre-Task-2.1 code path,
        which fell through to the slow general/composer engine before
        abstaining). Lever C now names it directly via the fast peptide
        path: was ABSTAIN before this change, now round-trips AND completes
        in low single-digit seconds (measured ~3.2s; asserted here with a
        generous 30s bound so the test itself stays a correctness check,
        not a timing benchmark -- true perf-guard work is Task 2.3)."""
        smi = (
            "CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(C)C)"
            "NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H](NC(=O)[C@H](C)NC(=O)"
            "[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)"
            "NC(=O)[C@H](CCSC)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CCCCN)"
            "NC(=O)[C@H](CO)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(=O)O)"
            "NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)"
            "[C@@H](NC(=O)CNC(=O)[C@H](CCC(=O)O)NC(=O)CNC(=O)"
            "[C@@H](N)Cc1c[nH]cn1)[C@@H](C)O)[C@@H](C)O)C(C)C)C(=O)"
            "N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)"
            "N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC(N)=O)"
            "C(=O)NCC(=O)NCC(=O)N1CCC[C@H]1C(=O)N[C@@H](CO)C(=O)"
            "N[C@@H](CO)C(=O)NCC(=O)N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)"
            "N1CCC[C@H]1C(=O)N[C@@H](CO)C(=O)N[C@@H](CCCCN)C(=O)"
            "N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)"
            "N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(N)=O"
        )
        import time
        t0 = time.time()
        result = _dt_name_compound(smi)
        elapsed = time.time() - t0
        assert not is_failure_name(result), result
        assert result.endswith("lysinamide"), result
        assert same_key(smi, result), result
        assert elapsed < 30, f"took {elapsed:.1f}s, expected low single digits"


# ── Mono-N-methylated N-terminus ──────────────────────────────────────────


@pytest.mark.unit
class TestNMethylNTerminus:
    """A free (non-acylated) mono-N-methyl on the N-terminal residue's own
    backbone amino nitrogen, rendered as an 'N-methyl' prefix (the
    established 'N-methyl-D-aspartic acid' convention)."""

    def test_simple_hand_built_n_methylalanylglycine(self):
        #: N-methyl-Ala-Gly PIN is the substitutive form
        # (V38-PEPTIDE-PIN-VERDICT.md; peptide names are non-PIN). RT verified.
        smi = "CN[C@@H](C)C(=O)NCC(=O)O"
        result = _dt_name_compound(smi)
        assert not is_failure_name(result), result
        assert result == "[(2S)-2-(methylamino)propanamido]acetic acid", result
        assert same_key(smi, result), result

    def test_real_backlog_composed_n_methyl_and_c_amide(self):
        """Real 10-residue backlog witness with BOTH caps at once: a mono-
        N-methylated Ile N-terminus AND a C-terminal valinamide -- proves
        Lever C composes the two independently-detected caps correctly.
        Was NOT abstaining before this change (the general/composer
        pipeline already emitted an ugly-but-valid name for it), but this
        change reroutes it to the correct systematic acylamino-convention
        name via the (higher-priority) peptide dispatch -- asserted here as
        a round-trip regression guard, not an abstain->emit transition."""
        smi = (
            "CC[C@H](C)[C@H](NC)C(=O)N[C@@H](CO)C(=O)N1CCC[C@H]1C(=O)"
            "N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)"
            "N[C@@H](C)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(C)C)C(=O)"
            "N[C@H](C(N)=O)C(C)C"
        )
        #: reroutes through the substitutive PIN path
        # (V38-PEPTIDE-PIN-VERDICT.md); the general/composer path emits a valid,
        # RT-verified name that carries a leading (2S) descriptor, so it no
        # longer literally STARTS with 'N-methyl' (it contains it). Exact
        # emission asserted below; full-InChIKey round-trip verified.
        result = _dt_name_compound(smi)
        assert not is_failure_name(result), result
        # change-asserted-value (lane/l17 item 7): was '(2S)-N-methylisoleucyl...valinamide'; the orphan leading '(2S)', the
        # descriptor of the C-terminal valine hoisted to the front by the parent that was the CONH2 end, is gone now that the
        # parent amide is chosen by / (rules.amide_parent). MACHINE-CONFIRMED: ``same_key`` below reads the new name
        # back to the input's full InChIKey (every residue is L, which OPSIN reads from the retained names), and the old
        # name read back alike, so neither spelling lost a stereocentre. UNVERIFIED: that the book prefers the name without
        # the leading descriptor (the book gives no peptide name of this kind); it rests on the retained residue names
        # carrying the configuration, and the open item is a Blue Book / peptide example with a C-terminal amide.
        assert result == (
            "N-methylisoleucylserylprolylalanylleucylleucylalanyl"
            "serylleucylvalinamide"
        ), result
        assert result.endswith("valinamide"), result
        assert same_key(smi, result), result


# ── Side-chain-acid / side-chain-amide trap: must abstain, never misplace ─


@pytest.mark.unit
class TestSideChainTrapAbstains:
    """the contributor guide-flagged risk: a Glu/Asp side-chain acid, or an Asn/Gln
    side-chain amide, sitting alongside a genuine C-terminal amide cap must
    never be misread as (or compete with) that cap. Both must ABSTAIN
    rather than emit a guessed/misplaced name."""

    def test_glutamine_cterm_amide_abstains(self):
        """Real 26-residue backlog witness whose C-terminal residue is
        glutamine with ITS OWN alpha-carboxyl also amidated -- two primary
        carboxamides on one residue (side chain + alpha), which
        `_strip_terminal_amide`'s exactly-one-match requirement correctly
        refuses to disambiguate. Must abstain, never emit a wrong/misplaced
        name."""
        smi = (
            "CC[C@H](C)[C@H](NC(=O)CN)C(=O)NCC(=O)N[C@@H](C)C(=O)"
            "N[C@H](C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)"
            "N[C@H](C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](C(=O)N[C@H](C(=O)NCC(=O)"
            "N[C@@H](CC(C)C)C(=O)N1CCC[C@H]1C(=O)N[C@@H](C)C(=O)"
            "N[C@@H](CC(C)C)C(=O)N[C@H](C(=O)N[C@@H](CO)C(=O)"
            "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@H](C(=O)N[C@@H](CCCCN)C(=O)"
            "N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCNC(=N)N)"
            "C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CCC(N)=O)C(N)=O)[C@@H](C)CC)"
            "[C@@H](C)CC)[C@@H](C)O)[C@@H](C)O)C(C)C)C(C)C"
        )
        result = _dt_name_compound(smi)
        assert is_failure_name(result), (
            f"glutamine C-terminal double-amide must abstain (ambiguous "
            f"side-chain vs alpha amide), got: {result}"
        )

    def test_c_terminal_aldehyde_out_of_scope_abstains(self):
        """Real 16-residue backlog witness whose C-terminus is a reduced
        ALDEHYDE (a protease-inhibitor-style cap), not a primary
        carboxamide -- outside this lever's scope by design. Must still
        abstain cleanly (no false-positive strip), not regress into a
        wrong name."""
        smi = (
            "CCC(C)C(NC(=O)C(CCSC)NC(=O)CNC(=O)C(CCCCN)NC(=O)C(C)NC(=O)"
            "C(CC(C)C)NC(=O)C(N)CCSC)C(=O)NC(CCCCN)C(=O)NC(CCCCN)C(=O)"
            "NC(Cc1ccccc1)C(=O)N1CCCC1C(=O)NC(CC(N)=O)C(=O)N1CCCC1C(=O)"
            "NC(Cc1ccc(C)cc1)C(=O)NC(C(=O)NC(C=O)CC(C)C)C(C)O"
        )
        result = _dt_name_compound(smi)
        assert is_failure_name(result), (
            f"C-terminal aldehyde cap must abstain (out of this lever's "
            f"scope), got: {result}"
        )


# ── Regression: existing standard/Lever-A peptides unchanged ─────────────


@pytest.mark.unit
class TestRegressionUnaffected:
    """Spot-checks that ordinary standard-chain peptides and the existing
    Lever-A single-residue exclusion are unchanged (PIN byte-identical)."""

    def test_glycylglycine_unchanged(self):
        #: substitutive PIN (V38-PEPTIDE-PIN-VERDICT.md); RT verified.
        assert name_compound("NCC(=O)NCC(=O)O") == "(2-aminoacetamido)acetic acid"

    def test_alanylglycine_unchanged(self):
        #: substitutive PIN (V38-PEPTIDE-PIN-VERDICT.md); RT verified.
        assert (name_compound("N[C@@H](C)C(=O)NCC(=O)O")
                == "[(2S)-2-aminopropanamido]acetic acid")

    def test_standard_tripeptide_unchanged(self):
        """Ile-Ala-Pro, a free-COOH/free-NH2 standard tripeptide -- must be
        completely unaffected by Lever C (neither cap is present)."""
        smi = "CC[C@H](C)[C@H](N)C(=O)N[C@@H](C)C(=O)N1CCC[C@@H]1C(=O)O"
        result = _dt_name_compound(smi)
        assert not is_failure_name(result), result
        assert "N-methyl" not in result, result
        assert not result.endswith("amide"), result
        assert same_key(smi, result), result

    def test_single_residue_n_acetylglycine_unchanged(self):
        result = name_compound("CC(=O)NCC(=O)O")
        assert result == "acetamidoacetic acid", result
