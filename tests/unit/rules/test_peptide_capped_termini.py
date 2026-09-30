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
from rdkit.Chem import inchi

from orthonym import name_compound
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

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
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)


# GATE ON (production default) -- see test_peptide_breadth.py's identical
# rationale: this module tests exactly what `name_compound` emits for a
# real caller.
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
        assert _full_rt(self.ILE_TO_PHE_NH2, result), result

    def test_leucinamide_cterm(self):
        result = _dt_name_compound(self.ILE_TO_LEU_NH2)
        assert not is_failure_name(result), result
        assert result.endswith("leucinamide"), result
        assert _full_rt(self.ILE_TO_LEU_NH2, result), result

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
        assert result == "2-[(2S)-2-aminopropanamido]acetamide", result
        assert _full_rt(smi, result), result

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
        assert _full_rt(smi, result), result
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
        assert _full_rt(smi, result), result

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
        assert result == (
            "(2S)-N-methylisoleucylserylprolylalanylleucylleucylalanyl"
            "serylleucylvalinamide"
        ), result
        assert result.endswith("valinamide"), result
        assert _full_rt(smi, result), result


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
        assert _full_rt(smi, result), result

    def test_single_residue_n_acetylglycine_unchanged(self):
        result = name_compound("CC(=O)NCC(=O)O")
        assert result == "acetamidoacetic acid", result
