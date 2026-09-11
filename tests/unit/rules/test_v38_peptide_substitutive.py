"""
 (backbone-substitutive peptide -- CIP-locanted splice + ester C-terminus).

This increment extends `rules.peptides._try_backbone_substitutive` (the
last-resort, RT-gated peptide fallback added in a phase) with two new
capabilities, plus the dispatch-gate broadening that makes the second one
reachable end-to-end:

  * Site A -- CIP-locanted amido splice: a residue whose systematic free-acid
    name carries a leading stereodescriptor + locant (e.g.
    '(2S)-2-aminohexanoic acid') is now a valid "open" swap parent, and a
    residue is spliced into its real '(stereo)-N-amino' token as a systematic
    '(2S)-2-...amido' prefix, enclosure), rather than
    only the retained 'alanyl'/'glycyl' acyl-prepend form. So
    `CCCC[C@H](NC(=O)[C@H](C)N)C(=O)O` (Ala->norleucine, a free-COOH
    C-terminus that already dispatched) names
    '(2S)-2-[(2S)-2-aminopropanamido]hexanoic acid'.

  * Site B -- ESTER C-terminus parent: a peptide whose C-terminal residue's
    alpha-carboxyl is esterified ('...oate', NO free -COOH) is now named with
    a fully systematic 'methyl...oate' parent via
    `_name_ester_parent_systematic`, exposing the same '(stereo)-N-amino'
    splice point. So `COC(=O)[C@H](CC(C)C)NC(=O)[C@@H](N)CC(C)C` (Leu-Leu
    methyl ester) names
    'methyl (2S)-2-[(2S)-2-amino-4-methylpentanamido]-4-methylpentanoate'.

  * Dispatch gate: `amino_acids.is_peptide` gained an alpha-amino-ESTER
    alternative (`_TERMINAL_ESTER_ALPHA_SMARTS`) so an ester-C-terminus
    peptide (no free -COOH, no primary carboxamide) actually reaches
    `rules.peptides.name_peptide` instead of falling through to 'unknown
    organic compound'. This is the exact structural analogue of the
    primary-amide alternative already present. Scoped to an alpha-amino-acid
    ester + `count_peptide_bonds >= 1`, so ordinary esters are NOT re-routed.

Every candidate is RT-gated by `name_peptide`'s existing `_rt_verified` call
(full-InChIKey OPSIN round-trip): a wrong or unparseable guess degrades to
abstention, never ships (0-wrong ABSOLUTE). Targets below are byte-pinned AND
RT-verified.
"""

import random

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import name_compound
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse

# GATE ON (production default): this module tests exactly what `name_compound`
# emits for a real caller. Mirrors test_peptide_capped_termini.py's rationale.
pytestmark = pytest.mark.opsin_gate


def _best_effort(smiles: str) -> str:
    """Best-effort tier (default tier is a plain `name_compound`)."""
    return name_compound(
        smiles,
        general_fallback=True,
        general_fallback_unverified=True,
        allow_aromatic_general=True,
    )


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


# Site B (ester C-terminus) + Site A (CIP-locanted, free-COOH C-terminus).
LEU_LEU_OME = "COC(=O)[C@H](CC(C)C)NC(=O)[C@@H](N)CC(C)C"
LEU_LEU_OME_NAME = (
    "methyl (2S)-2-[(2S)-2-amino-4-methylpentanamido]-4-methylpentanoate"
)
ALA_NORLEUCINE = "CCCC[C@H](NC(=O)[C@H](C)N)C(=O)O"
ALA_NORLEUCINE_NAME = "(2S)-2-[(2S)-2-aminopropanamido]hexanoic acid"


@pytest.mark.unit
class TestSiteB_EsterCTerminus:
    """A peptide whose C-terminus is an alkyl ester ('...oate') -- was
    ABSTAIN ('unknown organic compound') because `is_peptide` did not match
    an ester C-terminus, so `name_peptide` was never reached."""

    def test_leu_leu_methyl_ester_byte_and_rt(self):
        result = name_compound(LEU_LEU_OME)
        assert not is_failure_name(result), result
        assert result == LEU_LEU_OME_NAME, result
        assert _full_rt(LEU_LEU_OME, result), result

    def test_ala_ala_methyl_ester_rt(self):
        """A second, smaller ester-C-terminus witness. Names the RIGHT
        molecule via the CIP-locanted splice (byte value pinned; both centres
        defined) and round-trips."""
        smi = "COC(=O)[C@@H](C)NC(=O)[C@H](C)N"
        result = name_compound(smi)
        assert not is_failure_name(result), result
        assert result == "methyl (2R)-2-[(2S)-2-aminopropanamido]propanoate", result
        assert _full_rt(smi, result), result


@pytest.mark.unit
class TestSiteA_CIPLocantedSplice:
    """A free-COOH C-terminus peptide whose parent residue is a NON-standard
    (norleucine) acid -- its systematic free-acid name carries a leading
    '(2S)-2-amino' the residue splices into."""

    def test_ala_norleucine_byte_and_rt(self):
        result = name_compound(ALA_NORLEUCINE)
        assert not is_failure_name(result), result
        assert result == ALA_NORLEUCINE_NAME, result
        assert _full_rt(ALA_NORLEUCINE, result), result


@pytest.mark.unit
class TestNonStandardLinearPeptides:
    """More non-standard / ester linear peptides that abstain or emit an
    uglier form today. Contract (0-wrong ABSOLUTE): each must either name the
    RIGHT molecule and round-trip, or stay a clean abstention -- NEVER a wrong
    molecule."""

    CASES = [
        # Gly-Gly methyl ester (achiral parent -> systematic swap declines;
        # falls through to the general pipeline, which names it validly).
        "COC(=O)CNC(=O)CN",
        # Ala-Gly ethyl ester (glycine ester parent, achiral).
        "CCOC(=O)CNC(=O)[C@H](C)N",
        # Phe-Gly methyl ester.
        "COC(=O)CNC(=O)[C@@H](N)Cc1ccccc1",
    ]

    @pytest.mark.parametrize("smi", CASES)
    def test_names_right_molecule_or_abstains(self, smi):
        result = name_compound(smi)
        if is_failure_name(result):
            return  # clean abstention is acceptable
        # If a name is emitted, it MUST describe the input molecule.
        assert _full_rt(smi, result), (smi, result)


@pytest.mark.unit
class TestDeterminism:
    """The same molecule named from randomized SMILES atom orderings must give
    byte-identical output (the contributor guide a project rule: determinism)."""

    @pytest.mark.parametrize(
        "smi,expected",
        [
            (LEU_LEU_OME, LEU_LEU_OME_NAME),
            (ALA_NORLEUCINE, ALA_NORLEUCINE_NAME),
        ],
    )
    def test_randomized_orderings_identical(self, smi, expected):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        rng = random.Random(0xC0FFEE)
        variants = {Chem.MolToSmiles(mol)}  # canonical form included
        for _ in range(4):
            order = list(range(mol.GetNumAtoms()))
            rng.shuffle(order)
            variants.add(
                Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
            )
        names = {v: name_compound(v) for v in variants}
        assert len(set(names.values())) == 1, names
        assert next(iter(names.values())) == expected, names


@pytest.mark.unit
class TestControlsUnchanged:
    """Byte-identity controls: molecules that must NOT change under the
    dispatch broadening. Standard peptides (free-COOH / primary-amide
    C-terminus) and non-peptide controls. Checked in BOTH the default and the
    best-effort tiers."""

    CONTROLS = [
        ("NCC(=O)NCC(=O)NCC(=O)O", "glycylglycylglycine"),
        (
            "CC(C)[C@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)NCC(=O)O",
            "valylphenylalanylglycine",
        ),
        (
            "CC(C)(N)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O",
            "2-amino-2-methylpropanoylalanylalanine",
        ),
        ("CCO", "ethanol"),
        ("c1ccccc1", "benzene"),
        # Ordinary esters WITHOUT a peptide bond must not be re-routed.
        ("COC(C)=O", "methyl acetate"),
        ("CCOC(=O)c1ccccc1", "ethyl benzoate"),
    ]

    @pytest.mark.parametrize("smi,expected", CONTROLS)
    def test_default_tier(self, smi, expected):
        assert name_compound(smi) == expected, (smi, name_compound(smi))

    @pytest.mark.parametrize("smi,expected", CONTROLS)
    def test_best_effort_tier(self, smi, expected):
        assert _best_effort(smi) == expected, (smi, _best_effort(smi))
