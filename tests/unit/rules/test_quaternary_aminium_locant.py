""" /: a quaternary ammonium whose parent branch is
substituted must number that branch with the N-attached carbon as C1 (the amine
principal group takes the lowest locant). Regression: `CC[N+](CC)(CC)CCF` emitted
`1-fluoro-N,N,N-triethylethanaminium` (fluoro at C1 — the chain numbered from the
wrong end when equal-length ethyl co-branches competed); correct is `2-fluoro-...`.
Mission-corpus (drug-like PubChem) bounded lever — see DRUGLIKE-BASELINE-2026-08-10.md.
"""
import pytest
from rdkit import Chem
from orthonym.rules.ions import name_quaternary_aminium
from orthonym.namer import _validity_gate_name_to_smiles


def _site(mol):
    for a in mol.GetAtoms():
        if (a.GetSymbol() == "N" and a.GetFormalCharge() == 1
                and a.GetTotalNumHs() == 0 and a.GetDegree() >= 4):
            return {"atom_idx": a.GetIdx()}
    return None


def _canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


@pytest.mark.parametrize("smiles", [
    "CC[N+](CC)(CC)CCF",   # 2-fluoro-N,N,N-triethylethan-1-aminium (the regression)
    "FCC[N+](C)(C)C",      # 2-fluoro-N,N,N-trimethylethan-1-aminium (was already OK)
    "ClCC[N+](CC)(CC)CC",  # 2-chloro-N,N,N-triethyl... (mixed-branch chloro)
    "OCC[N+](CC)(CC)CC",   # 2-hydroxy-N,N,N-triethyl... (mixed-branch hydroxy)
])
def test_substituted_quaternary_aminium_locant_rt_exact(smiles):
    mol = Chem.MolFromSmiles(smiles)
    name = name_quaternary_aminium(mol, _site(mol))
    assert name, f"no name for {smiles}"
    opsin = _validity_gate_name_to_smiles(name)
    assert opsin is not None, f"OPSIN could not parse {name!r}"
    assert _canon(opsin) == _canon(smiles), (
        f"NOT rt_exact: {name!r} -> {_canon(opsin)} != {_canon(smiles)}")


# =============================================================================
# a phase : two independent bugs in `name_quaternary_aminium` that
# made a quaternary ammonium with an acid or phenol on the parent branch
# abstain (correctly, via -- 0-wrong held, but breadth was lost).
#
# Bug A (.1, ~ions.py:4183-4199): after the manual `principal_chain.
# reverse` that re-orients the N-carbon to C1, `features.atom_to_locant`
# was NOT recomputed -- it stayed keyed to the PRE-reversal chain order, so
# every downstream locant lookup (the aminium suffix locant in particular)
# disagreed with the chain it was nominally numbering. `[N+](C)(C)(C)
# CCCC(=O)O` emitted the wrong-topology '...propan-3-aminium' (
# correctly suppressed it -> abstain) instead of '...propan-1-aminium'.
#
# Bug B (.2, ~ions.py:4142-4143 injection site): `PG_ATTACHMENT_INDICES
# ['tertiary_amine'] == [1, 2, 3]` (rules/seniority.py) assumes the canonical
# 3-carbon tertiary-amine SMARTS shape (N + 3 C). The quaternary-N override
# injects a 4-carbon match (N + 4 C, a 5-tuple); `_pg_attachment_atoms`
# (rules/parent_selection.py) silently drops whichever branch lands at tuple
# position 4 -- RDKit neighbour-iteration order, not seniority. When the
# dropped branch is the one reaching a competing ring (e.g. a phenol several
# atoms down the chain), the PG-count tie sees 0-vs-0 (the 3
# surviving branches are bare methyls), falls through to the ring-senior-
# to-chain default, and the ring wins -- emptying `principal_chain` and
# losing the amine parent entirely. `C[N+](C)(C)CCc1ccc(O)cc1` (candicine
# cation, ChEBI CHEBI:3350) emitted a garbled ring-parent name (/
# OPSIN-unparseable -> abstain) instead of naming the amine chain with the
# phenol ring as a substituent.
#
# Both are RT-verified (opsin_parse + InChIKey match against the input) --
# see the quaternary-aminium-report.md for the verification transcript.
# =============================================================================

from orthonym.namer import Orthonym


def _pin_name(smiles):
    return Orthonym(style='pin').name(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    # Bug A: N-attached carbon must be C1 even though the chain also carries
    # a demoted carboxylic-acid ('carboxy') substituent at the far end.
    ("[N+](C)(C)(C)CCCC(=O)O", "3-carboxy-N,N,N-trimethylpropan-1-aminium"),
    # Bug B: the amine chain must win parent selection over the competing
    # phenol ring (ChEBI CHEBI:3350, candicine cation).
    # wp7 change-asserted-value: (the Blue Book) -- the substituted ethane
    # parent keeps its locant '1' (':43572' '2-aminoethan-1-aminium chloride (PIN)'). Was
    # '...-N,N,N-trimethylethanaminium'. OPSIN 2.9.0 full-InChIKey exact.
    ("C[N+](C)(C)CCc1ccc(O)cc1", "2-(4-hydroxyphenyl)-N,N,N-trimethylethan-1-aminium"),
])
def test_quaternary_aminium_bug_fixes_rt_exact(smiles, expected):
    name = _pin_name(smiles)
    assert name == expected, f"{smiles} -> {name!r}, expected {expected!r}"
    opsin_smiles = _validity_gate_name_to_smiles(name)
    assert opsin_smiles is not None, f"OPSIN could not parse {name!r}"
    assert _canon(opsin_smiles) == _canon(smiles), (
        f"NOT rt_exact: {name!r} -> {_canon(opsin_smiles)} != {_canon(smiles)}")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[N+](C)(C)C", "N,N,N-trimethylmethanaminium"),
    ("c1cccc[n+]1C", "1-methylpyridin-1-ium"),
    ("C[N+](C)(C)CCCC(=O)[O-]", "4-(trimethylazaniumyl)butanoate"),
    # / substitutive '-bis(aminium)' PIN (the Blue Book,:42366)
    ("C[N+](C)(C)CCCCCC[N+](C)(C)C",
     "N1,N1,N1,N6,N6,N6-hexamethylhexane-1,6-bis(aminium)"),
])
def test_quaternary_aminium_regressions_unchanged(smiles, expected):
    name = _pin_name(smiles)
    assert name == expected, f"{smiles} -> {name!r}, expected {expected!r}"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    # Task 8B RC2, the Blue Book): a quaternary N bonded DIRECTLY to an
    # (hetero)aromatic ring is named on the ring-amine parent (aniline is retained),
    # not the chain amine. This class USED to fail closed -- `_assemble_amine_name`
    # cannot name an aromatic ring parent ('_generate_ring_parent' returns '' ->
    # malformed 'anaminium') -- and is now built via a neutral primary-amine proxy.
    ("c1ccccc1[N+](C)(C)C", "N,N,N-trimethylanilinium"),        # the Blue Book (PIN)
    ("CC[N+](C)(C)c1ccccc1", "N-ethyl-N,N-dimethylanilinium"),  # mixed N-substituents
    ("C[N+](C)(C)c1ccc2ccccc2c1", "N,N,N-trimethylnaphthalen-2-aminium"),
    ("C[N+](C)(C)c1ccncc1", "N,N,N-trimethylpyridin-4-aminium"),  # heteroaromatic ring
])
def test_aryl_quaternary_aminium_now_built(smiles, expected):
    """Task 8B RC2, the Blue Book): aryl-quaternary-N ->...anilinium."""
    name = _pin_name(smiles)
    assert name == expected, f"{smiles} -> {name!r}, expected {expected!r}"


@pytest.mark.opsin_gate
def test_substituted_aryl_quaternary_aminium_fails_closed():
    """A SUBSTITUTED aromatic ring parent is deliberately DECLINED scope
    boundary): the bounded producer does not alphabetise the ring's C-substituent
    prefixes against the N-locant block, and OPSIN still parses the mis-spelled
    concatenation (the RT gate cannot catch it), so this must fail closed (abstain),
    never emit a mis-spelled name."""
    name = _pin_name("C[N+](C)(C)c1ccc(C)cc1")
    assert name == "unknown organic compound", (
        f"expected fail-closed abstention, got {name!r}")
