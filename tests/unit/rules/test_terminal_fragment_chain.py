"""The terminal fragment namer: straight saturated chains.

Every expectation here was OPSIN-verified on 2026-08-04 by wrapping the token in
a parent -- `(2-oxabutyl)benzene` -> `C(OCC)c1ccccc1`. A BARE token does not
parse, so never assert against `opsin("2-oxabutyl")`.

The completeness invariant is the point of the module: `result.atoms` must equal
the input fragment. An atom-short name is the defect this phase exists to remove,
so it is asserted on every case rather than spot-checked.
"""
import pytest
from rdkit import Chem

from orthonym.rules.terminal_fragment import (
    TerminalFragmentName,
    terminal_fragment_name,
)


def _frag(smiles, attach_smarts_idx=0):
    """Whole molecule as one fragment, attached at atom `attach_smarts_idx`."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol, set(range(mol.GetNumAtoms())), attach_smarts_idx


@pytest.mark.parametrize("smiles,expected", [
    # plain alkyl -- no replacement prefix
    ("CC",      "ethyl"),
    ("CCCC",    "butyl"),
    ("C",       "methyl"),
    # one heteroatom: locant 1 is the attachment atom, numbering runs away
    ("COCC",    "2-oxabutyl"),
    ("CSCC",    "2-thiabutyl"),
    ("CNCC",    "2-azabutyl"),
    ("CCOCC",   "3-oxapentyl"),
    # two of a kind -> multiplier
    ("COCOC",   "2,4-dioxapentyl"),
    # two kinds -> cited in seniority order, each with its locant
    ("CSCOCC",  "4-oxa-2-thiahexyl"),
    ("COCNC",   "2-oxa-4-azapentyl"),
    # group 14 heteroatoms are in the admitted table
    ("C[SiH2]CC", "2-silabutyl"),
])
def test_straight_saturated_chain(smiles, expected):
    mol, frag, attach = _frag(smiles)
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None, f"{smiles} refused"
    assert got.name == expected
    assert got.basis == "chain"


@pytest.mark.parametrize("smiles", ["CC", "COCC", "CSCOCC", "C[SiH2]CC"])
def test_completeness_invariant_every_atom_accounted(smiles):
    mol, frag, attach = _frag(smiles)
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None
    assert got.atoms == frozenset(frag), (
        "an atom-short name is the defect this module exists to remove"
    )


def test_attachment_atom_is_locant_one():
    mol, frag, attach = _frag("COCC")
    got = terminal_fragment_name(mol, frag, attach)
    assert got.numbering[attach] == 1


def test_off_table_element_refuses_rather_than_inventing_a_morpheme():
    """Zn has no row in Table 1.5, so no morpheme can be spelled for it.

    `ring_replacement.build_replacement_prefix` reports it in `unexpressed` and
    its contract obliges the caller to refuse -- inventing 'zna' is how
    `3-znaspiro[5.5]undecane` happened.
    """
    mol = Chem.MolFromSmiles("C[Zn]CC")
    assert mol is not None
    got = terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 0)
    assert got is None


def test_ring_fragments_are_still_declined_in_this_task():
    """Branched acyclic fragments now name (Task 2); ring-bearing ones are Task 4."""
    mol = Chem.MolFromSmiles("C1CCCCC1")
    assert terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 0) is None


def test_never_raises_on_degenerate_input():
    mol = Chem.MolFromSmiles("CCO")
    assert terminal_fragment_name(mol, set(), 0) is None
    assert terminal_fragment_name(None, {0}, 0) is None
    assert terminal_fragment_name(mol, {0}, 99) is None


# ---------------------------------------------------------------------------
# Task 2: branches (recursion). ``_frag`` always attaches at atom index 0 --
# the FIRST atom written in the SMILES -- so the expected name is whatever
# name that specific atom, as the free-valence locant 1, actually gets under
# the deterministic longest-path backbone selection Task 1 already built.
#
# ⚠ Every expected value below was corrected from the task brief and
# OPSIN-verified by wrapping the token in `(token)benzene` and comparing the
# canonical structure to the fragment attached at atom 0 (2026-08-04):
#   (2-methylpropyl)benzene    -> PhCH2CH(CH3)2      == CC(C)C attach@0
#   (2-methylbutyl)benzene     -> PhCH2CH(CH3)CH2CH3 == CC(C)CC attach@0
#   (3-methyl-2-oxabutyl)benzene -> PhCH2OCH(CH3)2   == COC(C)C attach@0
#   (2,2-dimethylpropyl)benzene -> PhCH2C(CH3)3       == CC(C)(C)C attach@0
# The brief's originals (1-methylpropyl / 1-methylbutyl / 1-methyl-2-oxapropyl
# / 1,1-dimethylpropyl) all name a DIFFERENT molecule -- e.g. "1-methylpropyl"
# is sec-butyl, which requires an unbranched n-butane skeleton attached at an
# INTERNAL atom; "CC(C)C" is isobutane (2-methylpropane) and atom 0 is a leaf,
# so no attach point in it can ever produce sec-butyl's name. Every corrected
# value here is the textbook name for its group (isobutyl, neopentyl, ...).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    # backbone is the longest path from the attachment atom; the remaining
    # methyl is a locanted prefix
    ("CC(C)C",      "2-methylpropyl"),
    ("CC(C)CC",     "2-methylbutyl"),
    # branch on a heteroatom-bearing backbone
    ("COC(C)C",     "3-methyl-2-oxabutyl"),
    # two identical branches -> multiplier
    ("CC(C)(C)C",   "2,2-dimethylpropyl"),
])
def test_branched_saturated_chain(smiles, expected):
    mol, frag, attach = _frag(smiles)
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None, f"{smiles} refused"
    assert got.name == expected


@pytest.mark.parametrize("smiles", ["CC(C)C", "CC(C)(C)C", "COC(C)C"])
def test_branched_fragments_are_complete(smiles):
    mol, frag, attach = _frag(smiles)
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None
    assert got.atoms == frozenset(frag)


def test_a_branch_that_cannot_be_named_refuses_the_whole_fragment():
    """Partial success is the atom-drop bug wearing a different hat.

    If a branch has no admitted morpheme, the fragment must refuse -- emitting
    the backbone alone would drop the branch's atoms, which is precisely what
    this module exists to prevent.

    NOTE: for THIS molecule the off-table Zn actually lands ON the longest-path
    backbone (not in a branch) -- ``CC([Zn]C)C`` from atom 0 is 5 atoms long
    through Zn (0,1,2,3) versus 3 atoms stopping at the other leaf (0,1,4), so
    the greedy longest-path selector prefers the Zn-containing path. The whole
    fragment still refuses (via the backbone's own `rp.unexpressed` check,
    inherited from Task 1), so the assertion holds, but it does not by itself
    exercise the NEW branch-recursion refusal this task adds -- see the next
    test for that.
    """
    mol = Chem.MolFromSmiles("CC([Zn]C)C")
    assert mol is not None
    assert terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 0) is None


def test_an_unnameable_branch_off_the_backbone_refuses_the_whole_fragment():
    """Genuinely exercises branch-recursion refusal (added scope, beyond the brief).

    ``CCC([Zn])CC`` from atom 0: the longest path (0,1,2,4,5, five carbons)
    leaves Zn as a one-atom BRANCH hanging off backbone locant 3 -- confirmed
    via ``_backbone_from`` directly. The branch's recursive
    ``_terminal_fragment_name`` call returns None (Zn is off Table 1.5), which
    must refuse the whole fragment rather than emit the 5-carbon backbone alone
    and silently drop the Zn atom.
    """
    mol = Chem.MolFromSmiles("CCC([Zn])CC")
    assert mol is not None
    assert terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 0) is None


# ---------------------------------------------------------------------------
# Coordinator-added: EMBEDDED fragments -- a proper subset of a larger
# molecule, with the attachment atom bonded to an atom OUTSIDE the fragment.
# Every test above passes the WHOLE molecule as the fragment; the real call
# sites hand this module an embedded piece, so this shape must be covered too.
# ---------------------------------------------------------------------------
def test_embedded_straight_chain_fragment_bonded_to_an_outside_ring():
    """CCOc1ccccc1 (phenetole): name only the ethyl-oxy atoms {C,C,O}, attached
    at the O, which bonds to the ring atom OUTSIDE the fragment."""
    mol = Chem.MolFromSmiles("CCOc1ccccc1")
    assert mol is not None
    frag = {0, 1, 2}         # the two chain carbons + the ether oxygen
    attach = 2                # the O; bonded to ring atom 3, which is NOT in frag
    assert mol.GetAtomWithIdx(attach).GetSymbol() == "O"
    ring_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
                      if n.GetIdx() not in frag]
    assert ring_neighbors, "attach must be bonded to an atom outside the fragment"

    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None
    assert got.name == "1-oxapropyl"
    assert got.atoms == frozenset(frag)


def test_embedded_branched_fragment_bonded_to_an_outside_ring():
    """CC(C)Cc1ccccc1 (isobutylbenzene): name only the isobutyl carbons
    {C,C,C,C}, attached at the CH2 that bonds to the ring OUTSIDE the fragment."""
    mol = Chem.MolFromSmiles("CC(C)Cc1ccccc1")
    assert mol is not None
    frag = {0, 1, 2, 3}       # the four isobutyl carbons
    attach = 3                 # the CH2; bonded to ring atom 4, which is NOT in frag
    ring_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
                      if n.GetIdx() not in frag]
    assert ring_neighbors, "attach must be bonded to an atom outside the fragment"

    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None
    assert got.name == "2-methylpropyl"
    assert got.atoms == frozenset(frag)
