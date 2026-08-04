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


def test_ring_fragments_now_name_via_the_composite_branch():
    """RETIRED SCOPE TEST. Tasks 1-3 declined ring-bearing fragments and this test
    pinned that intermediate limitation; Task 4 deliberately removes it. Kept as a
    positive assertion rather than deleted, so the transition is visible in history
    instead of a test simply vanishing. Ring coverage proper lives in
    tests/unit/rules/test_terminal_fragment_composite.py."""
    mol = Chem.MolFromSmiles("C1CCCCC1")
    got = terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 0)
    assert got is not None and got.name == "cyclohexan-1-yl"


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


# ---------------------------------------------------------------------------
# Fix wave 1: a branch that is ITSELF compound (a branched sub-fragment, not a
# single simple substituent) must be spliced in with P-16.5.1.1 enclosing
# marks, and TWO IDENTICAL compound branches must use the DERIVED multiplier
# (bis/tris/...) rather than the basic one (di/tri/...) -- P-16.3.3(b)/
# P-16.3.5(a) (BlueBookV2.md:4857). Before this fix, the module spliced the
# recursive branch name in bare and always used SIMPLE_MULTIPLIERS, so the
# multi-locant case emitted a name OPSIN 2.9.0 cannot even parse:
# '(5,6-di1-methylethyldecyl)benzene' fails; '(5,6-bis(1-methylethyl)decyl)
# benzene' parses. Every expected value below was verified directly against
# `terminal_fragment_name` (2026-08-04) and matches the reproduction that
# motivated the fix.
#
# `_frag` always attaches at atom index 0 -- the first atom written in the
# SMILES -- so for 'CCCCC(C(C)C)CCCC' that is a TERMINAL carbon of the main
# chain, making the longest path from it the full 9-carbon backbone (nonyl)
# with the branch hanging at locant 5, matching Task 2's already-established
# deterministic longest-path backbone selection.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    # one compound branch (isopropyl-shaped), single locant
    ("CCCCC(C(C)C)CCCC",   "5-(1-methylethyl)nonyl"),
    # one compound branch that is itself a longer compound (isobutyl-shaped)
    ("CCCCC(CC(C)C)CCCC",  "5-(2-methylpropyl)nonyl"),
    # a tert-butyl-shaped branch
    ("CCCCC(C(C)(C)C)CCCC", "5-(1,1-dimethylethyl)nonyl"),
])
def test_compound_branch_gets_enclosing_marks(smiles, expected):
    mol, frag, attach = _frag(smiles)
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None, f"{smiles} refused"
    assert got.name == expected
    assert got.atoms == frozenset(frag)


def test_two_identical_compound_branches_use_derived_multiplier():
    """TWO IDENTICAL compound branches -> 'bis', not 'di', AND each occurrence
    still carries its own enclosing marks: '5,6-bis(1-methylethyl)decyl'.

    This is the exact shape that produced an OPSIN-UNPARSEABLE name before the
    fix ('5,6-di1-methylethyldecyl'): the multi-locant case is the critical
    one, since a single-locant compound branch happened to still parse via
    OPSIN leniency even though it was not a legal Blue Book spelling.
    """
    mol, frag, attach = _frag("CCCCC(C(C)C)C(C(C)C)CCCC")
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None
    assert got.name == "5,6-bis(1-methylethyl)decyl"
    assert got.atoms == frozenset(frag)


@pytest.mark.parametrize("smiles,expected", [
    # regression guard: a SIMPLE branch (bare 'methyl') is still cited bare,
    # with the BASIC multiplier -- these must NOT gain enclosing marks or
    # switch to bis/tris just because compound-branch handling now exists.
    ("CC(C)C",    "2-methylpropyl"),
    ("CC(C)(C)C", "2,2-dimethylpropyl"),
])
def test_simple_branch_still_bare_with_basic_multiplier(smiles, expected):
    mol, frag, attach = _frag(smiles)
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None, f"{smiles} refused"
    assert got.name == expected
    assert got.atoms == frozenset(frag)


# ---------------------------------------------------------------------------
# Task 3: backbone unsaturation (ene/yne). The explicit '-1-yl' form is used
# throughout ('but-3-en-1-yl', never the bare '...enyl' contraction) --
# OPSIN-verified 2026-08-04 that '(2-oxabut-3-en-1-yl)benzene' parses to the
# right structure.
#
# ⚠ TWO of the task brief's four expected values were corrected here
# (2026-08-04), verified directly from atom H-count + bond connectivity (no
# OPSIN needed -- this is pure structural bookkeeping), independently of what
# this module's own implementation happens to emit:
#
#   'C=CCC': atom 0 has 2 H and is DOUBLE-bonded to atom 1 -- it IS the
#   alkene-bearing terminal carbon of but-1-ene (CH2=CH-CH2-CH3), not the
#   methyl terminal. `_frag` attaches at atom 0, and the free valence MUST
#   receive locant 1 (only one numbering direction exists from a terminal
#   atom of an unbranched chain), which forces the double bond onto locants
#   1,2: 'but-1-en-1-yl'. The brief's 'but-3-en-1-yl' names the substituent
#   from the OTHER terminal (methyl end, atom 3) of the SAME parent alkene --
#   a chemically different group ('but-1-en-1-yl' is -CH=CH-CH2-CH3;
#   'but-3-en-1-yl' is -CH2-CH2-CH=CH2).
#
#   'C#CCC': identical shape -- atom 0 has 1 H and is TRIPLE-bonded to atom 1
#   (the terminal alkyne carbon of but-1-yne), so attach=0 gives
#   'but-1-yn-1-yl', not the brief's 'but-3-yn-1-yl'.
#
# 'CC=CC' (atom 0 is the CH3 terminal, double bond is at position 1-2) and
# 'COC=C' (atom 0 is CH3, the alkene is at the far end) both already attach at
# the saturated end, so their brief values are unchanged and check out.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("C=CCC",   "but-1-en-1-yl"),
    ("CC=CC",   "but-2-en-1-yl"),
    ("C#CCC",   "but-1-yn-1-yl"),
    # replacement + unsaturation together
    ("COC=C",   "2-oxabut-3-en-1-yl"),
])
def test_unsaturated_backbone(smiles, expected):
    mol, frag, attach = _frag(smiles)
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None, f"{smiles} refused"
    assert got.name == expected
    assert got.atoms == frozenset(frag)


def test_a_saturated_backbone_is_unaffected_by_the_unsaturation_code():
    """Regression guard for Task 3: adding ene/yne handling must not change the
    saturated spelling. (This asserts a NAME, not a refusal -- see the separate
    refusal test below.)"""
    mol = Chem.MolFromSmiles("CCCC")
    got = terminal_fragment_name(mol, set(range(4)), 0)
    assert got is not None and got.name == "butyl"


def test_an_unspellable_bond_order_refuses():
    """A bond order with no morpheme must refuse, not be spelled as single.

    Spelling an unrepresentable bond as single denotes a DIFFERENT molecule,
    which is the failure mode this whole phase exists to remove. Built with an
    explicitly aromatic acyclic bond, which the ene/yne branch cannot express.
    """
    mol = Chem.RWMol(Chem.MolFromSmiles("CCCC"))
    mol.GetBondBetweenAtoms(1, 2).SetBondType(Chem.BondType.AROMATIC)
    got = terminal_fragment_name(mol.GetMol(), set(range(4)), 0)
    assert got is None


# ---------------------------------------------------------------------------
# Fix wave 1: MULTIPLIED backbone unsaturation. Before this fix, two-or-more
# double (or triple) bonds of the SAME type on a backbone omitted BOTH the
# multiplying prefix ('di'/'tri') AND the linking vowel 'a', e.g.
# 'pent-1,3-en-1-yl' instead of 'penta-1,3-dien-1-yl' -- a string OPSIN 2.9.0
# cannot parse at all, even though `terminal_fragment_name` still returned a
# populated result with a COMPLETE `atoms` set (the worst outcome: a clean
# success carrying a name that denotes nothing).
#
# Every expected value below was OPSIN-verified 2026-08-04 by wrapping the
# token in a parent and comparing the canonical SMILES to the independently
# built expected structure:
#   (penta-1,3-dien-1-yl)benzene         -> CC=CC=Cc1ccccc1
#   (buta-1,2-dien-1-yl)benzene          -> CC=C=Cc1ccccc1      (allene)
#   (penta-1,3-diyn-1-yl)benzene         -> CC#CC#Cc1ccccc1
#   (hexa-1,3,5-trien-1-yl)benzene       -> C=CC=CC=Cc1ccccc1
#   (hexa-1,3-dien-5-yn-1-yl)benzene     -> C#CC=CC=Cc1ccccc1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    # two double bonds -> 'di' + linking 'a'
    ("C=CC=CC",     "penta-1,3-dien-1-yl"),
    # two double bonds, CUMULATED (allene) at the far end -> adjacent locants
    ("C=C=CC",      "buta-1,2-dien-1-yl"),
    # two triple bonds -> 'di' + linking 'a', same rule as ene
    ("C#CC#CC",     "penta-1,3-diyn-1-yl"),
    # three double bonds -> 'tri' + linking 'a'
    ("C=CC=CC=C",   "hexa-1,3,5-trien-1-yl"),
    # mixed: TWO enes (multiplied, 'di') + ONE yne (not multiplied) -- 'ene'
    # precedes 'yne' (P-31.1.1.1) and the linking 'a' appears ONCE, on the stem
    ("C=CC=CC#C",   "hexa-1,3-dien-5-yn-1-yl"),
])
def test_multiplied_backbone_unsaturation(smiles, expected):
    mol, frag, attach = _frag(smiles)
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None, f"{smiles} refused"
    assert got.name == expected
    assert got.atoms == frozenset(frag)


def test_single_of_each_type_is_byte_identical_to_before_the_fix():
    """Regression pin: ONE ene and ONE yne together must NOT gain a
    multiplier or a linking 'a' -- neither suffix is multiplied here, so
    'pent-1-en-3-yn-1-yl' is unchanged (no 'a', no 'di'/'tri'). This is the
    case invariant 1's "must not change" clause protects.
    """
    mol, frag, attach = _frag("C=CC#CC")
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None
    assert got.name == "pent-1-en-3-yn-1-yl"
    assert got.atoms == frozenset(frag)


def test_replacement_prefix_with_multiplied_ene():
    """A replacement prefix composes normally with a MULTIPLIED ene: the
    heteroatom's own locant ('2-oxa') is unaffected by the linking 'a' the
    multiplied suffix adds to the hydrocarbon stem it's fused onto.

    Fragment: C(attach)-O-CH=C=CH2 -- a cumulated diene (locants 3,4) past an
    oxa replacement at backbone position 2. OPSIN-verified: parses inside
    '(2-oxapenta-3,4-dien-1-yl)benzene' -> C=C=COCc1ccccc1.
    """
    mol, frag, attach = _frag("COC=C=C")
    got = terminal_fragment_name(mol, frag, attach)
    assert got is not None
    assert got.name == "2-oxapenta-3,4-dien-1-yl"
    assert got.atoms == frozenset(frag)
