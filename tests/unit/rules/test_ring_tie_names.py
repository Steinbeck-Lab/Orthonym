"""Two rings of equal seniority joined by =CH-, tied on the number and the locants of their prefixes: the parent
is chosen by and then on the two names.

 "CRITERIA RELATED TO ALPHANUMERICAL ORDER OF NAMES" (the Blue Book): "The preferred IUPAC name is the
name that is earlier in alphanumerical order (see. Alphabetic letters are considered first in the order that
they appear in the name; all Roman letters are considered before any italic letters, unless the latter are used as
locants or are a part of a compound or composite locant, for example, 'N' and '4a'. Then, if still there is a
choice, numerical locants are considered in the order of their appearance in the name."

 under " CRITERIA RELATED TO NUMBER AND LOCATION OF SUBSTITUENT GROUPS" (:21791): "The preferred IUPAC
name is based on the senior parent structure that has the lower locant or set of locants for substituents cited as
prefixes to the parent structure (other than 'hydro/dehydro' prefixes) in their order of citation in the name."

The criteria of that come before are applied first: (h) the lower locants of the suffix,
:21305, under " SENIORITY CRITERIA APPLICABLE TO RINGS, RING SYSTEMS, OR ACYCLIC CHAINS",:21014: "The senior
ring, ring system, or principal chain has the lower locants for characteristic groups cited as a suffix"; 'octane-1,7-
diol (PIN) is senior to octane-1,8-diol (PIN)',:21327), (k) the greater number of isotopically modified atoms
,:21477: "The senior parent structure contains the greater number of isotopically modified atoms or
groups";.2 to.6,:21490-:21523, the nuclides of higher atomic number and mass number and the locants of the modified
atoms), and (l) stereogenic centres,:21529), which is not worked out: applies "when there
is a choice between parent structures that differ only by 'Z' and 'E' configurations", which two rings of different
groups are not.

The Blue Book gives no worked example with two rings joined by =CH-; the row names are the two rules applied to the
two names each molecule has (each ring as the parent). They were also derived by an independent implementation of
the rules (the tie panel of topic P45: a composer of both names from the text of the book, 1,625 molecules, both
tiers, 0 disagreements) and every name is read back by a fresh OPSIN call to the input. The function-level tests use
the prefix names of the book's own examples.
"""
import random

import pytest
from rdkit import Chem

from orthonym.rules.ring_tie_names import decide_by_names, senior_ring_by_names
from tests.support.rt_assert import name_is_rt_exact
from tests.support.pin_tiers import (
    assert_declined_at_default,
    assert_pin_at_both_tiers,
    name_breadth,
    name_default,
)

pytestmark = pytest.mark.opsin_gate

RING = list(range(6))        # a six-membered ring, the atom at position i is the i-th round the ring


# --- the comparison on the prefix names of the Blue Book's own examples --------------------------------------
# Each ring is the list of (ring position, prefix name) of one candidate parent. The rings are numbered by,
# so the positions are chosen to give the locants of the example; the names are the book's.

def _decide(named_a, named_b, suffix=((), ())):
    return decide_by_names([RING, RING], [list(suffix[0]), list(suffix[1])], [named_a, named_b])


def test_p45_5_example_1_bromo_before_dibromo():
    """ example (1) (:22243): '1-bromo-4-chloro-2-{2-[(1,4-dibromonaphthalen-2-yl)methoxy]ethyl}naphthalene (PIN)
    [not 1,4-dibromo-2-{[2-(1-bromo-4-chloronaphthalen-2-yl)ethoxy]methyl}naphthalene; in both names the locant set
    is '1,2,4', and the locants appear in the name in the same order set '1,4,2' so no decision can be made by
     or; but 'bromo' in the PIN is earlier alphabetically than 'dibromo']"."""
    pin = [(0, "bromo"), (3, "chloro"), (1, "2-[(1,4-dibromonaphthalen-2-yl)methoxy]ethyl")]
    other = [(0, "bromo"), (3, "bromo"), (1, "[2-(1-bromo-4-chloronaphthalen-2-yl)ethoxy]methyl")]
    assert _decide(pin, other) == 0
    assert _decide(other, pin) == 1


def test_p45_5_example_4_difluoro_before_dinitro():
    """ example (4) (:22260): '4-(1,2-difluoropropyl)-5,6-dinitroheptanoic acid (PIN) [not 4-(1,2-dinitropropyl)-
    5,6-difluoroheptanoic acid; 'difluoro' in the PIN is earlier alphabetically than 'dinitro']': the compound prefix
    is alphabetized at the first letter of its complete name,:3477), 'dinitro' at 'nitro',
    :3448, 'multiplicative prefixes... do not alter the alphabetical order already established')."""
    pin = [(0, "1,2-difluoropropyl"), (1, "nitro"), (2, "nitro")]
    other = [(0, "1,2-dinitropropyl"), (1, "fluoro"), (2, "fluoro")]
    assert _decide(pin, other) == 0


def test_p15_3_2_4_3_bromochloro_before_bromophenyl():
    """ "When a choice is possible, unprimed locants are assigned to the identical parent structure having
    the substituent cited first in alphanumerical order" (:6273; example:6283):
    '1-bromo-3-[(3-chlorophenyl)methyl]benzene (PIN) [not 1-[(3-bromophenyl)methyl]-3-chlorobenzene; 'bromochloro' is
    preferred alphanumerically to 'bromophenyl'; see ]'."""
    pin = [(0, "bromo"), (2, "(3-chlorophenyl)methyl")]
    other = [(0, "(3-bromophenyl)methyl"), (2, "chloro")]
    assert _decide(pin, other) == 0


def test_p45_5_numerals_when_the_letters_are_the_same():
    """ (:22234) "Then, if still there is a choice, numerical locants are considered in the order of their
    appearance in the name", and "Unsymmetrically substituted component units" (:6400, example
    :6407): "'1-chloroethoxy' precedes '2-chloroethoxy' in alphanumerical order; see "."""
    assert _decide([(0, "1-chloroethoxy")], [(0, "2-chloroethoxy")]) == 0
    assert _decide([(0, "2-chloroethoxy")], [(0, "1-chloroethoxy")]) == 1


def test_p45_2_3_decides_before_the_letters():
    """ (:21791) comes before, so the locants in the order of citation decide against the letters:
    both rings have the locant set 1,2,4; ring 0 cites bromo 2, chloro 1, fluoro 4 (2,1,4) and ring 1 cites bromo 1,
    chloro 2, iodo 4 (1,2,4), which is lower, although 'bromochlorofluoro' is earlier than 'bromochloroiodo'."""
    ring0 = [(0, "bromo"), (1, "chloro"), (4, "fluoro")]
    ring1 = [(0, "bromo"), (1, "chloro"), (3, "iodo")]
    assert _decide(ring0, ring1) == 1
    assert _decide(ring1, ring0) == 0


def test_p45_3_and_p45_4_stand_between_p45_2_3_and_p45_5():
    """ (:22172, nonstandard bonding numbers of substituents) and (:22208, isotopic modification of
    substituents) are applied after and before: where the locants in the order of citation
    decide, they decide whatever the substituents are; where they tie, the letters are not reached, because a
    criterion in between is not worked out."""
    deciding_a = [(0, "bromo"), (1, "chloro")]
    deciding_b = [(0, "bromo"), (2, "chloro")]
    assert decide_by_names([RING, RING], [[], []], [deciding_a, deciding_b], between=True) == 0
    tie_a, tie_b = [(0, "chloro"), (3, "methyl")], [(0, "bromo"), (3, "methyl")]
    assert decide_by_names([RING, RING], [[], []], [tie_a, tie_b]) == 1
    assert decide_by_names([RING, RING], [[], []], [tie_a, tie_b], between=True) is None


def test_a_string_of_letters_that_starts_the_other_is_not_decided():
    """The rest of the name (the same parent hydride and suffix) would decide, and it is not built: undecided."""
    assert _decide([(0, "methyl")], [(0, "methylcyclohexyl")]) is None


def test_the_same_names_are_not_decided():
    assert _decide([(0, "chloro"), (3, "methyl")], [(0, "chloro"), (3, "methyl")]) is None


def test_no_comparison_inside_a_branch_named_as_a_sort_key():
    """``perception.chains._P45_NAMING_BRANCHES``: a comparison started while branches are named as sort keys is
    not made (it would re-enter itself through every branch it names)."""
    from orthonym.perception.chains import _P45_NAMING_BRANCHES
    mol = Chem.MolFromSmiles("CC1CCC(CC1)=CC1CCC(CC)CC1")
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    cycles = [list(r) for r in mol.GetRingInfo().AtomRings()]
    assert senior_ring_by_names(mol, rings, cycles, [[], []], set()) in (0, 1)
    token = _P45_NAMING_BRANCHES.set(True)
    try:
        assert senior_ring_by_names(mol, rings, cycles, [[], []], set()) is None
    finally:
        _P45_NAMING_BRANCHES.reset(token)


def test_the_labels_of_the_parents_do_not_stop_p45_5_but_those_of_a_substituent_do():
    """ compared the labels of the parents before began; (:22208) is the criterion for the labels of the
    SUBSTITUENTS and is not worked out. So decides a tie when the labelled molecule has its labels on the rings, and is
    not reached when a label sits on a substituent (``ring_tie_names.senior_ring_by_names``, the labelled molecule read as
    ``locant_omission.isotopic_skeleton_scope`` declares it)."""
    from orthonym.assembly.locant_omission import isotopic_skeleton_scope
    skeleton = Chem.MolFromSmiles("CC1CCC(CC1)=CC1CCC(CC)CC1")
    rings = [set(r) for r in skeleton.GetRingInfo().AtomRings()]
    cycles = [list(r) for r in skeleton.GetRingInfo().AtomRings()]
    def decide(labelled):
        with isotopic_skeleton_scope(Chem.MolFromSmiles(labelled)):
            return senior_ring_by_names(skeleton, rings, cycles, [[], []], set())
    assert decide("CC1CCC(CC1)=CC1CCC(CC)CC1") in (0, 1)
    assert decide("CC1C[13CH2]C(CC1)=CC1CCC(CC)CC1") in (0, 1)
    assert decide("[13CH3]C1CCC(CC1)=CC1CCC(CC)CC1") is None
    assert decide("CC1CCC(CC1)=CC1CCC(C[13CH3])CC1") is None


# --- whole molecules, both tiers -------------------------------------------------------------------------------

#: the letters of the two whole names decide ('ethylc' before 'ethylm', 'bromoch' before 'bromocy',...)
P45_5_ROWS = [
    ("CC1CCC(CC1)=CC1CCC(CC)CC1", "1-[(4-ethylcyclohexyl)methylidene]-4-methylcyclohexane"),
    ("CCC1CCC(CC1)=CC1CCC(C)CC1", "1-[(4-ethylcyclohexylidene)methyl]-4-methylcyclohexane"),
    ("ClC1CCC(CC1)=CC1CCC(Br)CC1", "1-bromo-4-[(4-chlorocyclohexylidene)methyl]cyclohexane"),
    ("CC1CCC(CC1)=CC1CCC(OC)CC1", "1-[(4-methoxycyclohexyl)methylidene]-4-methylcyclohexane"),
    ("C1(=CC2C(CC)CCCC2)C(C)CCCC1", "1-[(2-ethylcyclohexyl)methylidene]-2-methylcyclohexane"),
    ("C1(=CC2C(C)CCCC2)C(CC)CCCC1", "1-[(2-ethylcyclohexylidene)methyl]-2-methylcyclohexane"),
    ("C1(=CC2C(CC)CCC2)C(C)CCC1", "1-[(2-ethylcyclopentyl)methylidene]-2-methylcyclopentane"),
    # a multiplied prefix inside the compound prefix is part of its letters
    ("C1(=CC2C(F)CC(F)CC2)C(Cl)CCC(OC)C1", "1-chloro-2-[(2,4-difluorocyclohexyl)methylidene]-4-methoxycyclohexane"),
    # the ring that carries the principal group: the same criteria on the prefixes
    ("C1(=CC2C(O)C(Br)CCC2)C(O)C(CC)CCC1", "2-bromo-6-[(3-ethyl-2-hydroxycyclohexylidene)methyl]cyclohexan-1-ol"),
]

#: the same locant sets, a lower set in the order of citation
P45_2_3_ROWS = [
    ("C1(=CC2C(Br)C(CC)CCC2)C(C)C(F)CCC1",
     "1-[(2-bromo-3-ethylcyclohexyl)methylidene]-3-fluoro-2-methylcyclohexane"),
    ("C1(=CC2CC(CC)C(CC)CC2)CC(F)C(OC)CC1",
     "1,2-diethyl-4-[(3-fluoro-4-methoxycyclohexylidene)methyl]cyclohexane"),
    ("C1(=CC2C(F)CC(CC)CC2)C(Br)CC(OC)CC1",
     "1-[(2-bromo-4-methoxycyclohexylidene)methyl]-4-ethyl-2-fluorocyclohexane"),
    ("C1(=CC2C(F)C(O)CCC2)C(Cl)C(O)CCC1",
     "2-chloro-3-[(2-fluoro-3-hydroxycyclohexyl)methylidene]cyclohexan-1-ol"),
    ("C1(=CC2C(C)C(O)CCC2)C(O)CCCC(C)1",
     "2-[(3-hydroxy-2-methylcyclohexyl)methylidene]-3-methylcyclohexan-1-ol"),
]


@pytest.mark.parametrize("smiles,pin", P45_5_ROWS + P45_2_3_ROWS)
def test_ring_tie_is_decided_by_the_two_names(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", [r[0] for r in P45_5_ROWS[:3] + P45_2_3_ROWS[:2]])
def test_the_name_does_not_depend_on_the_order_of_the_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    expected = name_default(smiles)["name"]
    for seed in range(4):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        alt = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
        assert name_default(alt)["name"] == expected, alt
        assert name_breadth(alt)["name"] == expected, alt


@pytest.mark.parametrize("smiles", [
    # two cyclohexene rings: the ring numbering of an 'ene' ring is not worked out, so the tie is not decided
    "CC1=CCC(CC1)=CC1CCC(CC)C=C1",
    "CC1=CCC(CC1)=CC1CCC(C)C=C1",
])
def test_a_tie_that_is_not_worked_out_still_declines_at_the_default_tier(smiles):
    assert_declined_at_default(smiles)
    assert name_breadth(smiles)["tier"] != "pin_verified"


# --- the criteria of that come before -------------------------------------------------------------

# (h) (:21305): the ring whose principal groups take the lower locants, under its own numbering, is the
# parent, before,, and ('cyclohexane-1,2-diol' is senior to 'cyclohexane-1,3-diol',
# as 'octane-1,7-diol (PIN)' is senior to 'octane-1,8-diol (PIN)',:21327). The first two rows carry the same
# groups on both rings (the third has two prefixes against one on the ring that takes the lower locants, so
# would choose the other ring); the second has the ring that takes the lower locants on the single-bonded side; the
# fourth has more prefixes on the ring that does not take the lower locants would choose that one).
P44_4_1_8_ROWS = [
    ("OC1C(O)CC(=CC2C(O)CC(O)CC2)CC1", "4-[(2,4-dihydroxycyclohexyl)methylidene]cyclohexane-1,2-diol"),
    ("OC1CC(O)CCC1=CC1CC(O)C(O)CC1", "4-[(2,4-dihydroxycyclohexylidene)methyl]cyclohexane-1,2-diol"),
    ("OC1C(O)CC(=CC2C(O)CCCC2O)CC1", "4-[(2,6-dihydroxycyclohexyl)methylidene]cyclohexane-1,2-diol"),
    ("OC1C(O)CC(=CC2C(O)CCC(C)C2O)CC1", "4-[(2,6-dihydroxy-3-methylcyclohexyl)methylidene]cyclohexane-1,2-diol"),
]

# (k) (:21473): the parent structure (the ring with its principal groups) is chosen by the labels it carries,
# before and:.1 the greater number of isotopically modified atoms (:21477),.2 the greater number of
# nuclides of higher atomic number (:21490, '14C is senior to 2H'),.3 of higher mass number (:21496, '14C is senior to
# 13C', '3H is senior to 2H'),.4 the lowest locants of the modified atoms (:21510),.5 the lower locants of the nuclides of
# higher atomic number (:21516, '[4-13C,5-2H1]pentanoic acid' before '[5-13C,4-2H1]'),.6 the lowest locants of the
# nuclides of higher mass number (:21523, '[4-13C,3-14C]butan-1-ol' before '[3-13C,4-14C]'). The isotope decorator names the
# molecule without its labels, so the labelled molecule is read as well (``locant_omission.skeleton_label_maps``).
# Each row is a pair of rings that are alike but for their labels, so the labels alone decide; the name is that of the
# parent that carries the labels the criterion prefers (the labels of the parent are the descriptor after the prefix).
P44_4_1_11_ROWS = [
    #.1 more modified atoms
    ("CC1CCC(CC1)=CC1CC[13CH](CC)CC1", "1-ethyl-4-[(4-methylcyclohexylidene)methyl](1-13C)cyclohexane"),
    ("[2H]C1([2H])CCC(CC1)=CC1CCC(CC)CC1", "1-[(4-ethylcyclohexyl)methylidene](4,4-2H2)cyclohexane"),
    ("C1CCC(CC1)=C[13CH]1CCCCC1", "1-(cyclohexylidenemethyl)(1-13C)cyclohexane"),
    ("C1(=CC2CCCCC2)CC[13CH2]CC1", "1-(cyclohexylmethylidene)(4-13C)cyclohexane"),
    #.1 comes before.2: two 2H atoms are senior to one 14C (the book's own example,:21484), although 14C outranks 2H
    ("C1(=CC2CC[14CH2]CC2)CCC([2H])([2H])CC1", "1-[(4-14C)cyclohexylmethylidene](4,4-2H2)cyclohexane"),
    #.2 14C before 2H
    ("C1(=CC2CCC([2H])CC2)CC[14CH2]CC1", "1-[(4-2H1)cyclohexylmethylidene](4-14C)cyclohexane"),
    #.3 14C before 13C; 3H before 2H
    ("C1(=CC2CC[13CH2]CC2)CC[14CH2]CC1", "1-[(4-13C)cyclohexylmethylidene](4-14C)cyclohexane"),
    ("C1(=CC2CCC([3H])CC2)CCC([2H])CC1", "1-[(4-2H1)cyclohexylidenemethyl](4-3H1)cyclohexane"),
    #.4 the lower locant, 3 before 4
    ("C1(=CC2CC[13CH2]CC2)C[13CH2]CCC1", "1-[(4-13C)cyclohexylmethylidene](3-13C)cyclohexane"),
    #.4 comes before.5: the locants of all the labels together (2,2,4 before 3,4,4), although the 13C of the other ring has
    # the lower locant (3 before 4)
    ("C1(=CC2C[13CH2]C([2H])([2H])CC2)C([2H])([2H])C[13CH2]CC1",
     "1-[(3-13C,4,4-2H2)cyclohexylmethylidene](4-13C,2,2-2H2)cyclohexane"),
    #.5 13C at 3 and 2H at 4 before 13C at 4 and 2H at 3 (the locants together tie)
    ("C1(=CC2CC([2H])[13CH2]CC2)C[13CH2]C([2H])CC1",
     "1-[(4-13C,3-2H1)cyclohexylmethylidene](3-13C,4-2H1)cyclohexane"),
    #.6 14C at 3 and 13C at 4 before 13C at 3 and 14C at 4 (the locants and the atomic numbers tie)
    ("C1(=CC2C[13CH2][14CH2]CC2)C[14CH2][13CH2]CC1",
     "1-[(3-13C,4-14C)cyclohexylmethylidene](4-13C,3-14C)cyclohexane"),
    # would choose the ring with the methyl: (k).4 comes first (13C at 4 before 13C at 5)
    ("C1(=CC2CC[13CH2]CC2)C(C)CC[13CH2]C1", "1-{[2-methyl(5-13C)cyclohexylidene]methyl}(4-13C)cyclohexane"),
    # the same labels on both rings: every criterion of (k) ties, the sides are alike, decides as without labels
    ("C1(=CC2CC[13CH2]CC2)CC[13CH2]CC1", "1-[(4-13C)cyclohexylidenemethyl](4-13C)cyclohexane"),
    ("C1(=CC2CC[13CH2][14CH2]C2)C[14CH2][13CH2]CC1", "1-[(4-13C,3-14C)cyclohexylidenemethyl](4-13C,3-14C)cyclohexane"),
]


@pytest.mark.xfail(strict=True, reason=(
    "holds for this spelling only. The ring is chosen (the one with 13C, 14C and 2H) but its numbering "
    "between (3-13C,4-14C,5-2H1) and (5-13C,4-14C,3-2H1) is left to the order the atoms were written in "
    "(P-14.4 (i), :3336, gives the lower locants to the nuclides of higher atomic number); every spelling "
    "is now named as RDKit's canonical SMILES is, which takes the second (the base names that spelling "
    "the same)"))
def test_p44_4_1_11_5_comes_before_p44_4_1_11_6():
    """.5 (:21516) before.6 (:21523): the locants together tie (3, 4, 5 in both rings); the carbon nuclides of the ring with 13C at
    3 and 14C at 4 have the lower locants (3, 4 against 3, 5), although the 14C of the other ring has the lower locant (3 against 4).
    The ring with 13C at 3, 14C at 4 and 2H at 5 is the parent (the name is not labelled a PIN; it names that ring)."""
    smiles = "C1(=CC2C[13CH2][14CH2]C([2H])C2)C[14CH2]C([2H])[13CH2]C1"
    row = name_breadth(smiles)
    assert row["name"].endswith("(3-13C,4-14C,5-2H1)cyclohexane"), row
    assert row["tier"] != "pin_verified" and name_is_rt_exact(row["name"], smiles), row


def test_p44_4_1_11_2_comes_before_p44_4_1_11_3():
    """.2 (:21490) before.3 (:21496): the ring with three 13C (atomic numbers 6, 6, 6) is senior to the ring with a 14C and two 2H
    (6, 1, 1), although 14C has the higher mass number at the first place. The name the engine builds is not labelled a PIN
    (``systematic_verified``); the parent it names is the ring with the three 13C, and it reads back to the input."""
    smiles = "C1(=CC2C[13CH2][13CH2][13CH2]C2)CC([2H])([2H])[14CH2]CC1"
    row = name_breadth(smiles)
    assert row["name"].endswith("(3,4,5-13C3)cyclohexane"), row
    assert row["tier"] != "pin_verified" and name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("smiles,pin", P44_4_1_8_ROWS + P44_4_1_11_ROWS)
def test_the_criteria_of_p44_4_1_come_before_p45(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", [
    # a label on an atom of a principal group, on both rings: the locant of the label is not numbered here
    # would compare it), so the tie is not decided
    "[18OH]C1CCC(CC1)=CC1CCC([18OH])CC1",
    # a specified configuration on a ring: criterion (l) of,:21529) is not worked out, so
    # and do not decide the tie
    "CC1CCC(CC1)=C[C@H]1CC[C@@H](CC)CC1",
    "CC1CCC(CC1)=C[C@H]1CC[C@H](CC)CC1",
    "CO[C@@H]1CCC[C@@H](C=C2[C@@H](Cl)CCC[C@@H]2Cl)[C@@H]1C",
    # and (:22172,:22208) stand between and: the label is on a substituent
    "CC1CCC(CC1)=CC1CCC(C[13CH3])CC1",
])
def test_a_criterion_not_worked_out_declines_at_the_default_tier(smiles):
    assert_declined_at_default(smiles)


# The rescue of the wider tiers: when the ring the criteria choose leads to no name (a specified E/Z link that no ring-ylidene
# prefix carries), the other ring is offered as the parent (``spelling_offer.offer_other_tie_ring``): best-effort names the
# molecule as it did before the criteria chose the ring, with the label the tier gives that name (OPSIN reads no ring-ylidene
# stereodescriptor: ``pin_unverified``, ``verified`` ``opsin_constitution``); the default tier still declines it.
OTHER_RING_ROWS = [
    ("C[C@@H]1/C(=C/[C@@H]2C[C@@H](O)[C@@H](O)C2)[C@H](O)C[C@H]1O",
     "(1R,3R,4Z,5R)-4-{[(1s,3R,4S)-3,4-dihydroxycyclopentyl]methylidene}-5-methylcyclopentane-1,3-diol"),
    ("O[C@@H]1C[C@H](O)[C@H](Cl)/C1=C\\[C@H]1C[C@@H](O)[C@@H](O)C1",
     "(1R,3S,4R,5Z)-4-chloro-5-{[(1r,3R,4S)-3,4-dihydroxycyclopentyl]methylidene}cyclopentane-1,3-diol"),
]


@pytest.mark.parametrize("smiles,name", OTHER_RING_ROWS)
def test_the_other_ring_is_offered_when_the_chosen_ring_has_no_name(smiles, name):
    assert name_default(smiles)["tier"] == "abstain"
    row = name_breadth(smiles)
    assert (row["name"], row["tier"], row["is_pin"], row["verified"]) == (
        name, "pin_unverified", False, "opsin_constitution"), row


def test_the_labelled_molecule_is_read_onto_the_skeleton():
    """ compares the isotopically modified atoms of the parents; the isotope decorator names the molecule
    without its labels, so ``skeleton_label_maps`` maps the skeleton onto the labelled molecule it declares: a labelled
    atom is its nuclide (atomic number, mass number), an isotopic hydrogen on an atom one more of atomic number 1
    ('two 2H atoms are senior to one 14C',:21484); a symmetric skeleton fits the labelled molecule more than once."""
    from orthonym.assembly.locant_omission import isotopic_skeleton_scope, skeleton_label_maps
    skeleton = Chem.MolFromSmiles("CCC1CCCCC1")
    assert skeleton_label_maps(skeleton) == [{}]                                   # no labelled molecule declared
    with isotopic_skeleton_scope(Chem.MolFromSmiles("C[13CH2]C1CCCCC1")):
        assert {repr(sorted(m.items())) for m in skeleton_label_maps(skeleton)} == {repr([(1, [(6, 13)])])}
    with isotopic_skeleton_scope(Chem.MolFromSmiles("CCC1CC([2H])([2H])CCC1")):
        maps = skeleton_label_maps(skeleton)
        assert {repr(sorted(m.items())) for m in maps} == {repr([(4, [(1, 2), (1, 2)])]), repr([(6, [(1, 2), (1, 2)])])}
    with isotopic_skeleton_scope(Chem.MolFromSmiles("CCC1CC[14CH2]CC1")):
        assert {repr(sorted(m.items())) for m in skeleton_label_maps(skeleton)} == {repr([(5, [(6, 14)])])}
    assert skeleton_label_maps(skeleton) == [{}]                                   # the scope ends with the naming


def test_the_suffix_locants_of_a_ring_are_the_lowest_set_it_can_give():
    """ (:21305) compares the locants of the suffix under the ring's own numbering, then
    ,:3189: the sets compare term by term in increasing order): the groups of the book's own
    'octane-1,7-diol' and 'octane-1,8-diol' fall on a ring as 1,2 against 1,3 against 1,4."""
    from orthonym.rules.ring_selection import _suffix_locants
    mol = Chem.MolFromSmiles("OC1C(O)CCCC1.OC1CC(O)CCC1.OC1CCC(O)CC1.OC1=CC(O)CC=C1")
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    got = []
    for ring in rings:
        oxygens = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O"
                   and any(n.GetIdx() in ring for n in a.GetNeighbors())]
        carbons = [n.GetIdx() for o in oxygens for n in mol.GetAtomWithIdx(o).GetNeighbors()]
        got.append(_suffix_locants(mol, ring, carbons))
    assert got == [(1, 2), (1, 3), (1, 4), (1, 3)]
