"""The parent of a molecule with two amide groups on different chains is chosen by and (lane/l17 item 7).

 "SENIORITY ORDER FOR CLASSES" (the Blue Book), Table 4.1: the amides are one class. The primary, secondary and
tertiary amide have the one rank, so the number of hydrogens on the nitrogen does not make an amide the principal group.
The parent is the senior of the parent structures of the class: "Systems composed of rings and chains" (:19336)
(1) "Within the same class, a ring or ring system has seniority over a chain"; "The principal chain has the greater
number of skeletal atoms" (:20938); then "CRITERIA RELATED TO NUMBER AND LOCATION OF SUBSTITUENT GROUPS" (:21596),
 "The preferred IUPAC name is based on the senior parent structure that has the maximum number of substituents
cited as prefixes" (:21604; its example (4), 'N,N,2-trimethyl-3-{...}propanamide (PIN)', counts the N-substituents),
 "the lower locant or set of locants for substituents cited as prefixes" (:21698; example (1): 'the locant set
N,2 in the PIN is lower than 2,5'; "Lowest set of locants" (:3189): "Italic capital and lower-case letter locants
are lower than Greek letter locants, which, in turn, are lower than numerals").

'CCC(=O)N(C)C(C)C(N)=O': the propanamide of the CONH2 end has one prefix (the amido group), the propanamide of the acyl end
has two (N-methyl and the N-substituent): the second is the parent. It was named with the primary amide as the parent
and the acyl nitrogen as a prefix, which is no PIN ('2-(N-methylpropanamido)propanamide') and was refused.

The criteria run in the order of "If the criteria of through, where applicable, do not effect a choice
of a senior parent structure, the following criteria are applied successively until there are no alternatives remaining"
(:21016) before: (a) the greater number of multiple bonds, (b) the greater number of double bonds, (j) the lower
locants for 'ene' and 'yne' endings, (k) the isotopically modified atoms "The senior ring, ring system, or
principal chain that has one or more isotopically modified atoms",:21473; "The senior parent structure
contains the greater number of isotopically modified atoms or groups",:21477). Criterion (l), "The senior ring,
ring system, or principal chain has one or more stereogenic centers" (:21529), is read as its examples show it ((Z)- over
(E)-cyclooctene): between parents that differ in configuration only, so it ranks no two chains of other constitutions.
"""
import random

import pytest
from rdkit import Chem

from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.assembly.n_acyl_amido import parent_amide_is_senior
from orthonym.rules.amide_parent import amide_parent_key, parent_amide_is_junior, senior_carboxamide_group
from orthonym.rules.seniority import get_principal_group
from tests.support.rt_assert import assert_full_rt
from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_TIERS = ["pin", "best-effort"]


# (smiles, the PIN, the criterion that makes the named amide the parent)
_TARGETS = [
    #: two prefixes (N-methyl, the N-substituent) against one (the amido group), the acyl chains of one length
    ("CCC(=O)N(C)C(C)C(N)=O", "N-(1-amino-1-oxopropan-2-yl)-N-methylpropanamide", "P-45.2.1"),
    #: the butanamide chain is longer than the propanamide chain of the CONH2 end
    ("CCCC(=O)N(C)C(C)C(N)=O", "N-(1-amino-1-oxopropan-2-yl)-N-methylbutanamide", "P-44.3.2"),
    # with the secondary amide of the book's own example (4) pattern: three prefixes against two
    ("CC(C)C(=O)N(C)C(C)C(=O)NC", "N,2-dimethyl-N-[1-(methylamino)-1-oxopropan-2-yl]propanamide", "P-45.2.1"),
    # (1): the benzamide is a ring amide, senior to the chain amide with the CONH2
    ("NC(=O)C(C)N(C)C(=O)c1ccccc1", "N-(1-amino-1-oxopropan-2-yl)-N-methylbenzamide", "P-44.1.2.2"),
    ("NC(=O)CNC(=O)c1ccccc1", "N-(2-amino-2-oxoethyl)benzamide", "P-44.1.2.2"),
    #: one prefix each, so the locants decide: N is lower than 2
    ("CCC(=O)NC(C)C(N)=O", "N-(1-amino-1-oxopropan-2-yl)propanamide", "P-45.2.2"),
    ("CC(=O)NCC(N)=O", "N-(2-amino-2-oxoethyl)acetamide", "P-45.2.2"),
    # among three amides (two secondary, one primary): the dodecanamide chain is the longest, its two junior
    # amides are cited inside the N-substituent, not as a second suffix
    ("CCCCCCCCCCCC(=O)NC(C)C(=O)NC(C)C(N)=O",
     "N-{1-[(1-amino-1-oxopropan-2-yl)amino]-1-oxopropan-2-yl}dodecanamide", "P-44.3.2"),
    ("CCCCCCCCCCCCCC(=O)N[C@@H](CCCC)C(=O)N[C@@H](CCCC)C(N)=O",
     "N-[(2S)-1-{[(2S)-1-amino-1-oxohexan-2-yl]amino}-1-oxohexan-2-yl]tetradecanamide", "P-44.3.2"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name,rule", _TARGETS)
def test_the_senior_amide_is_the_parent(smiles, name, rule, tier):
    """The named amide is the parent by the criterion of the row; the name is a PIN read back whole by OPSIN."""
    row = tier_row(smiles, tier)
    assert row["name"] == name, (rule, row)
    assert row["tier"] == "pin_verified", (rule, row)
    assert_full_rt(row["name"], smiles)


# the group of the other amide stays a prefix where its parent is the senior one
_CONTROLS = [
    # the secondary amide with the CONH2-free chain holds two prefixes (N-methyl and the amido group) against one
    ("CCC(=O)NC(C)C(=O)NC", "N-methyl-2-propanamidopropanamide"),
    #: the propanamide chain of the CONH2 end is longer than the acetyl chain
    ("ClCC(=O)N(C)C(C)C(N)=O", "2-(2-chloro-N-methylacetamido)propanamide"),
    ("CC(=O)N(C)C(C)C(N)=O", "2-(N-methylacetamido)propanamide"),
    # a ring primary amide is the parent of a chain amide
    ("NC(=O)c1ccc(N(C)C(=O)C)cc1", "4-(N-methylacetamido)benzamide"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _CONTROLS)
def test_the_group_of_the_junior_amide_stays_a_prefix(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row
    assert_full_rt(row["name"], smiles)


# a group of the N-substituent is cited once, inside the name of that substituent: the amide parent adds no second prefix
# for it (the 'amino' of '4-aminobutyl' was cited again as an 'amino' of the parent, a name no gate accepts)
_N_SUBSTITUENT_GROUPS = [
    ("CCCCCCCCCCCC(=O)NCCCCN", "N-(4-aminobutyl)dodecanamide"),
    ("CC(=O)NCC(N)=O", "N-(2-amino-2-oxoethyl)acetamide"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _N_SUBSTITUENT_GROUPS)
def test_a_group_of_the_n_substituent_is_cited_once(smiles, name, tier):
    """ (the Blue Book): R-CO-NHR' is named by citing R' as a prefix preceded by the locant N: the
    groups R' holds are part of its name."""
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", ["CCC(=O)N(C)C(C)C(N)=O", "CCCC(=O)N(C)C(C)C(N)=O", "NC(=O)CNC(=O)c1ccccc1"])
def test_the_pin_tier_does_not_name_the_amido_prefix_form(smiles):
    """The prefix form is the name of the junior parent: no tier ships it as a PIN."""
    junior = {"CCC(=O)N(C)C(C)C(N)=O": "2-(N-methylpropanamido)propanamide",
              "CCCC(=O)N(C)C(C)C(N)=O": "2-(N-methylbutanamido)propanamide",
              "NC(=O)CNC(=O)c1ccccc1": "2-benzamidoacetamide"}[smiles]
    for tier in _TIERS:
        assert tier_row(smiles, tier)["name"] != junior


# --- the choice --------------------------------------------------------------------------------------------------------

def _principal(smiles):
    mol = Chem.MolFromSmiles(smiles)
    group, matches = get_principal_group(mol, detect_functional_groups(mol))
    return mol, group, matches


@pytest.mark.parametrize("smiles,group", [
    ("CCC(=O)N(C)C(C)C(N)=O", "tertiary_amide"),             #
    ("CCCC(=O)N(C)C(C)C(N)=O", "tertiary_amide"),            #
    ("NC(=O)CNC(=O)c1ccccc1", "secondary_amide"),            # (1)
    ("CCC(=O)NC(C)C(N)=O", "secondary_amide"),               #, N lower than 2
    ("CCCCCCCCCCCC(=O)NC(C)C(=O)NC(C)C(N)=O", "secondary_amide"),   # among three: the dodecanamide
    ("CC(=O)N(C)C(C)C(N)=O", "primary_amide"),               #: the propanamide chain is longer
    ("NC(=O)c1ccc(N(C)C(=O)C)cc1", "primary_amide"),         # the ring amide is the parent
    ("CCC(=O)NC(C)C(=O)NC", "secondary_amide"),              # one subtype only: nothing to choose between
])
def test_the_principal_group_is_the_amide_of_the_senior_parent(smiles, group):
    assert _principal(smiles)[1] == group


def test_the_principal_group_holds_the_match_of_the_senior_amide_only():
    """Two secondary amides on chains of their own: the suffix is the one of the senior chain, the other is cited in the
    name of the N-substituent that holds it (one match, so the single-amide namer takes it)."""
    _, group, matches = _principal("CCCCCCCCCCCC(=O)NC(C)C(=O)NC(C)C(N)=O")
    assert group == "secondary_amide" and len(matches) == 1
    assert matches[0][0] == 11, matches          # the carbonyl carbon of the dodecanoyl group (atoms 0-10 are its chain)


@pytest.mark.parametrize("smiles", [
    "NC(=O)CCC(=O)N(C)C",        # the two amide groups lie on one chain: one diamide, the suffix counts them together
    "NC(=O)CC(C)C(=O)NC",
])
def test_amides_on_one_chain_keep_the_head_of_the_seniority_order(smiles):
    """A diamide is not a choice between two parents: ``senior_carboxamide_group`` returns the head unchanged."""
    mol = Chem.MolFromSmiles(smiles)
    fg = detect_functional_groups(mol)
    head = ("primary_amide", fg["primary_amide"])
    assert senior_carboxamide_group(mol, fg, head) == head


def test_the_key_orders_the_criteria():
    """Ring, then the longer chain, then the multiple bonds, then the number of prefixes, then the lower locants."""
    def key(smiles, carbonyl_pattern="[CX3](=O)[NX3]"):
        mol = Chem.MolFromSmiles(smiles)
        c = mol.GetSubstructMatch(Chem.MolFromSmarts(carbonyl_pattern))[0]
        return amide_parent_key(mol, c)

    assert key("NC(=O)c1ccccc1") < key("NC(=O)CCCCCC")                   # (1): a ring over a chain
    assert key("NC(=O)CCC") < key("NC(=O)CC")                            #: the greater number of skeletal atoms
    assert key("NC(=O)C=C") < key("NC(=O)CC")                            #: the greater number of multiple bonds
    assert key("CN(C)C(=O)CC") < key("NC(=O)CC")                         #: three prefixes against none
    assert key("CNC(=O)CC") < key("NC(=O)CC")                            #: N-methyl is a prefix
    assert key("CC(C)C(=O)NC") < key("CCC(=O)NC")                        #: 2-methyl is a prefix
    assert key("NC(=O)CC(C)C") < key("NC(=O)CCC")                        # the branch counts: 3-methylbutanamide
    assert key("CNC(=O)CC") < key("NC(=O)C(C)C")                         #: N-methyl (N) is lower than 2-methyl (2)


def test_the_key_is_deterministic_under_a_renumbering():
    smiles = "CCC(=O)N(C)C(C)C(N)=O"
    keys = set()
    for seed in range(5):
        mol = Chem.MolFromSmiles(smiles)
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        renum = Chem.RenumberAtoms(mol, order)
        fg = detect_functional_groups(renum)
        head = get_principal_group(renum, fg)
        keys.add(head[0])
    assert keys == {"tertiary_amide"}


# --- the criteria of between the multiple-bond count and (fix round of lane/l17 item 7) -------------------

_P44_4_1_TARGETS = [
    # (j) "the lower locant(s) for endings or prefixes that express changes in the level of hydrogenation, i.e., for 'ene'
    # and 'yne' endings" (j),:21029): the ene at 2 of the primary amide's chain outranks the ene at 3 of the
    # acyl chain, though the acyl chain holds more prefixes is the later criterion)
    ("C=CCC(=O)N(C)CC=CC(N)=O", "4-(N-methylbut-3-enamido)but-2-enamide", "P-44.4.1 (j)"),
    # (b) "the greater number of double bonds" (b),:21021): the but-2-enamide chain, one multiple bond each
    ("CC#CC(=O)N(C)CC=CC(N)=O", "4-(N-methylbut-2-ynamido)but-2-enamide", "P-44.4.1 (b)"),
    # (k) "has one or more isotopically modified atoms": the acetamide chain with the two 2H, not the
    # N-methylacetamide chain that would take
    ("CC(=O)N(C)C([2H])([2H])C(N)=O", "2-(N-methylacetamido)(2,2-2H2)acetamide", "P-44.4.1 (k)"),
    # (k) in the other direction: the 13C of the acyl chain makes it the parent, as it did by the prefix count alone
    ("[13CH3]CC(=O)N(C)C(C)C(N)=O", "N-(1-amino-1-oxopropan-2-yl)-N-methyl(3-13C)propanamide", "P-44.4.1 (k)"),
    # (k): the 13C of the primary amide's carbonyl makes that chain the parent that the prefix count would not choose
    ("CC(=O)N(C)C(C)[13C](N)=O", "2-(N-methylacetamido)(1-13C)propanamide", "P-44.4.1 (k)"),
    # stays before the isotopes: the isobutanamide chain is the longer, the 2H of the shorter one do not matter
    ("CC(C)C(=O)N(C)C([2H])([2H])C(=O)N", "N-[2-amino-2-oxo(2H2)ethyl]-N,2-dimethylpropanamide", "P-44.3.2"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name,rule", _P44_4_1_TARGETS)
def test_the_p44_4_1_criteria_choose_the_parent_before_the_prefix_count(smiles, name, rule, tier):
    """The parent is the chain the criterion of the row prefers; the PIN is read back whole by OPSIN. Before the criteria of
     were applied the name cited the senior amide as the prefix of the junior one, or put the N-substituted
    nitrogen's chain first and shipped 'N-(4-amino-4-oxobut-2-en-1-yl)-N-methylbut-3-enamide' as a PIN."""
    row = tier_row(smiles, tier)
    assert row["name"] == name, (rule, row)
    assert row["tier"] == "pin_verified", (rule, row)
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,junior", [
    ("C=CCC(=O)N(C)CC=CC(N)=O", "N-(4-amino-4-oxobut-2-en-1-yl)-N-methylbut-3-enamide"),
    ("CC(=O)N(C)C([2H])([2H])C(N)=O", "N-[2-amino-2-oxo(2H2)ethyl]-N-methylacetamide"),
])
def test_no_tier_ships_the_junior_parent_as_a_pin(smiles, junior):
    for tier in _TIERS:
        row = tier_row(smiles, tier)
        assert not (row["name"] == junior and row["tier"] == "pin_verified"), row


def test_a_triple_bond_chain_is_junior_to_a_double_bond_chain_of_one_multiple_bond():
    """ (a) ties (one multiple bond each), (b) "the greater number of double bonds" decides."""
    mol = Chem.MolFromSmiles("CC#CC(=O)N(C)CC=CC(N)=O")
    yne, ene = [m[0] for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]"))]
    assert amide_parent_key(mol, ene) < amide_parent_key(mol, yne)


def test_the_ene_locant_decides_before_the_prefix_count():
    """ (j) before: the primary amide's chain (ene at 2, one prefix) is senior to the acyl chain (ene at 3,
    two prefixes)."""
    mol = Chem.MolFromSmiles("C=CCC(=O)N(C)CC=CC(N)=O")
    keys = {m[0]: amide_parent_key(mol, m[0]) for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]"))}
    primary = next(c for c in keys if mol.GetAtomWithIdx(c).GetNeighbors()[2].GetDegree() == 1
                   or any(nb.GetAtomicNum() == 7 and nb.GetDegree() == 1 for nb in mol.GetAtomWithIdx(c).GetNeighbors()))
    assert keys[primary] == min(keys.values()) and len(set(keys.values())) == 2


def test_isotopes_are_read_from_the_labelled_molecule_of_the_skeleton():
    """The decorator names the isotope-stripped skeleton: the labels are read from the labelled molecule it declares
    (``assembly.locant_omission.isotopic_skeleton_scope``), so criterion (k) sees them."""
    from orthonym.assembly.locant_omission import isotopic_skeleton_scope
    from orthonym.rules.isotopes import strip_isotopes
    original = Chem.MolFromSmiles("CC(=O)N(C)C([2H])([2H])C(N)=O")
    stripped, _ = strip_isotopes(original)
    fg = detect_functional_groups(stripped)
    head = get_principal_group(stripped, fg)
    assert head[0] == "tertiary_amide"                   # no label seen: the prefix count
    with isotopic_skeleton_scope(original):
        group, matches = get_principal_group(stripped, detect_functional_groups(stripped))
    assert group == "primary_amide"                      #: the chain with the two 2H


# --- a group is no prefix of a parent that is junior to the amide of the group (n_acyl_amido) ------------------------------

def _acyl_group(mol, n_pattern="[NX3;H0]([CX4])([CX4])C(=O)[CX4]"):
    match = mol.GetSubstructMatch(Chem.MolFromSmarts(n_pattern))
    n, acyl_c = match[0], match[3]
    return acyl_c, {n, match[1], match[2], acyl_c, *(nb.GetIdx() for nb in mol.GetAtomWithIdx(acyl_c).GetNeighbors())}


def test_the_group_that_holds_the_principal_amide_is_no_prefix():
    """The principal amide of 'CCC(=O)N(C)C(C)C(N)=O' is the tertiary one: its own amido group is not cited as a prefix
    of a parent outside it (it returned True: 'lies inside the group itself')."""
    mol = Chem.MolFromSmiles("CCC(=O)N(C)C(C)C(N)=O")
    acyl_c, group = _acyl_group(mol)
    assert parent_amide_is_senior(mol, acyl_c, group) is False


def test_the_group_of_the_junior_amide_is_a_prefix_of_the_senior_parent():
    mol = Chem.MolFromSmiles("CC(=O)N(C)C(C)C(N)=O")      # the propanamide chain of the CONH2 end is longer
    acyl_c, group = _acyl_group(mol)
    assert parent_amide_is_senior(mol, acyl_c, group) is True


def test_a_group_is_compared_with_every_amide_that_can_be_a_parent():
    """Three amides: the amido group of the junior acetamide is a prefix of a parent outside it even when the principal
    group of the molecule is a third amide."""
    mol = Chem.MolFromSmiles("CC(=O)N(C)CC(=O)NC(C)C(=O)NC")
    acyl_c, group = _acyl_group(mol)
    assert parent_amide_is_senior(mol, acyl_c, group) is True


def test_a_parent_amide_junior_to_another_is_found_by_the_choice_between_parents():
    """The acetamide chain (two skeletal atoms) is junior to both propanamide chains (three):."""
    mol = Chem.MolFromSmiles("CCC(=O)N(C)C(C)C(=O)NCC(=O)NC1CC1")
    keys = {m[0]: amide_parent_key(mol, m[0]) for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]"))}
    assert len(keys) == 3
    junior = max(keys, key=keys.get)
    assert any(nb.GetAtomicNum() == 6 and nb.GetTotalNumHs() == 2 for nb in mol.GetAtomWithIdx(junior).GetNeighbors())
    assert parent_amide_is_junior(mol, {junior}) is True
    assert parent_amide_is_junior(mol, {min(keys, key=keys.get)}) is False


# --- three amides: the parent is the senior amide at every tier, no tier ships the junior one as a PIN --------------------

_THREE_AMIDES = [
    "CCC(=O)N(C)C(C)C(=O)NCC(=O)NC1CC1",
    "CCC(=O)N(C)C(C)C(=O)NCC(=O)Nc1ccccc1",
]


@pytest.mark.parametrize("smiles", _THREE_AMIDES)
def test_three_amides_are_never_named_on_the_junior_acetamide_as_a_pin(smiles):
    """The propanamide chains (three skeletal atoms) outrank the acetamide chain (two):. The decomposition cut the
    middle amide and named the acetamide with the other two cited in its prefixes, 'N-cyclopropyl-2-[2-(N-methylpropanamido)
    propanamido]acetamide', a pin_verified name of a junior parent. The PIN tier declines; the best-effort tier names the
    senior propanamide (with the tier it earns) and never the acetamide as a PIN."""
    for tier in _TIERS:
        row = tier_row(smiles, tier)
        assert not (row["tier"] == "pin_verified" and (row["name"] or "").endswith("acetamide")), row
        if row["tier"] != "abstain":
            assert_full_rt(row["name"], smiles)
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    best = tier_row(smiles, "best-effort")
    assert best["name"] and best["name"].endswith("-N-methylpropanamide"), best
    assert best["name"].startswith("N-"), best


@pytest.mark.parametrize("smiles,name", [
    # the three amides of a peptide-like chain whose senior amide is the acyl end
    ("CCC(=O)N(C)C(C)C(=O)NCC(=O)NC", "N-methyl-N-(1-{[2-(methylamino)-2-oxoethyl]amino}-1-oxopropan-2-yl)propanamide"),
    ("CCC(=O)N(C)C(C)C(=O)NCC(N)=O", "N-{1-[(2-amino-2-oxoethyl)amino]-1-oxopropan-2-yl}-N-methylpropanamide"),
])
def test_three_amides_name_the_senior_parent_as_a_pin(smiles, name):
    for tier in _TIERS:
        row = tier_row(smiles, tier)
        assert row["name"] == name and row["tier"] == "pin_verified", row
        assert_full_rt(row["name"], smiles)


# --- the N-substituent carbons are no atoms of the parent for the licence (item 1's fix, kept on this branch) ---

def test_the_bracketed_n_substituent_row_is_spelled_with_its_locants():
    """'2-(4-chlorophenoxy)-N-(3-methylbutyl)-N-{...}acetamide': the amide is the parent, its 2-substituent keeps its locant,
    each N-substituent is cited with N and enclosed alone. It was
    '(4-chlorophenoxy)[N-(3-methylbutyl)](N-{...})acetamide', a name no tier labels a PIN."""
    smiles = "CC1=CN=C(S1)NC(=O)CN(CCC(C)C)C(=O)COC2=CC=C(C=C2)Cl"
    row = tier_row(smiles, "best-effort")
    assert row["name"] == ("2-(4-chlorophenoxy)-N-(3-methylbutyl)-N-{2-[(5-methyl-1,3-thiazol-2-yl)amino]-2-oxoethyl}"
                           "acetamide"), row
    assert row["tier"] != "pin_verified", row
    assert_full_rt(row["name"], smiles)


# --- two ring carboxamides: the senior ring system is the parent (1), then -------------------------------

@pytest.mark.parametrize("tier", _TIERS)
def test_of_two_ring_carboxamides_the_senior_ring_system_is_the_parent(tier):
    """ "SENIORITY ORDER ONLY FOR RINGS AND RING SYSTEMS" (:19402), (a) heterocycles are senior to
    carbocycles: the pyridine ring is senior to the benzene ring, so the pyridine-4-carboxamide is the parent. The two
    ring carboxamides tied under the key and the primary amide won ('4-(pyridine-4-carboxamido)benzamide')."""
    smiles = "NC(=O)c1ccc(NC(=O)c2ccncc2)cc1"
    row = tier_row(smiles, tier)
    assert row["name"] == "N-(4-carbamoylphenyl)pyridine-4-carboxamide", row
    assert row["tier"] == "pin_verified", row
    assert_full_rt(row["name"], smiles)


# --- two ring carboxamides of one ring system: (the number of prefixes, then the lower locants, N lowest) -------------

def _keys(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {m[0]: amide_parent_key(mol, m[0]) for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]"))}


# (smiles, the PIN, the criterion that makes the named ring carboxamide the parent)
_TIED_RING_TARGETS = [
    # example (1): 'the locant set N,2 in the PIN is lower than 2,5': N is lower than a numeral
    ("c1ccc(cc1)C(=O)Nc1ccc(cc1)C(N)=O", "N-(4-carbamoylphenyl)benzamide", "P-45.2.2: N before 4"),
    ("NC(=O)c1ccncc1NC(=O)c1ccncc1", "N-(4-carbamoylpyridin-3-yl)pyridine-4-carboxamide", "P-45.2.2: N before 3"),
    ("NC(=O)c1ccccc1NC(=O)c1ccccc1", "N-(2-carbamoylphenyl)benzamide", "P-45.2.2: N before 2"),
    #: the maximum number of prefixes: N-aryl and 4-methyl against the one 4-amido prefix
    ("Cc1ccc(cc1)C(=O)Nc1ccc(cc1)C(N)=O", "N-(4-carbamoylphenyl)-4-methylbenzamide", "P-45.2.1: two against one"),
    ("NC(=O)c1ccc(N(C)C(=O)c2ccccc2)cc1", "N-(4-carbamoylphenyl)-N-methylbenzamide", "P-45.2.1: two against one"),
    ("O=C(Nc1ccc(cc1)C(=O)NC)c1ccccc1", "4-benzamido-N-methylbenzamide", "P-45.2.1: N-methyl and 4-amido against N-aryl"),
    # with the N terms equal: the numerals of the numbering of the ring decide, {N,3} against {N,4}
    ("Cc1cccc(c1)C(=O)Nc1ccc(cc1)C(=O)NC", "3-methyl-N-[4-(methylcarbamoyl)phenyl]benzamide", "P-45.2.2: N,3 before N,4"),
    ("Cc1ccc(cc1)C(=O)Nc1cccc(c1)C(=O)NC", "N-methyl-3-(4-methylbenzamido)benzamide", "P-45.2.2: N,3 before N,4"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name,rule", _TIED_RING_TARGETS)
def test_of_two_ring_carboxamides_of_one_ring_system_p45_2_chooses_the_parent(smiles, name, rule, tier):
    """ "CRITERIA RELATED TO NUMBER AND LOCATION OF SUBSTITUENT GROUPS" (:21596): the two benzamides (two pyridine-4-
    carboxamides) tie under, so the criteria of and decide. The tie kept the head of
    SENIORITY_ORDER (the primary amide) and shipped '4-benzamidobenzamide' as the PIN of the first row."""
    row = tier_row(smiles, tier)
    assert row["name"] == name, (rule, row)
    assert row["tier"] == "pin_verified", (rule, row)
    assert_full_rt(row["name"], smiles)


def test_the_key_of_two_tied_ring_carboxamides_counts_the_prefixes_and_ranks_n_before_a_numeral():
    def key_of(mol, smarts):
        return amide_parent_key(mol, mol.GetSubstructMatch(Chem.MolFromSmarts(smarts))[0])

    mol = Chem.MolFromSmiles("c1ccc(cc1)C(=O)Nc1ccc(cc1)C(N)=O")
    secondary, primary = key_of(mol, "[CX3](=O)[NX3;H1]"), key_of(mol, "[CX3](=O)[NX3;H2]")
    assert secondary[:3] == primary[:3]                          # the groups and the ring system tie: decides
    assert secondary < primary                                   # the locant N (N-aryl) is lower than 4 (4-benzamido)
    # N-methyl and 4-benzamido (two prefixes) are senior to the one N-aryl prefix
    mol = Chem.MolFromSmiles("O=C(Nc1ccc(cc1)C(=O)NC)c1ccccc1")
    assert key_of(mol, "c[CX3](=O)[NX3;H1]c")[3] == -1 and key_of(mol, "c[CX3](=O)[NX3;H1]C")[3] == -2
    assert key_of(mol, "c[CX3](=O)[NX3;H1]C") < key_of(mol, "c[CX3](=O)[NX3;H1]c")


# ---: the parent with the maximum number of carboxamide groups, before the seniority of the rings ------------------

# (smiles, the PIN)
_GROUP_COUNT_TARGETS = [
    ("NC(=O)c1cc(C(N)=O)cc(c1)NC(=O)c1ccncc1", "5-(pyridine-4-carboxamido)benzene-1,3-dicarboxamide"),
    ("NC(=O)c1cc(C(N)=O)cc(c1)NC(=O)C1CCCN1", "5-(pyrrolidine-2-carboxamido)benzene-1,3-dicarboxamide"),
    ("NC(=O)c1ccc(C(N)=O)c(c1)NC(=O)c1cccs1", "2-(thiophene-2-carboxamido)benzene-1,4-dicarboxamide"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _GROUP_COUNT_TARGETS)
def test_the_parent_with_more_carboxamide_groups_outranks_a_senior_ring(smiles, name, tier):
    """ "The senior parent structure has the maximum number of substituents corresponding to the principal
    characteristic group (suffix)" (:18875), applied "before those applicable to rings and ring systems (see ": the
    benzene with two carboxamides is the parent although the pyridine, the pyrrolidine or the thiophene is the senior ring
     (a)). The pyridine-4-carboxamide was the parent and its N-substituent, the benzene with two groups, was cited
    as 'N-(3,5-dicarbamoylphenyl)': no PIN and a name the PIN tier declined."""
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,junior", [
    ("NC(=O)c1ccncc1NC(=O)c1cc(C(N)=O)ccc1", "3-(3-carbamoylbenzamido)pyridine-4-carboxamide"),
    ("O=C(Nc1ccncc1C(N)=O)c1cccc(c1)C(N)=O", "4-(3-carbamoylbenzamido)pyridine-3-carboxamide"),
])
@pytest.mark.parametrize("tier", _TIERS)
def test_a_pyridine_carboxamide_is_never_the_pin_parent_of_a_benzene_dicarboxamide(smiles, junior, tier):
    """: the benzene-1,3-dicarboxamide is the parent (the PIN is 'N1-(4-carbamoylpyridin-3-yl)benzene-1,3-
    dicarboxamide',, which the ring diamide producers do not build). No tier names the molecule on the pyridine
    as a PIN; the best-effort tier names it, read back whole, below the PIN."""
    row = tier_row(smiles, tier)
    assert row["name"] != junior or row["tier"] != "pin_verified", row
    if tier == "pin":
        assert row["tier"] != "pin_verified", row
    else:
        assert row["name"] and row["tier"] != "pin_verified", row
        assert_full_rt(row["name"], smiles)


def test_the_key_counts_the_carboxamide_groups_of_the_ring_system_first():
    mol = Chem.MolFromSmiles("NC(=O)c1cc(C(N)=O)cc(c1)NC(=O)c1ccncc1")
    keys = {c: amide_parent_key(mol, c) for c in (m[0] for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]")))}
    benzene = [k for k in keys.values() if k[0] == -2]
    pyridine = [k for k in keys.values() if k[0] == -1]
    assert len(benzene) == 2 and len(pyridine) == 1, keys
    assert max(benzene) < pyridine[0]                            # before: two groups outrank the senior ring
    # the pyridine is the senior ring system: without the count it would win (the order of the two keys is reversed)
    assert pyridine[0][2] < benzene[0][2]


# --- a fused ring carboxamide and an amide of a junior ring: the junior parent is no PIN, the molecule keeps a name ------------

@pytest.mark.parametrize("smiles,junior", [
    ("NC(=O)c1ccc(cc1)NC(=O)c1ccc2ccccc2n1", "4-(quinoline-2-carboxamido)benzamide"),
    ("NC(=O)c1ccc(cc1)NC(=O)c1ccc2[nH]ccc2c1", "4-(1H-indole-5-carboxamido)benzamide"),
])
@pytest.mark.parametrize("tier", _TIERS)
def test_the_amide_of_a_fused_ring_outranks_the_benzamide_and_is_never_cited_as_a_prefix_in_a_pin(smiles, junior, tier):
    """ (a) the heterocycle is senior to the carbocycle, so the quinoline-2-carboxamide (the indole-5-carboxamide)
    is the parent: the PIN is 'N-(4-carbamoylphenyl)quinoline-2-carboxamide'. The strict path builds no N-substituted
    carboxamide of a fused parent (the carbamoyl prefix: 'N-methylquinoline-2-carboxamide' is no name of it either), so the PIN
    tier declines; the best-effort tier keeps a name read back whole, labelled below the PIN, and it is not the
    junior-parent name that was shipped as a PIN."""
    row = tier_row(smiles, tier)
    assert row["tier"] != "pin_verified", row
    if tier == "best-effort":
        assert row["name"] and row["name"] != junior, row
        assert row["tier"] in ("systematic_verified", "best_effort", "pin_unverified"), row
        assert_full_rt(row["name"], smiles)
