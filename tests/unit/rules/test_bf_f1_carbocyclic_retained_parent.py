"""DT4 F1 task 3: a retained polycyclic hydrocarbon parent with monocyclic hydrocarbon attached components
(``bridged_fused_pin.fusion_names.retained_parent_name``, the carbocyclic half of ``poly_fusion``).

Governing text (the Blue Book): (:11903) ring systems "that have no accepted retained or systematic
name... are named by prefixing to the name of a component ring or ring system (the parent component)
designations of the other component(s) (attached components)"; (:12000) 'cyclopenta', 'cyclohepta' for
monocyclic hydrocarbons other than benzene; (b) (:12163) the parent is "a component containing the
greater number of rings", then (c) (:12234) the larger ring, (g) (:12317) "the greatest number of rings in a
horizontal row" and (j) (:12418) the lower fusion carbon locants; (:12909) identical attached
components take 'di' with the letters separated by commas ('dibenzo[c,g]phenanthrene',:12982, "not
naphtho[2,1-c]phenanthrene; two attached components preferred to one"; the book marks no '(PIN)' there);
 (:13970) the numbers are omitted for 'benzo' and the prefixes, the
letters are kept ('benzo[a]tetracene (PIN)',:13972); (:23710) fusion names for at least two rings of
five or more members.

NUMBERING. The book prints no numbering for these skeletons, and OPSIN 2.9.0 numbers the name (the engine's
source for every derived parent), so each numbering is pinned below by a hand derivation that uses neither OPSIN
nor the engine. (:12501) "The numbering of peripheral atoms in the preferred orientation starts from
the uppermost ring... from the nonfused atom most counterclockwise in the ring selected and proceeds in a
clockwise direction around the system, including fusion heteroatoms but not fusion carbon atoms. Each fusion
carbon atom is given the same number as the immediately preceding nonfusion skeletal atom, modified by a Roman
letter". The orientation is (:12080) (a) the maximum number of rings in a horizontal row, (b) the
maximum number of rings in the upper right quadrant, (c) the minimum in the lower left quadrant, (d) the maximum
above the row, the five-membered ring drawn as a distorted hexagon:12060); then (c)
(:12580) low locants to the fusion carbon atoms. The derivation was done on a hexagonal lattice by a scratch
program that reproduces the book's tetracene (:12513), chrysene and fluorene numberings and naphthalene's; the
result for each skeleton is the periphery below (the order of the atoms round the ring system) and its interior
fusion bonds, which are what ``test_the_numbering_is_the_hand_derived_one`` compares with the engine's.

- cyclopenta[a]chrysene: chrysene's rings A-B-C-D zigzag, the cyclopentane fused at side a (C1-C2) of the end ring A.
  Two rings in the horizontal row is the most the skeleton allows (16 of 36 orientations); (b) leaves 2 orientations
  (3 1/2 rings in the upper right quadrant), which give one numbering; fusion locants 3a, 5a, 5b, 7a, 11a, 11b,
  13a, 13b.
- cyclopenta[a]fluorene: fluorene's six-five-six row of three rings (the most, 4 of 24 orientations), the
  cyclopentane fused at side a (C1-C2); (b) leaves one orientation (1 3/4 rings in the upper right quadrant);
  fusion locants 3a, 5a, 5b, 9a, 10a, 10b.
- dicyclopenta[a,i]phenanthrene: phenanthrene's three rings with a cyclopentane at side a (C1-C2) and one at side i
  (C7-C8), the skeleton's own C2 symmetry; two rings in the row, (b) leaves 2 orientations that give one numbering;
  fusion locants 3a, 5a, 5b, 7a, 10a, 10b, 12a, 12b.

Each name was read back by OPSIN 2.9.0 to the skeleton (the engine's own check in ``parents._parent_for_key``), and
OPSIN reads the other letters ('[b]', '[c]', '[a,j]') to other skeletons, which is what makes the letters the ones
of these skeletons; the rows below were also read back to the input's full standard InChIKey outside the engine
(``name_is_rt_exact``).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused_pin import fusion_names, parents
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.unit

#: name -> (the ring skeleton as an element-aware single-bond graph, the periphery in order from locant 1,
#: the interior fusion bonds), the numbering of the hand derivation of the module docstring
SKELETONS = {
    "cyclopenta[a]chrysene": (
        "C1CCC2C(C1)CCC1C2CCC2C3CCCC3CCC21",
        "1 2 3 3a 4 5 5a 5b 6 7 7a 8 9 10 11 11a 11b 12 13 13a 13b",
        [("11a", "7a"), ("11b", "5b"), ("13a", "5a"), ("13b", "3a")]),
    "cyclopenta[a]fluorene": (
        "C1CCC2C(C1)CC1C3CCCC3CCC21",
        "1 2 3 3a 4 5 5a 5b 6 7 8 9 9a 10 10a 10b",
        [("10a", "5a"), ("10b", "3a"), ("5b", "9a")]),
    "dicyclopenta[a,i]phenanthrene": (
        "C1CC2CCC3C(CCC4C5CCCC5CCC43)C2C1",
        "1 2 3 3a 4 5 5a 5b 6 7 7a 8 9 10 10a 10b 11 12 12a 12b",
        [("10a", "7a"), ("10b", "5b"), ("12a", "5a"), ("12b", "3a")]),
}


def _bond_set(skeleton, numbering):
    return {frozenset((str(numbering[b.GetBeginAtomIdx()]), str(numbering[b.GetEndAtomIdx()])))
            for b in skeleton.GetBonds()}


@pytest.mark.parametrize("name", sorted(SKELETONS))
def test_the_row_skeletons_are_named_by_the_rules(name):
    skeleton = Chem.MolFromSmiles(SKELETONS[name][0])
    assert fusion_names._retained_rule_name(skeleton) == name
    assert fusion_names.retained_parent_name(skeleton) == name


@pytest.mark.parametrize("name", sorted(SKELETONS))
def test_the_numbering_is_the_hand_derived_one(name):
    key, periphery, interior = SKELETONS[name]
    skeleton = Chem.MolFromSmiles(key)
    expected = {frozenset(pair) for pair in zip(periphery.split(), periphery.split()[1:] + ["1"])}
    expected |= {frozenset(pair) for pair in interior}
    got = parents.fused_parent(skeleton, set(range(skeleton.GetNumAtoms())))
    assert got is not None and got.name == name and got.source == "retained_fusion_name+opsin"
    # every numbering the engine offers is this one (the skeleton of dicyclopenta[a,i]phenanthrene has a C2
    # symmetry, so there are two numberings that give one locant graph)
    for numbering in got.numberings:
        assert _bond_set(skeleton, numbering) == expected


@pytest.mark.parametrize("book_name,expected", [
    ("benzo[a]tetracene", "benzo[a]tetracene"),                        #:13972 (PIN)
    ("dibenzo[c,g]phenanthrene", "dibenzo[c,g]phenanthrene"),          #:12982 (a book example, not marked '(PIN)')
    #:12982 "not naphtho[2,1-c]phenanthrene; two attached components preferred to one": the same skeleton
    ("naphtho[2,1-c]phenanthrene", "dibenzo[c,g]phenanthrene"),
])
def test_the_book_names_are_rebuilt_from_their_skeletons(book_name, expected):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smiles = opsin_parse(book_name)
    assert smiles, book_name
    skeleton = Chem.MolFromSmiles(parents._key_of(Chem.MolFromSmiles(smiles)))
    assert fusion_names._retained_rule_name(skeleton) == expected


def test_cyclopenta_a_phenanthrene_comes_from_the_table_with_its_traditional_numbering():
    # (:12493) "Anthracene, phenanthrene,... and cyclopenta[a]phenanthrene are exceptions; traditional
    # numberings are retained": the steroid skeleton is a table parent, never this function's
    skeleton = Chem.MolFromSmiles("C1CCC2C(C1)CCC1C3CCCC3CCC21")
    assert fusion_names.retained_parent_name(skeleton) is None
    got = parents.fused_parent(skeleton, set(range(skeleton.GetNumAtoms())))
    assert got.name == "cyclopenta[a]phenanthrene" and got.source != "retained_fusion_name+opsin"
    assert all(isinstance(loc, int) for loc in got.numberings[0].values())     # 1-17, the steroid numbering


def test_a_polyacene_is_named_by_its_own_series_rule_not_as_benzopentacene():
    # Polyacenes (:11438): hexacene is 'hexacene'; the rule reading (pentacene + benzo) would give
    # 'benzo[b]pentacene' (the table lists pentacene, not hexacene). A member of the series of
    # (``hydrocarbon_series``) is declined here, as a table name is.
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smiles = opsin_parse("hexacene")
    skeleton = Chem.MolFromSmiles(parents._key_of(Chem.MolFromSmiles(smiles)))
    assert fusion_names._retained_rule_name(skeleton) == "benzo[b]pentacene"      # the rule alone
    assert fusion_names.retained_parent_name(skeleton) is None                    # the series guard


def test_a_tetravalent_fusion_atom_is_left_to_the_von_baeyer_name():
    # the angular triquinane (isocomene, silphinene): one atom with four ring bonds in three rings. Its
    # numbering is not derived by here, so no fusion name is built
    skeleton = Chem.MolFromSmiles("C1CC2CCC3CCCC23C1")
    assert fusion_names.retained_parent_name(skeleton) is None
    assert parents._parent_for_key.__wrapped__("C1CC2CCC3CCCC23C1") is None


@pytest.mark.parametrize("smiles", [
    "c1ccc2c(c1)oc1c2oc2ccccc12",          # a heteroring: poly_fusion's heterocyclic half names those
    "c1ccc2c(c1)Cc1ccccc1-2",              # fluorene itself: one component, a table name
    "C1=CC=C2C=CC=CC=C2C=C1",              # benzo[7]annulene: two rings, fusion_names' class
])
def test_outside_the_class_is_declined(smiles):
    skeleton = Chem.MolFromSmiles(parents._key_of(Chem.MolFromSmiles(smiles)))
    assert fusion_names._retained_rule_name(skeleton) is None


@pytest.mark.parametrize("book_name", ["cyclohepta[a]naphthalene", "cyclohepta[b]naphthalene", "benzo[a]azulene",
                                       "benzo[a]heptalene"])
def test_the_three_ring_names_agree_with_the_older_builder(book_name):
    # fusion_names._rule_name names the three-ring skeletons first; where both name one, they name it alike
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smiles = opsin_parse(book_name)
    skeleton = Chem.MolFromSmiles(parents._key_of(Chem.MolFromSmiles(smiles)))
    older = fusion_names._rule_name(skeleton)
    assert older is not None
    assert fusion_names._retained_rule_name(skeleton) == older


ROWS = [
    # a hopane diol: C(CH3)2OH on the five-membered ring (position 1), the 3-hydroxy on position 9
    ("C[C@]12CC[C@H]([C@@H]1[C@H]3CC[C@@H]4[C@]5(CC[C@H](C([C@@H]5CC[C@]4([C@@]3(CC2)C)C)(C)C)O)C)C(C)(C)O",
     "(1R,3aR,5aR,5bR,7aR,9R,11aR,11bR,13aR,13bS)-1-(2-hydroxypropan-2-yl)-3a,5a,5b,8,8,11a-hexamethylicosahydro-"
     "1H-cyclopenta[a]chrysen-9-ol"),
    # a lupane acid: COOH on the fusion atom 3a, isopropenyl on 1, the 9-hydroxy
    ("CC(=C)C1CC[C@]2(C1C3CCC4[C@@](C3CC2)(CCC5[C@@]4(CCC(C5(C)C)O)C)C)C(=O)O",
     "(3aS,5bR,11aR)-9-hydroxy-5b,8,8,11a-tetramethyl-1-(prop-1-en-2-yl)icosahydro-3aH-cyclopenta[a]chrysene-"
     "3a-carboxylic acid"),
    # a B-nor steroid: the cyclopentane D ring is the 'a'-fused one, the aldehyde on the five-membered B ring
    ("CC(C)CCC[C@@H](C)[C@H]1CCC2C3C(C=O)[C@]4(O)C[C@@H](O)CC[C@]4(C)C3CC[C@@]21C",
     "(3R,3aR,5bR,8S,9aR)-8,9a-dihydroxy-3a,5b-dimethyl-3-[(2R)-6-methylheptan-2-yl]hexadecahydrocyclopenta[a]"
     "fluorene-10-carbaldehyde"),
    # an A-nor lupane diacid: the principal groups take 1 and 7a (the other numbering gives 10 and 12b)
    ("C=C(C)[C@@H]1CC[C@]2(C(=O)O)CC[C@]3(C)[C@H](CC[C@@H]4[C@@]5(C)[C@H](C(=O)O)[C@@H](O)C(C)(C)[C@@H]5CC[C@]43C)"
     "[C@@H]12",
     "(1S,2R,3aR,5aR,5bR,7aS,10R,10aR,10bR,12aR,12bR)-2-hydroxy-3,3,5a,5b,12b-pentamethyl-10-(prop-1-en-2-yl)"
     "octadecahydrodicyclopenta[a,i]phenanthrene-1,7a(1H)-dicarboxylic acid"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", ROWS)
def test_the_rows_are_named_and_read_back(smiles, name):
    with jvm_slots(1, purpose="bf-f1-carbo"):
        row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), (row["name"], row["tier"])
    assert name_is_rt_exact(name, smiles)


def test_without_the_function_the_skeletons_have_no_parent(monkeypatch):
    # mutation: with no retained-parent name (the state before the task) none of the three skeletons has a parent
    monkeypatch.setattr(fusion_names, "retained_parent_name", lambda key_mol: None)
    for key, _, _ in SKELETONS.values():
        assert parents._parent_for_key.__wrapped__(key) is None
