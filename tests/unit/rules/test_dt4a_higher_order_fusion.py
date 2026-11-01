"""DT4-A: second-order and higher attached components in polycomponent fusion names
(``bridged_fused_pin.poly_fusion._tree_name``).

Before this, ``poly_fusion.rule_name`` built fusion names from first-order attached components only, so an
ortho-fused skeleton with a component fused to an attached component (the steroid acetonide
'naphtho[2',1':4,5]indeno[1,2-d][1,3]dioxole' class, the loratadine core
'benzo[5,6]cyclohepta[1,2-b]pyridine') had no fused parent and the default tier declined it.

Blue Book (VERIFIED, ``the Blue Book Blue Book``):

* ``### **** Constructing polycomponent fusion names`` (:12706): "A component attached to a
  first-order attached component is called a 'second-order' attached component, and so on."
* ``### **** Fusion names composed of first- and higher-order attached components`` (:12718):
  "numerical locants are used instead of letters and the two sets of locants are separated by a colon. The
  locants of second-order attached components are primed to contrast with those of first-order attached
  components. The locants of third-order attached components are doubly primed, and so on."
* ``### **** Order of seniority for selecting parent components`` (b) (:12970): "the location that
  results in a name that does not require attached components higher than first-order".
* ``### **** Order of citation of fusion prefixes`` (:13040): "Each second-order
  attached component is cited in front of the first-order attached component to which it is fused... If there
  are two or more different components, or sets of components, attached to a lower order component, they are
  cited in alphabetical order".
* ``## **** Order of seniority of locants (letters and numbers)`` (:13076) (a)-(h).

Each expected name below is read back by OPSIN 2.9.0 to the full InChIKey of its molecule; the book names
are the printed '(PIN)' ones (indicated hydrogen left to the caller), the line of the book in the comment.
"""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import parents, poly_fusion
from tests.support.pin_tiers import name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _skeleton(smiles):
    return Chem.MolFromSmiles(parents._key_of(Chem.MolFromSmiles(smiles)))


#: (SMILES by OPSIN 2.9.0's reading of the printed name, the printed '(PIN)' name, its line in the Blue Book)
BOOK_HIGHER_ORDER = [
    ('C1=COC2=c3oc4c(c3C=CSC2=C1)N=NC=4', "pyrano[2'',3'':6',7']thiepino[4',5':4,5]furo[3,2-c]pyrazole", 13433),
    ('C1=Nc2nc3c(nc2=N1)-c1occc1N=3', "furo[2',3':4,5]pyrrolo[2,3-b]imidazo[4,5-e]pyrazine", 13048),
    ('c1ccc2c(c1)[nH]c1c2ccc2c3ccoc3ncc21', "furo[3',2':5,6]pyrido[3,4-a]carbazole", 13397),
    ('c1ccc2c3c4cnnc-4occc-3nc2c1', "pyrazolo[4',3':6,7]oxepino[4,5-b]indole", 13409),
    ('c1coc2ccc3ccoccc-3cc-2c1', "pyrano[3',2':4,5]cyclohepta[1,2-d]oxepine", 13417),
    ('c1ccc2nc3ccc4nccc-4cc3cc2c1', "pyrrolo[3',2':4,5]cyclohepta[1,2-b]quinoline", 13423),
    ('C1=c2c(ccc3c2=Cc2ncccc2-3)OC1', "furo[3',2':4,5]indeno[2,1-b]pyridine", 13461),
    ('c1ccc2c(c1)ccc1c3ccsc3c3ccc4ccccc4c3c21', "naphtho[2',1':3,4]phenanthro[1,2-b]thiophene", 13026),
    ('C1=Cc2c3c(c4ccccc4c2=C1)=NCO3', 'cyclopenta[3,4]naphtho[1,2-d][1,3]oxazole', 13012),
]


@pytest.mark.parametrize("smiles,name,line", BOOK_HIGHER_ORDER)
def test_the_book_higher_order_names_are_rebuilt_by_the_rules(smiles, name, line):
    """The printed name of each example is the one the rules give: primed locants, colon, order of citation,
    the lowest letters and locants of (a)-(h) (the book line is the third parameter)."""
    assert poly_fusion.rule_name(_skeleton(smiles)) == name


@pytest.mark.parametrize("smiles,name,line", BOOK_HIGHER_ORDER[1:7])
def test_the_fused_parent_of_a_book_higher_order_skeleton_is_its_printed_name(smiles, name, line):
    mol = Chem.MolFromSmiles(smiles)
    got = parents.fused_parent(mol, set(range(mol.GetNumAtoms())))
    assert got is not None and got.name == name


#: the cause-A rows of the DT4 census (internal notes) the lane names at the default tier: the whole
#: molecule, the name; each name reads back to the full InChIKey
CENSUS_ROWS = [
    ('1', 'CN1CC(=O)N2[C@H](c3ccc4c(c3)OCO4)c3[nH]c4ccccc4c3C[C@@H]2C1=O', "(6R,12aR)-6-(2H-1,3-benzodioxol-5-yl)-2-methyl-2,3,6,7,12,12a-hexahydropyrazino[1',2':1,6]pyrido[3,4-b]indole-1,4-dione"),
    ('7', 'CC1(C)O[C@@H]2C[C@H]3[C@@H]4C[C@H](F)C5=CC(=O)C=C[C@]5(C)[C@@]4(F)[C@@H](O)C[C@]3(C)[C@]2(C(=O)CO)O1', "(4aS,4bR,5S,6aS,6bS,9aR,10aS,10bS,12S)-4b,12-difluoro-5-hydroxy-6b-(hydroxyacetyl)-4a,6a,8,8-tetramethyl-4a,4b,5,6,6a,6b,9a,10,10a,10b,11,12-dodecahydro-2H,8H-naphtho[2',1':4,5]indeno[1,2-d][1,3]dioxol-2-one"),
    ('10', 'CC1CC(C(O)C(C)(C)O)OC2CC3(C)C4(C)CCC5C(C)(C)C(=O)CCC5(C)C4CCC3(O)C12O', "9-(1,2-dihydroxy-2-methylpropyl)-6a,6b-dihydroxy-1,1,4a,7,11a,11b-hexamethyloctadecahydronaphtho[2',1':4,5]indeno[2,1-b]pyran-2(1H)-one"),
    ('24', 'CC1(C)O[C@@H]2C[C@H]3[C@@H]4CCC5=CC(=O)C=C[C@]5(C)[C@@]4(F)[C@@H](O)C[C@]3(C)[C@]2(C(=O)CO)O1', "(4aS,4bR,5S,6aS,6bS,9aR,10aS,10bS)-4b-fluoro-5-hydroxy-6b-(hydroxyacetyl)-4a,6a,8,8-tetramethyl-4a,4b,5,6,6a,6b,9a,10,10a,10b,11,12-dodecahydro-2H,8H-naphtho[2',1':4,5]indeno[1,2-d][1,3]dioxol-2-one"),
    ('34', 'O=c1[nH]c2c(nc3ccccn32)c2ccccc12', "pyrido[2',1':2,3]imidazo[4,5-c]isoquinolin-5(6H)-one"),
    ('51', 'O=C1O[C@H]2C(=C1c1c[nH]c3ccccc13)c1c([nH]c3ccccc13)[C@H]2c1c[nH]c2ccccc12', "(9R,9aR)-3,9-di(1H-indol-3-yl)-9,9a-dihydrofuro[3',2':3,4]cyclopenta[1,2-b]indol-2(8H)-one"),
    ('52', 'C=C1CC[C@@H]2[C@@]3(C)CC=C4[C@H](C(C)C)CC[C@@]4(C)[C@@H]3C[C@H](O)[C@@]2(C)[C@@H]2O[C@@H](c3ccc(O)cc3)O[C@@H]12', "(2S,3aS,6aR,6bS,9S,11aS,11bR,13S,13aS,13bS)-2-(4-hydroxyphenyl)-6b,11a,13a-trimethyl-4-methylidene-9-(propan-2-yl)-3a,4,5,6,6a,6b,7,9,10,11,11a,11b,12,13,13a,13b-hexadecahydro-2H-cyclopenta[5',6']naphtho[2',1':3,4]cyclohepta[1,2-d][1,3]dioxol-13-ol"),
    ('64', 'COc1ccc2c3c([nH]c2c1)[C@H](CC(C)(C)O)N1C(=O)[C@@H]2CCCN2C(=O)[C@@]1(O)[C@H]3O', "(5aS,6S,12S,14aS)-5a,6-dihydroxy-12-(2-hydroxy-2-methylpropyl)-9-methoxy-1,2,3,5a,6,11,12,14a-octahydro-5H,14H-pyrrolo[1'',2'':4',5']pyrazino[1',2':1,6]pyrido[3,4-b]indole-5,14-dione"),
    ('90', 'C[C@]12CC=C3[C@H]([C@@H]1C[C@@H]4[C@]2(OC(O4)(C)C)C(=O)CF)CCC5=CC(=O)C=C[C@@]53C', "(4aS,6aS,6bS,9aR,10aS,10bS)-6b-(fluoroacetyl)-4a,6a,8,8-tetramethyl-4a,6,6a,6b,9a,10,10a,10b,11,12-decahydro-2H,8H-naphtho[2',1':4,5]indeno[1,2-d][1,3]dioxol-2-one"),
]


@pytest.mark.parametrize("row,smiles,name", CENSUS_ROWS)
def test_census_rows_are_named_at_the_default_tier(row, smiles, name):
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (name, "pin_verified", True), d
    assert name_is_rt_exact(name, smiles)


#: molecules the default tier named before DT4-A with a fusion name of first-order components only: their names
#: must not change (recorded before the change)
CONTROLS = [
    ('c1cc2nc3ccsc3cc2o1', 'furo[3,2-b]thieno[2,3-e]pyridine'),
    ('c1cc2nc3cocc3nc2s1', 'furo[3,4-b]thieno[2,3-e]pyrazine'),
    ('C1=CN2C=CC3=CNOC3=C2O1', '2H-[1,2]oxazolo[5,4-c][1,3]oxazolo[3,2-a]pyridine'),
    ('c1cc2nc3ccoc3cc2o1', "difuro[3,2-b:2',3'-e]pyridine"),
    ('c1cc2cc3csnc3cc2s1', 'thieno[3,2-f][2,1]benzothiazole'),
    ('c1cnc2cc3cc4cnoc4cc3cc2c1', '[1,2]benzoxazolo[6,5-g]quinoline'),
    ('C1=NC=c2ccc3c(c21)N=c1cc2occc2cc1=3', 'furo[3,2-h]pyrrolo[3,4-a]carbazole'),
    ('c1ccc2c(c1)ccc1ccc3ccc4ccccc4c3c12', 'dibenzo[c,g]phenanthrene'),
    ('c1cc2nc3sccc3nc2cn1', 'pyrido[3,4-b]thieno[2,3-e]pyrazine'),
    ('C1=Cc2nc3c(cc2=N1)OCC=3', '2H-furo[3,2-b]pyrrolo[2,3-e]pyridine'),
    ('C1=Nc2cc3c(nc2=N1)SCC=3', '6H-imidazo[4,5-b]thieno[3,2-e]pyridine'),
    ('c1cc2cc3ccsc3nc2o1', 'furo[2,3-b]thieno[3,2-e]pyridine'),
    ('c1ccc2c(c1)ccc1occc12', 'naphtho[2,1-b]furan'),
    ('c1cc2c(cnc3sccc32)s1', "dithieno[2,3-b:3',2'-d]pyridine"),
    ('C1=Cc2nc3c(cc2=N1)N=CC=3', "dipyrrolo[3,2-b:2',3'-e]pyridine"),
    ('c1nc2nc3ccsc3cc2o1', 'oxazolo[4,5-b]thieno[2,3-e]pyridine'),
]


@pytest.mark.parametrize("smiles,name", CONTROLS)
def test_first_order_fusion_names_are_unchanged(smiles, name):
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier")) == (name, "pin_verified"), d


#: boundary rows that stay declined by this builder (the parent name is None): the book prints names for them
#: that need identical first-order attached components with a higher-order one (:13522 'furo[3,4-b]furo[3',2':4,5]
#: furo[2,3-e]pyridine'), a multiparent name, a benzo unit or more than six rings
DECLINED_SKELETONS = [
    "C1COC2OC3CCC4CCCCC4C3CC2C1",         # census row 18: two pyran rings, a multiparent skeleton
    "C1CCC2OC3C(CCC4OCCCC43)COC2C1",      # census row 20: a benzo unit and a dibenzo reading
    "C1CCC2C(C1)CCC1OC3CCC4COCCC4C3CC12", # census row 33's relative: pyran + benzene + pyran (benzo unit)
]


@pytest.mark.parametrize("key", DECLINED_SKELETONS)
def test_skeletons_outside_the_class_get_no_fusion_name(key):
    assert poly_fusion.rule_name(_skeleton(key)) is None


def test_identical_first_order_components_with_a_higher_order_one_stay_declined():
    """:13522 prints 'furo[3,4-b]furo[3',2':4,5]furo[2,3-e]pyridine (PIN)' (three furans on a pyridine, one fused
    to another): the builder takes only distinct first-order components, so it declines (fail closed) and
    does not offer a different name."""
    key = _skeleton("O1C=CC2=C1OC1=CC=3C(N=C12)=COC3")
    assert poly_fusion._tree_name(key, *_rings(key)) is None


def _rings(key):
    from orthonym.rules.bridged_fused_pin import fusion_components as fc, hetero_fusion as hf
    rings = fc.ortho_rings(key)
    return rings, hf._stage1_components(key, rings)


#: C1 (fix a performance pass): the locants are compared order by order, ``## **** Order of seniority of locants`` (h)
#: (:13433): "pyrano[2'',3'':6',7']thiepino[4',5':4,5]furo[3,2-c]pyrazole (PIN) (not pyrano[3'',2'':2',3']thiepino
#: [5',4':4,5]furo[3,2-c]pyrazole; the locant set '4',5'' is lower than '5',4'')". The oxepine and thiopyran/thiepine
#: analogues follow the same rule (OPSIN 2.9.0 reads each name to its skeleton, checked by the parent read-back).
ORDER_BY_ORDER = [
    ("C1=COC2=c3oc4c(c3C=CSC2=C1)N=NC=4", "pyrano[2'',3'':6',7']thiepino[4',5':4,5]furo[3,2-c]pyrazole",
     "pyrano[3'',2'':2',3']thiepino[5',4':4,5]furo[3,2-c]pyrazole"),
    ("O1C=CC=C2C1=C1C(=C3N=NC=C3O1)C=CO2", "pyrano[2'',3'':6',7']oxepino[4',5':4,5]furo[3,2-c]pyrazole",
     "pyrano[3'',2'':2',3']oxepino[5',4':4,5]furo[3,2-c]pyrazole"),
    ("S1C=CC=C2C1=C1C(=C3N=NC=C3O1)C=CS2", "thiopyrano[2'',3'':6',7']thiepino[4',5':4,5]furo[3,2-c]pyrazole",
     "thiopyrano[3'',2'':2',3']thiepino[5',4':4,5]furo[3,2-c]pyrazole"),
]


@pytest.mark.parametrize("smiles,name,rejected", ORDER_BY_ORDER)
def test_locants_are_decided_order_by_order_through_the_rules(smiles, name, rejected):
    got = poly_fusion.rule_name(_skeleton(smiles))
    assert got == name and got != rejected


def test_the_direction_of_a_higher_order_pair_is_chosen_by_the_lowest_set():
    """C2: ``pyrrolo[3',2':4,5]cyclohepta[1,2-b]quinoline`` (:13423, 'not pyrrolo[2',3':5,4]...; the locant set
    '4,5' is lower than '5,4''): a pair whose side closes back to locant 1 is read '1,6', not '6,1' (tadalafil's
    skeleton, census row 1: 'pyrazino[1',2':1,6]pyrido[3,4-b]indole')."""
    key = _skeleton("C1CCC2C(C1)NC1CN3CCNCC3CC12")
    assert poly_fusion.rule_name(key) == "pyrazino[1',2':1,6]pyrido[3,4-b]indole"


def test_a_descriptor_with_a_fusion_atom_locant_is_declined():
    # conanine: the reading 'naphtho[2',1':4,5]indeno[1,7a-c]pyrrole' fuses the pyrrole at the indene bond 1-7a; the book
    # prints no higher-order descriptor with a component's own fusion-atom locant, so the builder declines it and the list
    # name stays INTRODUCTION, the Blue Book; the exact-match list parent keeps its label)
    from orthonym import Orthonym
    from orthonym.rules.bridged_fused_pin.poly_fusion import _FUSION_ATOM_LOCANT
    assert _FUSION_ATOM_LOCANT.search("naphtho[2',1':4,5]indeno[1,7a-c]pyrrole")
    assert not _FUSION_ATOM_LOCANT.search("pyrazino[1',2':1,6]pyrido[3,4-b]indole")
    conanine = "C[C@H]1[C@H]2CC[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]23CN1C"   # the gold row P7A-CONANINE
    row = Orthonym().name_tiered(conanine)
    assert (row["name"], row["tier"]) == ("conanine", "systematic_verified"), row
