"""SF lane E, fix round: 'oxophosphanyl' is the prefix of a heterone only in the presence of a more
senior characteristic group.

 "HETERONES" (the Blue Book): "Compounds containing the -PO, -PO2, -AsO or -AsO2 are
called heterones (see,. In the presence of a more senior characteristic group
they are described by the compound prefixes oxophosphanyl,...". (:18162, item 16
:18189): 'ketones, pseudoketones and heterones' rank below radicals, ions, acids, anhydrides, esters,
acid halides, amides, hydrazides, imides, nitriles and aldehydes and above hydroxy compounds (17),
hydroperoxides (18), amines (19) and imines (20). (:29390): a heterone takes the suffix
'one' ('phenylphosphanone (PIN)',:28291), so '{4-[(oxophosphanyl)oxy]phenyl}methanol' is not the
PIN; it is '[4-(hydroxymethyl)phenoxy]phosphanone'. Ties inside class 16: (:29628) "no
seniority order difference between ketones and pseudoketones... the maximum number of carbonyl
groups or doubly bonded oxygen atoms",:18875), then (:18917) "N > P >... > O >
S > Se > Te > C": the phosphanone is the parent of a ketone on a carbon, oxygen or sulfur parent;
a ketone whose parent holds nitrogen, or two or more ketone groups, keeps the prefix.
 "Pseudoketones" (:28258): '(a) 1,3-dioxan-2-one (PIN)': a lactone and a cyclic carbonate
are class-16 pseudoketones on an oxygen ring, so they do not keep it either.

The label is lowered, never the name: the best-effort tier ships the name as
``systematic_verified``; the default tier declines it. Building the phosphanone parent is a
separate topic.
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_heterone import heterone_prefix_check
from tests.support.pin_tiers import name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# (smiles, the best-effort name, why the heterone is the principal characteristic group)
DECLINED = [
    ("C1=CC(=CC(=C1)OP=O)CO", "{3-[(oxophosphanyl)oxy]phenyl}methanol", "alcohol, class 17"),
    ("C1=CC(=CC=C1CN)OP=O", "1-{4-[(oxophosphanyl)oxy]phenyl}methanamine", "amine, class 19"),
    ("C1C2CC3CC1CC(C2)(C3)N=CC4=CC(=CC=C4)OP=O",
     "N-(adamantan-1-yl)-1-{3-[(oxophosphanyl)oxy]phenyl}methanimine", "imine, class 20"),
    ("C1=CC=C2C(=C1)C3=C2C(=CC=C3)OP=O", "1-[(oxophosphanyl)oxy]biphenylene", "a hydrocarbon"),
    ("CC(=O)C1=CC=CC=C1OP=O", "1-{2-[(oxophosphanyl)oxy]phenyl}ethan-1-one",
     "ketone on a carbon parent: P > C"),
    ("CC1=C(C(=C(C=C1)C(=O)C2=CC=CC=C2OP=O)C)C",
     "{2-[(oxophosphanyl)oxy]phenyl}(2,3,4-trimethylphenyl)methanone", "ketone on a carbon parent"),
    ("CC1=C(C(=O)OC2=C1C=CC(=C2)OP=O)C", "3,4-dimethyl-7-[(oxophosphanyl)oxy]-2H-1-benzopyran-2-one",
     "pseudoketone on an oxygen ring: P > O (P-64.1.2.1 (a))"),
    ("C1=CC(=CC=C1C2=C(OC(=O)O2)CBr)OP=O",
     "4-(bromomethyl)-5-{4-[(oxophosphanyl)oxy]phenyl}-2H-1,3-dioxol-2-one",
     "a cyclic carbonate: the pseudoketone of an oxygen ring, P > O (P-64.1.2.1 (a))"),
]

# the prefix is the PIN spelling: a ketone parent holding nitrogen, or a class senior to 16
KEPT = [
    ("C1=CC2=C(C=CC(=C2)OP=O)[NH+]=C1", "6-[(oxophosphanyl)oxy]quinolin-1-ium", "a cation, class 3"),
    ("C1COCC(N1OP=O)C2=C(C=C(C=C2)Cl)NS(=O)(=O)C3=CC(=C(C=C3)Cl)C(F)(F)F",
     "4-chloro-N-(5-chloro-2-{4-[(oxophosphanyl)oxy]morpholin-3-yl}phenyl)-3-(trifluoromethyl)"
     "benzene-1-sulfonamide", "a sulfonamide, class 11"),
    ("C1=CC2=C(C(=C1)OP=O)N=C(C(=N2)C#N)C#N", "5-[(oxophosphanyl)oxy]quinoxaline-2,3-dicarbonitrile",
     "a nitrile, class 14"),
    ("C1=CC=C2C(=C1)C(=O)N(C2=O)OP=O", "2-[(oxophosphanyl)oxy]-1H-isoindole-1,3(2H)-dione",
     "an imide on a nitrogen ring"),
    ("C1COCCN1C2=CC3=C(C(=C2)OP=O)C(=O)C4=CC=CC=C4N3",
     "3-(morpholin-4-yl)-1-[(oxophosphanyl)oxy]acridin-9(10H)-one", "a ketone on a nitrogen ring"),
    ("C1=CC=C2C(=C1)C(=O)NC(=N2)C3=CC=CC=C3OP=O", "2-{2-[(oxophosphanyl)oxy]phenyl}quinazolin-4(3H)-one",
     "a pseudoketone on a nitrogen ring"),
    ("COP(=O)(C1=CC=CC=C1OP=O)OC", "dimethyl {2-[(oxophosphanyl)oxy]phenyl}phosphonate",
     "an ester, class 9"),
]


@pytest.mark.parametrize("smiles,name,why", DECLINED)
def test_the_prefix_is_not_a_pin_where_the_heterone_is_the_principal_group(smiles, name, why):
    b = name_breadth(smiles)
    assert b.get("name") == name and b.get("tier") == "systematic_verified", (why, b)
    assert name_is_rt_exact(name, smiles), name
    d = name_default(smiles)
    assert d.get("tier") != "pin_verified", (why, d)
    assert not d.get("is_pin"), (why, d)


@pytest.mark.parametrize("smiles,name,why", KEPT)
def test_the_prefix_is_the_pin_where_a_senior_group_or_a_nitrogen_ketone_parent_holds(smiles, name, why):
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (name, "pin_verified", True), (why, d)


# ---- the check on structures alone: the ties inside class 16 -------------------------------

PRESENT = "x-[(oxophosphanyl)oxy]-y"


def _declines(smiles, name=PRESENT):
    return heterone_prefix_check(Chem.MolFromSmiles(smiles), name) is not None


@pytest.mark.parametrize("smiles", [
    "O=C1CCCC(=O)C1OP=O",            # two ketone groups on one ring:, the count
    "O=C1C=CC(=O)C=C1OP=O",          # a quinone
    "CC(=O)OCCOP=O", "OC(=O)CCOP=O", "CC(C)(C)c1cc(OP=O)ccc1C=O", "N#CCCOP=O",
    "O=S(=O)(N)c1ccc(OP=O)cc1", "C[N+](C)(C)CCOP=O", "OC(=O)CC(=O)OCCOP=O",
])
def test_senior_class_or_a_senior_ketone_parent_keeps_the_prefix(smiles):
    assert not _declines(smiles)


@pytest.mark.parametrize("smiles", [
    "O=C1CCCC1OP=O",                 # one ketone on a carbon ring
    "CC(=O)CCOP=O",                  # one ketone on a chain
    "O=C1CCOC1OP=O",                 # one ketone on an oxygen ring
    "CCCCCCCOP=O", "OCCOP=O", "NCCOP=O", "CC=NCCOP=O", "ClCCOP=O",
    "O=[N+]([O-])c1ccc(OP=O)cc1",    # a nitro group is a prefix, not a class
    "CC(=O)N1CCCCC1.OCCOP=O",        # components are read one by one: this one has no senior group
])
def test_no_senior_group_declines(smiles):
    assert _declines(smiles)


def test_a_name_without_the_prefix_is_not_read():
    assert not _declines("CCCCCCCOP=O", name="1-(phosphanonyloxy)heptane")
