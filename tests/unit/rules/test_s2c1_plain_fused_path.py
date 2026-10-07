"""Slice S2c-1 in the plain fused path: a two-ring fused system no catalogue entry holds is
named on the certified parent source of the bridged fused builder.

 (the Blue Book, heading "General principles"): "'ortho-Fused' or 'ortho- and
peri-fused' polycyclic ring systems with the maximum number of noncumulative double bonds
(mancude) that have no accepted retained or systematic name described in sections and
 are named by prefixing to the name of a component ring or ring system (the parent
component) designations of the other component(s) (attached components)." (:23710):
"Fusion nomenclature gives preferred IUPAC names only to compounds having at least two rings of
at least five or more members." (:11815): the benzo names, "for preferred IUPAC
names locants must be cited". (:14607): "In preferred IUPAC names, all indicated
hydrogen atoms must be cited when the names are constructed in accordance with the principles
of fusion nomenclature"; (:8320) the indicated hydrogen positions are the ring
atoms of bonding number three or more left with single bonds only after the maximum number of
noncumulative double bonds is placed (B, Si, Ge, Pb, P and Bi too: '1H-2,1,3-benzoxadisiline
(PIN)':21209, '2,7-dihydroxy-2H-1,3,2-benzodioxabismole-5-carboxylic acid (PIN)':39298,
'2-phenyl-2H,4H-[1,3,2]dioxaborolo[4,5-d]imidazole (PIN)':37248).

BOOK_PARENTS: two-ring parents of ``parent_table.BB_PARENTS`` (OPSIN 2.9.0's structure of the
printed name) that the plain path did not name as their PIN; RULE_BENZO_NAMES: benzo
names of the same class the book does not print; each the pin_verified name at both tiers.
ACCOMMODATING: a ketone on a position whose indicated hydrogen accommodates it,
:24768), cited by the shared hydrogen rule (``ring_hydrogen``): the pin_verified name at both
tiers.
OLDER_PATH: systems the certified route leaves to the older path unchanged (hydro prefixes or
added hydrogen, the lambda-convention of:12452, a ring acyl group the fused reader
would write as a prefix, a ring C=O beside a senior suffix group, which the shared hydrogen rule
does not write and the older path cites as 'oxo',:24864).

TARGET (spec item Q2): 'N=1N2C(C=CN1)=CC(C=C2)=O'. Components 1,2,3-triazine (Hantzsch-Widman,
 and pyridine; (d) (:12260) "A component containing the greater number of
heteroatoms of any kind" makes the triazine the parent; 'pyrido':12047); side c
(N3-C4) of [1,2,3]triazine with the pyridine locants 1,2 in its direction (:11911); numbering
N1, N2, C3, C4, C4a, C5, C6, C7, C8, N9 (OPSIN's reading of the name); the mancude parent takes
one indicated hydrogen, placed at the position of the suffix: (:24768) "When there
are an equal number of indicated hydrogen atoms and principal characteristic groups... the
indicated hydrogen atoms are placed at peripheral atoms that will accommodate these principal
characteristic groups", '7H-1-benzopyran-7-one (PIN)' (:24776). OPSIN 2.9.0 reads
'6H-pyrido[1,2-c][1,2,3]triazin-6-one' back to the input's full InChIKey
(an InChIKey). The indicated hydrogen of a suffix position comes from the
hydrogen lane's shared rule (spec Q1, ``rules/ring_hydrogen.py``).
"""
import random

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules import fused_rings
from tests.support.rt_assert import name_is_rt_exact

TARGET = 'N=1N2C(C=CN1)=CC(C=C2)=O'
TARGET_PIN = '6H-pyrido[1,2-c][1,2,3]triazin-6-one'

BOOK_PARENTS = [
    ('O1SSC2=C1C=CC=C2', '1,2,3-benzoxadithiole'),
    ('S1SCC2=C1C=CC=C2', '3H-1,2-benzodithiole'),
    ('N1=PN=CC2=C1C=CC=C2', '1,3,2-benzodiazaphosphinine'),
    ('O1P=NC2=C1C=CC=C2', '1,3,2-benzoxazaphosphole'),
    ('O1COCC2=C1C=CC=C2', '2H,4H-1,3-benzodioxine'),
    ('O1COC2=C1C=CC=C2', '2H-1,3-benzodioxole'),
    ('S1CSC2=C1C=CC=C2', '2H-1,3-benzodithiole'),
    ('O1C=COC2=C1C=CC=C2', '1,4-benzodioxine'),
    ('S1C=CSC=CC2=C1C=CC=C2', '1,4-benzodithiocine'),
    ('S1CC=CSC2=C1C=CC=C2', '2H-1,5-benzodithiepine'),
    ('S1C=CC=NC2=C1C=CC=C2', '1,5-benzothiazepine'),
    ('N1=CC=CC=CC2=C1C=CC=C2', '1-benzazocine'),
    ('[Se]1CC=CC2=C1C=CC=C2', '2H-1-benzoselenopyran'),
    ('[Te]1CC=CC2=C1C=CC=C2', '2H-1-benzotelluropyran'),
    ('S1OCC2=C1C=CC=C2', '3H-2,1-benzoxathiole'),
    ('C1ON=CC2=C1C=CC=C2', '1H-2,3-benzoxazine'),
    ('C=1OC=COC=C2C1C=CC=C2', '2,5-benzodioxocine'),
    ('C1[Se]C=CC2=C1C=CC=C2', '1H-2-benzoselenopyran'),
    ('C1[Te]C=CC2=C1C=CC=C2', '1H-2-benzotelluropyran'),
    ('C1SC=CC2=C1C=CC=C2', '1H-2-benzothiopyran'),
    ('O1CC=CC2=C1OC=CC2', '2H,5H-pyrano[2,3-b]pyran'),
    ('O1C=2C(C=CC1)=COC2', '2H-furo[3,4-b]pyran'),
    ('[As]1=COC=[As]C2=C1C=CC=C2', '3,1,5-benzoxadiarsepine'),
    ('N=1COC=C2C1C=CC=C2', '2H-3,1-benzoxazine'),
    ('C1=CNC=CC=CC=CC2=C1C=CC=C2', '3H-3-benzazacycloundecine'),
    ('C=1OCC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CN2C1C=CC=C2', '3H-pyrido[2,1-c][1,4]oxaazacyclohentriacontine'),
    ('N1=NCOC2=C1C=CC=C2', '3H-4,1,2-benzoxadiazine'),
    ('S1OOCC2=C1C=COC2', '4H,5H-pyrano[4,3-d][1,2,3]dioxathiine'),
    ('O1SOCC2=C1C=COC2', '4H,5H-pyrano[4,3-d][1,3,2]dioxathiine'),
    ('C1=CC=COC=CC=CC=COC=CC=CC2=C1C=CC=C2', '5,12-benzodioxacyclooctadecine'),
    ('C=1SC=CN=CC=COC=C2C1C=CC=C2', '9,2,5-benzoxathiaazacyclododecine'),
    ('O1COC=2OC=CC=CC21', '2H-[1,3]dioxolo[4,5-b]oxepine'),
    ('P=1OC=C2C1OCO2', '5H-[1,3]dioxolo[4,5-c][1,2]oxaphosphole'),
    ('O1COC=2COC=CC21', '2H,4H-[1,3]dioxolo[4,5-c]pyran'),
    ('O1P=CC2=C1OCO2', '5H-[1,3]dioxolo[4,5-d][1,2]oxaphosphole'),
    ('O1C=CC=COC=2C1=COC=CC=COC2', '[1,4]dioxocino[2,3-c][1,6]dioxecine'),
    ('S1CC=CSC=2C1=COC2', '2H-[1,4]dithiepino[2,3-c]furan'),
    ('O1C2=C(SC=C1)[Se]C=CO2', '[1,4]oxaselenino[2,3-b][1,4]oxathiine'),
    ('O1C=2C(=COC=CC=C1)OC=CC=COC2', '[1,5]dioxonino[3,2-b][1,5]dioxonine'),
    ('C=1C=CC[As]2C=CC=CC12', '4H-arsinolizine'),
    ('N1C=2C(=CC=C1)C=CC=CC2', '1H-cyclohepta[b]pyridine'),
    ('C=1SC=CC=2C1C=CC=CC2', 'cyclohepta[c]thiopyran'),
    ('C1=BC=C2C1=CC=C2', 'cyclopenta[c]borole'),
    ('C1OC=C2C1=CC=C2', '1H-cyclopenta[c]furan'),
    ('O1C=CC2=C1COC=C2', '7H-furo[2,3-c]pyran'),
    ('O1CSC=2C1=COC2', '2H-furo[3,4-d][1,3]oxathiole'),
    ('O1C2C(=CC=C1)P=CC=C2', '8aH-phosphinino[3,2-b]pyran'),
    ('O1C=COC=C2C1=CC=CO2', 'pyrano[3,2-e][1,4]dioxepine'),
    ('C=1OC=CC=2C1C=COC2', 'pyrano[4,3-c]pyran'),
    ('C1=C2N(CC=N1)C=CC=C2', '4H-pyrido[1,2-a]pyrazine'),
    ('N=1CSC=C2C1C=CC=N2', '2H-pyrido[3,2-d][1,3]thiazine'),
    ('O1[BiH]OC2=C1C=CC=C2', '2H-1,3,2-benzodioxabismole'),
    ('O1[GeH2]OC2=C1C=CC=C2', '2H-1,3,2-benzodioxagermole'),
    ('O1POC2=C1C=CC=C2', '2H-1,3,2-benzodioxaphosphole'),
    ('[SiH2]1O[SiH]=CC2=C1C=CC=C2', '1H-2,1,3-benzoxadisiline'),
    ('[SiH2]1OC=[SiH]C2=C1C=CC=C2', '1H-2,1,4-benzoxadisiline'),
    ('B1N=CC2=C1C=CC=C2', '1H-2,1-benzazaborole'),
    ('C=1O[PbH2]OC=C2C1C=CC=C2', '3H-2,4,3-benzodioxaplumbepine'),
    ('O1BON2C1=CC=CC2', '2H,5H-[1,3,4,2]dioxazaborolo[4,5-a]pyridine'),
    ('O1BOC2=C1N=CN2', '2H,4H-[1,3,2]dioxaborolo[4,5-d]imidazole'),
    ('B1C=CC2=C1C=CC=C2', '1H-1-benzoborole'),
]

RULE_BENZO_NAMES = [
    ('c1ccc2c(c1)[SiH2]cc2', '1H-1-benzosilole'),
    ('O1C=CC=CC=Cc2ccccc21', '1-benzoxonine'),
    ('S1C=CC=CC=Cc2ccccc21', '1-benzothionine'),
]

ACCOMMODATING = [
    (TARGET, TARGET_PIN),
    ('O=c1ccn2ncccc2c1', '6H-pyrido[1,2-b]pyridazin-6-one'),
]

OLDER_PATH = [
    ('c1cc2c[nH]cc2[nH]1', 'hydro: both N-H of pyrrolo[3,4-b]pyrrole'),
    # hydro on the retained quinoline skeleton: the older path names it, below the PIN,
    # '5H,8H-cyclohexa[b]pyridine' (the PIN is '5,8-dihydroquinoline';,
    # the Blue Book, and Table 2.8 (13),:11571)
    ('C1=CCc2ncccc2C1', 'hydro: 5,8-dihydro on the retained quinoline skeleton'),
    ('S1C=NC2=C1CCCCC2', 'hydro: the tetrahydrocyclohepta[d][1,3]thiazole'),
    ('O=c1ccc2ccccc2[nH]1', 'added hydrogen: quinolin-2(1H)-one'),
    ('N1=CS=CC2=C1C=CC=N2', 'lambda-convention: a pyridothiazine'),
    ('c1ccc2c(c1)O[SH2]O2', 'lambda-convention: a benzodioxathiole'),
    ('S1S2C(=CC1)C=CS2', 'lambda-convention: the dithiolodithiole of :14545'),
    ('CC(C)(C)N(C(C)(C)C)C(=O)c1ccc2OCOc2c1', 'a ring amide the fused reader writes as a prefix'),
    ('CC(=O)c1ccc2OCOc2c1', 'a ring ketone the fused reader writes as a prefix'),
    # a ring C=O beside a senior suffix group: the shared hydrogen rule does not write it; the
    # older path cites 'oxo' after the indicated hydrogen of its position,
    # the Blue Book) and keeps 'pin_verified' ('4-oxo-4H-1-benzopyran-2-carboxylic acid',
    # tests/unit/rules/test_oxo_prefix_senior_suffix.py)
    ('OC(=O)c1cc(=O)c2ccccc2o1', 'a ring C=O beside a senior suffix: the oxo prefix of the older path'),
    ('O=Cc1coc2ccccc2c1=O', 'a ring C=O beside a senior suffix: the oxo prefix of the older path'),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="s2c1-plain"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def _system(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring = {a for r in mol.GetRingInfo().AtomRings() for a in r}
    return mol, ring


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", BOOK_PARENTS + RULE_BENZO_NAMES)
def test_a_two_ring_parent_gets_its_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", ACCOMMODATING)
def test_the_suffix_position_cites_its_indicated_hydrogen(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)


#: the public emit tiers: the default 'pin', 'valid' (the general fallback of the web page),
#: 'complete' and 'best-effort' (the name-quality program spec, section 1, "Tiers")
PUBLIC_TIERS = ["pin", "valid", "complete", "best-effort"]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", PUBLIC_TIERS)
def test_no_tier_names_the_target_by_von_baeyer(tier):
    # (the Blue Book): the system has two rings of six members, so its
    # fusion name is the PIN; (:23843) ranks fused ring systems above non-fused
    # bridged systems
    row = _row(TARGET, tier)
    name = row.get("name")
    assert name.endswith("pyrido[1,2-c][1,2,3]triazin-6-one"), name
    assert "bicyclo[" not in name
    assert name_is_rt_exact(name, TARGET)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", PUBLIC_TIERS)
def test_the_target_gets_its_certified_fusion_pin(tier):
    row = _row(TARGET, tier)
    assert (row.get("name"), row["tier"]) == (TARGET_PIN, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(TARGET_PIN, TARGET)


@pytest.mark.parametrize("smiles,count", [
    ("O1[BiH]OC2=C1C=CC=C2", 1),        #:39298 '2H-1,3,2-benzodioxabismole'
    ("[SiH2]1O[SiH]=CC2=C1C=CC=C2", 1),  #:21209 '1H-2,1,3-benzoxadisiline (PIN)'
    ("O1[GeH2]OC2=C1C=CC=C2", 1),
    ("O1BOC2=C1N=CN2", 2),               #:37248 '2H,4H-[1,3,2]dioxaborolo[4,5-d]imidazole'
    (TARGET, 1),
    ("c1ccc2[nH]ccc2c1", 1),             # 1H-indole
    ("Cn1ccc2ccccc21", 1),               # 1-methyl-1H-indole: the substituted N is the position
    ("c1ccn2cccc2c1", 0),                # indolizine
    ("c1ccc2ccccc2c1", 0),               # naphthalene
    ("O1COCC2=C1C=CC=C2", 2),            # 2H,4H-1,3-benzodioxine
    ("O1CC=CC2=C1OC=CC2", 2),            # '2H,5H-pyrano[2,3-b]pyran (PIN)':24649
])
def test_the_indicated_hydrogen_of_the_mancude_parent(smiles, count):
    mol, ring = _system(smiles)
    got = fused_rings._certified_parent_hydrogen(mol, ring)
    assert got is not None and got[2] == count, got


def test_the_suffix_position_is_the_accommodating_one():
    mol, ring = _system(TARGET)
    indicated, accommodating, count = fused_rings._certified_parent_hydrogen(mol, ring)
    assert (len(indicated), count) == (0, 1)
    (atom,) = accommodating
    assert mol.GetAtomWithIdx(atom).GetSymbol() == "C"
    assert any(b.GetOtherAtom(mol.GetAtomWithIdx(atom)).GetSymbol() == "O"
               and b.GetBondTypeAsDouble() == 2.0 for b in mol.GetAtomWithIdx(atom).GetBonds())


@pytest.mark.parametrize("smiles,why", OLDER_PATH)
def test_the_certified_route_leaves_these_to_the_older_path(smiles, why):
    mol, ring = _system(smiles)
    has_subs = any(a.GetIdx() not in ring for a in mol.GetAtoms())
    assert fused_rings._certified_fusion_name(mol, ring, has_subs) is None, why


@pytest.mark.parametrize("smiles", [s for s, why in OLDER_PATH if not why.startswith("a ring")])
def test_hydro_added_hydrogen_and_lambda_are_not_the_mancude_parent(smiles):
    mol, ring = _system(smiles)
    assert fused_rings._certified_parent_hydrogen(mol, ring) is None


@pytest.mark.opsin_gate
def test_a_mancude_system_without_an_aromatic_ring_takes_no_hydro_prefix():
    # (:14569): the extra hydrogen of a mancude system is indicated hydrogen; the
    # fusion N (three ring bonds) carries none, so never '4,5-dihydropyrido[1,2-a]pyrazine'
    row = _row("C1=C2N(CC=N1)C=CC=C2", "pin")
    assert (row.get("name"), row["tier"]) == ("4H-pyrido[1,2-a]pyrazine", "pin_verified")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("[Se]1CC=CC2=C1C=CC=C2", "2H-1-benzoselenopyran"),
    ("C1SC=CC2=C1C=CC=C2", "1H-2-benzothiopyran"),
])
def test_an_imported_trivial_name_yields_to_the_certified_benzo_name(smiles, name):
    # 'selenochromene' / 'isothiochromene' have no PIN evidence (Table 2.8:11656,:11682 give
    # the benzo names as the PINs of chromene and isochromene)
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (tier, row.get("name"))


@pytest.mark.opsin_gate
def test_a_substituted_system_cites_the_indicated_hydrogen_at_the_substituted_nitrogen():
    # (:14607); '(1H-indol-1-yl)acetic acid (PIN)' (:2039) numbers the N-H
    # position as such; the certified parent pyrrolo[3,2-b]pyridine (BB_PARENTS)
    row = _row("Cn1ccc2ncccc21", "pin")
    assert (row.get("name"), row["tier"]) == ("1-methyl-1H-pyrrolo[3,2-b]pyridine", "pin_verified")


def _shuffled(smiles, seed):
    mol = Chem.MolFromSmiles(smiles)
    order = list(range(mol.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    return Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,tier", [
    ("O1COC2=C1C=CC=C2", "2H-1,3-benzodioxole", "pin"),
    ("O1CC=CC2=C1OC=CC2", "2H,5H-pyrano[2,3-b]pyran", "pin"),
    ("N=1CSC=C2C1C=CC=N2", "2H-pyrido[3,2-d][1,3]thiazine", "pin"),
    ("C1OC=C2C1=CC=C2", "1H-cyclopenta[c]furan", "pin"),
    (TARGET, TARGET_PIN, "pin"),
])
def test_the_name_does_not_depend_on_the_atom_order(smiles, name, tier):
    names = {_row(_shuffled(smiles, seed), tier).get("name") for seed in range(6)}
    assert names == {name}, names


def test_without_the_certified_source_the_plain_path_declines_the_target(monkeypatch):
    monkeypatch.setattr(fused_rings, "_certified_two_ring_parent", lambda mol, ring: None)
    assert fused_rings.name_fused_heterocycle(Chem.MolFromSmiles(TARGET)) is None
