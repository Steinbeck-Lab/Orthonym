"""LGA: a spiro ring system beside ring substituents is the parent, named as one ring system.

``2,2',7,7'-tetra(pyren-1-yl)-9,9'-spirobi[fluorene]`` was declined at the default tier: the namers of
the spiro systems read the whole molecule, a ring substituent added a ring to their count and a
ring-ring bond to their ring graph, and the parent was taken for pyrene (a 16-atom fused system
against the two fluorenes the spiro system was split into). The Blue Book:

* ``### **** MONOSPIRO RING SYSTEMS CONTAINING TWO IDENTICAL POLYCYCLIC COMPONENTS`` (the Blue Book),
   (:10146): 'spirobi' before the component name in square brackets, the locants of the
  spiro atom unprimed first: ``9,9'-spirobi[fluorene]``;
* ``**** The senior ring or ring system has the greater number of rings`` (:19461): the
  six rings of the spiro system against the four of pyrene;
* ``**** Polycyclic systems`` (:19532) "(a) spiro ring system" (:19536) before "(c) fused ring
  system": ``(1) 8-azaspiro[4.5]decane (PIN) > quinoline (PIN)`` (:19550);
  ``5-phenyl-1-oxa-6-thia-5λ5-phosphaspiro[4.4]nonane (PIN)`` (:46328), the spiro parent with a
  phenyl prefix;
* ``**** Parentheses`` (a) simple substituent prefixes having locants: ``tetra(naphthalen-2-yl)
  (preferred prefix)`` (:7085,:7090);
* ``****`` / ``### **** NUMBERING`` (:3219): the principal characteristic group first
  (c), then (f) "detachable alphabetized prefixes, all considered together" (:3301), then (g)
  "lowest locants for the substituent cited first as a prefix in the name" (:3307),
  ``1-methyl-4-nitronaphthalene (PIN) (not 4-methyl-1-nitronaphthalene)`` (:3318).
"""
import itertools
import random

import pytest
from rdkit import Chem, RDLogger

from orthonym import Orthonym
from orthonym.assembly import complex_ring_group as G
from orthonym.jvm_budget import jvm_slots
from orthonym.tiers import emit_tier_flags
from tests.support.rt_assert import assert_full_rt

RDLogger.DisableLog("rdApp.*")

SPIROBI = "c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc12"

#: row 4 of the user's 14 molecules: 2,2',7,7'-tetra(pyren-1-yl)-9,9'-spirobi[fluorene]
ROW_4 = ("C1(=CC=C2C=CC3=CC=CC4=CC=C1C2=C34)C3=CC=4C2(C1=CC(=CC=C1C4C=C3)C3=CC=C4C=CC1=CC=CC5=CC=C3C4=C15)"
         "C1=CC(=CC=C1C=1C=CC(=CC12)C1=CC=C2C=CC5=CC=CC4=CC=C1C2=C54)C5=CC=C4C=CC2=CC=CC1=CC=C5C4=C21")


# --------------------------------------------------------------------------- the helpers
def test_a_ring_substituent_is_its_own_ring_group():
    mol = Chem.MolFromSmiles("c1ccccc1-" + SPIROBI)
    assert sorted(len(g) for g in G.ring_groups(mol)) == [6, 25]


def test_the_two_halves_of_a_spiro_system_are_one_group():
    assert [len(g) for g in G.ring_groups(Chem.MolFromSmiles(SPIROBI))] == [25]


def test_identical_ring_groups_bonded_directly_are_one_ring_assembly_unit():
    # "Seniority criteria for ring assemblies are based on the appropriate criteria for rings
    # and ring systems given in " (the Blue Book): the two phenyls of biphenyl are one unit
    mol = Chem.MolFromSmiles("c1ccc(cc1)-c1ccc(cc1)C1CCC2(CC1)CCCC2")
    assert sorted(len(u) for u in G.assembly_units(mol, G.ring_groups(mol))) == [10, 12]


@pytest.mark.parametrize("smiles,sizes", [
    ("c1ccc(cc1)C1CCCCC1C1CC2(C1)CCC2", [6, 6, 7]),          # benzene and cyclohexane are not identical
    ("c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)C1CC2(C1)CCC2", [7, 18]),   # three phenyls in a chain
    ("c1ccc(cc1)-C1CC2(C1)CCC2", [6, 7]),                    # one phenyl is no assembly
])
def test_only_identical_groups_joined_directly_are_merged(smiles, sizes):
    mol = Chem.MolFromSmiles(smiles)
    assert sorted(len(u) for u in G.assembly_units(mol, G.ring_groups(mol))) == sizes


@pytest.mark.parametrize("smiles", [
    "c1ccc(cc1)-c1ccc(cc1)C1CCC2(CC1)CCCC2",           # biphenyl: 2 rings, 12 atoms > spiro[4.5]decane 10
    "c1ccc(cc1)-c1ccc(cc1)C1CC2(C1)CCC2",              # biphenyl 12 > spiro[3.3]heptane 7
    "c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)C1CC2(C1)CCC2",   # terphenyl: 3 rings
])
def test_a_ring_assembly_senior_to_the_spiro_system_leaves_it_no_core(smiles):
    # (d) "the greater number of rings" and (e) "the greater number of skeletal atoms" come
    # before the type list of (the book's 'bicyclo[6.2.1]undecane (PIN) > naphthalene (PIN)
    # [11 skeletal atoms > 10 skeletal atoms]' under,:19469,:19475): the assembly is the parent
    assert G.senior_spiro_core(Chem.MolFromSmiles(smiles)) is None


@pytest.mark.parametrize("smiles", [
    "c1ccc(cc1)-C1CC2(C1)CCC2",                  # 1 ring against 2 rings,:19461)
    "c1ccc2ccccc2c1C1CCC2(CC1)CCCC2",            # 2 rings, 10 atoms each: (a) spiro first
])
def test_a_single_ring_system_does_not_take_the_spiro_core_away(smiles):
    assert G.senior_spiro_core(Chem.MolFromSmiles(smiles)) is not None


def test_the_strictly_senior_spiro_system_is_cut_out_with_its_ring_atoms_only():
    mol = Chem.MolFromSmiles("c1ccccc1-" + SPIROBI)
    core = G.senior_spiro_core(mol)
    assert core is not None and len(core.to_full) == 25 == core.mol.GetNumAtoms()
    assert Chem.MolToSmiles(core.mol) == Chem.MolToSmiles(Chem.MolFromSmiles(SPIROBI))


@pytest.mark.parametrize("smiles,why", [
    ("c1ccccc1-c1ccccc1", "no spiro system"),
    ("C1CCC2(CC1)CCCC2", "one ring system"),
    ("c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccc(cc12)-c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc12",
     "two spiro systems that tie (the parent is chosen by the rules of ring assemblies, P-28.6)"),
    ("c1cc2ccc3cccc4ccc(c1)c2c34", "no spiro system"),
])
def test_no_core_where_the_spiro_system_is_not_the_one_senior_ring_system(smiles, why):
    assert G.senior_spiro_core(Chem.MolFromSmiles(smiles)) is None, why


def test_a_principal_characteristic_group_on_another_ring_system_outranks_the_ring_seniority():
    # before: benzoic acid is the parent although the spiro system is senior as a ring
    smiles = "OC(=O)c1ccc(cc1)C1CCC2(CC1)CCCC2"
    mol = Chem.MolFromSmiles(smiles)
    namer = Orthonym()
    features = namer._perceive(mol, smiles, Chem.CanonSmiles(smiles))
    namer._classify(features)
    assert features.principal_group_atoms
    assert G.senior_spiro_core(mol) is not None                 # senior without the group
    assert G.senior_spiro_core(mol, features) is None            # with the group, the benzene is the parent


# --------------------------------------------------------------------------- the names


@pytest.fixture(scope="module")
def namer():
    with jvm_slots(1, purpose="lga-test"):
        yield Orthonym(style="pin", **emit_tier_flags("pin"))   # emit_tier="pin": the PIN-only tier


@pytest.mark.opsin_gate
def test_row_4_is_named_as_the_pin_at_the_default_tier(namer):
    row = namer.name_tiered(ROW_4)
    assert (row["name"], row["tier"]) == (
        "2,2',7,7'-tetra(pyren-1-yl)-9,9'-spirobi[fluorene]", "pin_verified"), row
    assert_full_rt(row["name"], ROW_4)


@pytest.mark.opsin_gate
def test_row_4_is_the_pin_at_the_default_best_effort_tier_too():
    # the first run of the wider tiers asked the whole molecule whether it is a complex ring system and
    # spent its hang budget on a ring substituent taken for part of a cage: the name came from a later
    # producer in a non-PIN spelling ('2,7,2',7'-tetra(pyren-1-yl)spiro[fluorene-9,9'-fluorene]')
    with jvm_slots(1, purpose="lga-test"):
        row = Orthonym(style="pin", **emit_tier_flags("best-effort")).name_tiered(ROW_4)
    assert (row["name"], row["tier"]) == ("2,2',7,7'-tetra(pyren-1-yl)-9,9'-spirobi[fluorene]", "pin_verified"), row


SPIRO_PARENT = [
    ("c1ccccc1C1CC2(C1)CCC2", "2-phenylspiro[3.3]heptane"),
    ("c1ccccc1C1CCC2(CC1)CCCC2", "8-phenylspiro[4.5]decane"),
    ("c1ccc2ccccc2c1C1CCC2(CC1)CCCC2", "8-(naphthalen-1-yl)spiro[4.5]decane"),
    ("c1ccccc1-c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc12", "3-phenyl-9,9'-spirobi[fluorene]"),
    ("CC1(C)c2ccccc2-c2ccc(cc12)-c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc12",
     "3-(9,9-dimethyl-9H-fluoren-2-yl)-9,9'-spirobi[fluorene]"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", SPIRO_PARENT)
def test_the_spiro_system_is_the_parent_of_its_ring_substituent(namer, smiles, pin):
    row = namer.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", [
    ("c1ccc(cc1)-c1ccc(cc1)C1CCC2(CC1)CCCC2", "4-(spiro[4.5]decan-8-yl)-1,1'-biphenyl"),
    ("c1ccc(cc1)-c1ccc(cc1)C1CC2(C1)CCC2", "4-(spiro[3.3]heptan-2-yl)-1,1'-biphenyl"),
])
def test_a_ring_assembly_beside_a_spiro_system_is_never_named_as_a_substituent_of_it(namer, smiles, pin):
    # the PIN is the assembly with the spiro group as a prefix:20571, (e)); no
    # name is built for it yet, and the wrong one ('8-([1,1'-biphenyl]-4-yl)spiro[4.5]decane') is not
    # given: the row declines or carries the PIN
    row = namer.name_tiered(smiles)
    assert row["tier"] == "abstain" or (row["name"], row["tier"]) == (pin, "pin_verified"), row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", [
    ("OC(=O)c1ccc(cc1)C1CCC2(CC1)CCCC2", "4-(spiro[4.5]decan-8-yl)benzoic acid"),
    ("c1ccc(nc1)C1CCC2(CC1)CCCC2", "2-(spiro[4.5]decan-8-yl)pyridine"),
    ("C1(=CC=C2C=CC3=CC=CC4=CC=C1C2=C34)C12CC3CC(CC(C3)C1)C2", "1-(adamantan-1-yl)pyrene"),
])
def test_the_parent_stays_the_other_ring_where_the_rules_make_it_the_parent(namer, smiles, pin):
    # (the acid), (a) (a heterocycle), (four rings against two)
    row = namer.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row


# --------------------------------------------------------------------------- the numbering
SUBS = {"Br": ("Br", "bromo"), "Me": ("C", "methyl"), "Ph": ("c1ccccc1", "phenyl")}
LOCANTS = [f"{n}{p}" for p in ("", "'") for n in range(1, 9)]


def _attach(mol, idx, smiles):
    frag = Chem.MolFromSmiles(smiles)
    rw = Chem.RWMol(Chem.CombineMols(mol, frag))
    rw.AddBond(idx, mol.GetNumAtoms(), Chem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def _locant_key(loc):
    bare = loc.rstrip("'")
    return (int(bare), len(loc) - len(bare))


def _read_back_key(name):
    from orthonym.jars import find_jar
    from orthonym.jvm_bridge import opsin_stdout
    text, _ = opsin_stdout(name, False, str(find_jar("opsin")))
    mol = Chem.MolFromSmiles((text or "").strip())
    return Chem.MolToInchiKey(mol) if mol else None


def _expected_names(key_in, first, second):
    """Every numbering of the substituted spirobifluorene that OPSIN reads to the input, with the
    rules applied to the pair of locants: (f) the set, then (g) the locant of the substituent cited first."""
    valid = []
    for a, b in itertools.permutations(LOCANTS, 2):
        if first == second and _locant_key(a) > _locant_key(b):
            continue
        name = (f"{a},{b}-di{SUBS[first][1]}" if first == second
                else f"{a}-{SUBS[first][1]}-{b}-{SUBS[second][1]}") + "-9,9'-spirobi[fluorene]"
        if _read_back_key(name) == key_in:
            valid.append((a, b, name))

    def f_key(v):
        return sorted([_locant_key(v[0]), _locant_key(v[1])])

    def g_key(v):
        if first == second:
            return 0
        earlier = first if SUBS[first][1] < SUBS[second][1] else second
        return _locant_key(v[0] if earlier == first else v[1])

    best = min(valid, key=lambda v: (f_key(v), g_key(v)))
    return {v[2] for v in valid if (f_key(v), g_key(v)) == (f_key(best), g_key(best))}


@pytest.mark.opsin_gate
def test_the_lowest_locants_of_the_substituents_decide_the_numbering(namer):
    # the numbering is read independently: OPSIN says which numberings are valid, the rule picks one
    core = Chem.MolFromSmiles(SPIROBI)
    holes = [a.GetIdx() for a in core.GetAtoms() if a.GetIsAromatic() and a.GetTotalNumHs() == 1]
    rng = random.Random(3)
    checked = 0
    for i, j in rng.sample(list(itertools.combinations(holes, 2)), 5):
        for first, second in (("Br", "Ph"), ("Ph", "Ph")):
            mol = _attach(_attach(core, i, SUBS[first][0]), j, SUBS[second][0])
            smiles = Chem.MolToSmiles(mol)
            row = namer.name_tiered(smiles)
            assert row["tier"] == "pin_verified", row
            assert row["name"] in _expected_names(Chem.MolToInchiKey(mol), first, second), (smiles, row)
            checked += 1
    assert checked == 10


@pytest.mark.opsin_gate
def test_the_name_does_not_depend_on_the_order_of_the_atoms_of_the_input(namer):
    mol = Chem.MolFromSmiles("Cc1ccc(cc1)-c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc12")
    names = {namer.name_tiered(Chem.MolToSmiles(mol, doRandom=True, canonical=False))["name"] for _ in range(8)}
    assert names == {"3-(4-methylphenyl)-9,9'-spirobi[fluorene]"}
